# CUDA Kernel 三课全记录

> 教学背景：以 Perplexity AI 生产代码 `a2a_dispatch_send.cu` 为蓝本，
> 学习真实工业级 CUDA Kernel 的核心技术。
>
> 代码文件：`pplx-garden/p2p-all-to-all/a2a-kernels/src/a2a/a2a_dispatch_send.cu`

---

## 目录

- [第一课：Grid-Stride Loop + `__ldg` + `atomicAdd`](#第一课)
- [第二课：Warp Shuffle Prefix Sum](#第二课)
- [第三课：`uint4` 向量化 + NVLink P2P + 寄存器复用](#第三课)
- [综合练习 Kernel：mini_scatter_kernel.cu](#综合练习-kernel)

---

## 第一课

### Grid-Stride Loop + `__ldg` + `atomicAdd`

#### 背景：kernel 第一阶段做什么

Block 0 负责统计每个 expert 收到多少个 token，并给每个 `(token, k)` 分配唯一的 offset：

```cpp
// 生产代码片段（block 0 内部）
uint32_t *tokens_per_expert = (uint32_t*)shared_memory;

// 初始化：Stride Loop 清零
for (uint32_t i = threadIdx.x; i < num_experts; i += blockDim.x) {
    tokens_per_expert[i] = 0;
}
__syncthreads();

// 统计：每个 (token, k) 读出 expert id，原子加计数
for (uint32_t i = threadIdx.x; i < num_send_tokens * num_experts_per_token_bound; i += blockDim.x) {
    const uint32_t token = i / num_experts_per_token_bound;
    const uint32_t index = i % num_experts_per_token_bound;
    const uint32_t expert = __ldg(&indices[token * indices_stride + index]);
    token_offset[i] = atomicAdd(&tokens_per_expert[expert], 1);
}
```

---

#### 技术点 1：Grid-Stride Loop（跨步循环）

```
起点 = threadIdx.x      步长 = blockDim.x
```

每个线程从自己的 index 出发，每次跳一个 block 宽度，直到越界。

**示例**（`blockDim.x = 4`，`num_experts = 8`）：

| thread | 第1轮 i= | 第2轮 i= | 负责的格子 |
|--------|---------|---------|----------|
| 0      | 0       | 4       | [0], [4] |
| 1      | 1       | 5       | [1], [5] |
| 2      | 2       | 6       | [2], [6] |
| 3      | 3       | 7       | [3], [7] |

**为什么要用 Stride Loop，不用 `arr[threadIdx.x] = 0`？**

| 方式 | 问题 |
|------|------|
| `arr[threadIdx.x]` | 只能处理 `num_experts <= blockDim.x`，超出越界 |
| Stride Loop | 无论数组多长都正确，thread 不够就多跑几轮 |

**关键优点**：
- 无 warp divergence：所有线程执行次数相近
- 访问连续（coalesced）：相邻线程访问相邻地址 → GPU 缓存友好
- 负载均衡：每个线程承担相同工作量

---

#### 技术点 2：`__ldg`（只读缓存加载）

```cpp
const uint32_t expert = __ldg(&indices[token * indices_stride + index]);
```

`__ldg` = Load through Global (Texture) Cache

| | 普通 `*ptr` | `__ldg(ptr)` |
|--|--|--|
| 走的缓存 | L1/L2 | Texture Cache（只读路径）|
| 适用条件 | 可读写数据 | 整个 kernel 运行期间不会被修改的数据 |
| 性能 | 标准 | 对反复读同一地址更高效，减少 L1 压力 |

`indices` 数组在这个 kernel 里只读不写 → 用 `__ldg` 可以提示编译器走只读路径。

---

#### 技术点 3：`atomicAdd`（原子操作分配唯一 offset）

```cpp
token_offset[i] = atomicAdd(&tokens_per_expert[expert], 1);
```

**读-改-写原子性**：多个线程同时对同一个 expert 计数，`atomicAdd` 保证每个线程得到唯一的返回值（0, 1, 2, ...），不会有两个线程得到同一个 offset。

**为什么用 shared memory，不用全局内存 atomic？**

- `tokens_per_expert` 在 shared memory 里 → 同一 block 内的 atomic 比全局内存快 10-20x
- shared memory 在 SM 片上，延迟约 20 cycle；全局内存 atomic 延迟约 200-400 cycle

---

### 第一课作业与解答

**Q1：如果把 Stride Loop 改成 `if (threadIdx.x < num_experts) tokens_per_expert[threadIdx.x] = 0;`，什么情况下会出错？**

A：当 `num_experts > blockDim.x` 时，index 超过 `blockDim.x - 1` 的格子永远不会被清零。例如 `num_experts = 256`，`blockDim.x = 128`，则 `[128..255]` 全部留着垃圾值，后续 `atomicAdd` 结果错误。

---

**Q2：`atomicAdd` 返回的是加之前的旧值还是新值？为什么这里要用旧值？**

A：`atomicAdd(ptr, val)` 返回**旧值**（加法前）。这里用旧值是因为：第 0 个到达 expert E 的 token 想要 offset=0，第 1 个想要 offset=1……返回旧值就能给每个 token 分配唯一编号，等价于"排队取号"。如果返回新值，第一个 token 会得到 offset=1，偏差 1。

---

**Q3：`__syncthreads()` 在清零和统计之间的作用是什么？如果去掉会怎样？**

A：`__syncthreads()` 确保所有线程的清零操作完成后，才开始统计阶段。如果去掉：线程 A 可能还没把 `tokens_per_expert[3]` 清零，线程 B 就已经对 `tokens_per_expert[3]` 执行 `atomicAdd`——在垃圾值上累加，计数从非零值开始，offset 计算全部错乱。

---

## 第二课

### Warp Shuffle Prefix Sum

#### 背景：prefix sum 做什么

Phase 1 统计完 `tokens_per_expert[0..N-1]` 后，需要算 `expert_offsets`：

```
输入：tokens_per_expert = [3, 2, 5, 1, ...]
输出：expert_offsets    = [3, 5, 10, 11, ...]  ← 累计和（inclusive prefix sum）
```

`expert_offsets[e]` 就是"expert 0 到 e 一共接收了多少个 token"，
用于计算 scatter 时每个 token 在输出 buffer 里的绝对位置。

---

#### 技术点：`__shfl_up_sync` Warp 内 Prefix Sum

```cpp
// 每个线程持有自己的 expert 计数
uint32_t expert_offset = (i < num_experts) ? tokens_per_expert[i] : 0;

// Warp 内上扫描（inclusive prefix sum）
for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    unsigned warp_sum_expert = __shfl_up_sync(0xFFFFFFFF, expert_offset, offset);
    if (lane_id >= offset) {
        expert_offset += warp_sum_expert;
    }
}
```

**`__shfl_up_sync(mask, val, delta)`**：
- 每个 lane 读取**自己左边 delta 位的 lane** 的 `val`
- 不走 shared memory，直接 warp 内寄存器交换 → 极低延迟
- `mask = 0xFFFFFFFF` 表示 warp 里所有 32 个 lane 参与

**5 轮迭代后（offset = 1, 2, 4, 8, 16）**：

```
初始：  [a0, a1, a2, a3, a4, ...]
offset=1后：[a0, a0+a1, a1+a2, a2+a3, ...]
offset=2后：[a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, ...]
...
最终：  lane k 持有 a[0] + a[1] + ... + a[k]  ← inclusive prefix sum
```

O(log 32) = 5 步完成，比串行 O(32) 快 6x。

---

#### 跨 Warp 累加（两层 prefix sum）

```cpp
// Step 1：每个 Warp 完成内部 prefix sum
// → lane 31 持有该 Warp 所有元素的总和（就是该 Warp 的完整 prefix sum 的最后一项）

// Step 2：lane 31 把 Warp 总和存入 shared memory
if (lane_id == WARP_SIZE - 1) {
    expert_sums[warp_id] = expert_offset;
}
__syncthreads();

// Step 3：Warp 0 对 warp_sums 再做一次 prefix sum（跨 Warp）
if (warp_id == 0) {
    uint32_t total_expert_sum = (lane_id < num_warps) ? expert_sums[lane_id] : 0;
    for (int offset = 1; offset < num_warps; offset <<= 1) {
        unsigned warp_sum = __shfl_up_sync(0xFFFFFFFF, total_expert_sum, offset);
        if (lane_id >= offset) {
            total_expert_sum += warp_sum;
        }
    }
    if (lane_id < num_warps) {
        expert_sums[lane_id] = total_expert_sum;  // Warp i 之前所有 expert 的总和
    }
}
__syncthreads();

// Step 4：每个线程加上所在 Warp 之前的偏移量
if (i < num_experts) {
    expert_offsets[i] = (warp_id > 0)
        ? expert_sums[warp_id - 1] + expert_offset
        : expert_offset;
}
```

---

### 第二课作业与解答

**Q：为什么保存 Warp 总和要选 `lane_id == 31`（WARP_SIZE - 1），不选 lane 0？**

A：Warp Shuffle Prefix Sum 是**上扫描（inclusive scan）**，运算完成后：
- `lane 0` 只持有自己一个元素 `a[0]`
- `lane k` 持有 `a[0] + a[1] + ... + a[k]`
- `lane 31` 持有 `a[0] + ... + a[31]` = **整个 Warp 所有元素的总和**

需要的是 Warp 总和来做跨 Warp 累加，所以必须选 lane 31。选 lane 0 只会得到第一个元素，不是总和。

---

## 第三课

### `uint4` 向量化 + NVLink P2P + 寄存器复用（Load Once, Reuse）

#### 背景：Phase 2 做什么

Phase 1 算好了每个 expert 的 prefix sum（`expert_offsets`）。
Phase 2 的任务：把每个 token 的数据（hidden_dim 个元素）scatter 写到 K 个目标位置。

---

#### 技术点 1：`uint4` 向量化读写（16 bytes / 指令）

```cpp
uint4 *x_token_src = (uint4*)(x_ptr + token * x_stride);

// 每次读 16 字节（4 个 float），而不是 1 个 float
uint4 val = ld_global_nc_uint4(&x_token_src[i]);

// 写到目标位置
st_global_nc_uint4(&x_token_dst[i], val);
```

**为什么用 `uint4`？**

| 方式 | 每条指令传输量 | 总指令数（hidden=4096 float = 16KB） |
|------|-------------|-------------------------------------|
| float（4B）| 4 B | 4096 条 |
| float4 / uint4（16B）| 16 B | 1024 条 |

指令数减少 4x → 内存带宽利用率提升 → throughput 更高。

**地址对齐要求**：指针必须是 16 的倍数才能用 `uint4` 指令，这就是 CPU 端要 `round_up(hidden_dim * elemsize, 16)` 的原因。

---

#### 技术点 2：寄存器复用（Load Once, Reuse in Registers）

```cpp
// token_dim 在编译期已知（Fixed<N>） → NUM_STEPS 是 constexpr
constexpr size_t TOKEN_DIM = TokenDimTy::Value;
constexpr size_t NUM_STEPS = (TOKEN_DIM + NUM_THREADS - 1) / NUM_THREADS;

// 一次性把整个 token 读进寄存器
uint4 vals[NUM_STEPS];   // NUM_STEPS 是编译期常量 → 编译器把这个数组展开成寄存器变量
float scales[NUM_STEPS];

#pragma unroll(NUM_STEPS)
for (unsigned i = threadIdx.x, s = 0; i * sizeof(uint4) < TOKEN_DIM; i += NUM_THREADS, s++) {
    vals[s] = ld_global_nc_uint4(&x_token_src[i]);   // 读 1 次
}

// 对 K 个目标 expert 写出（从寄存器复用，不重读全局内存）
#pragma unroll
for (unsigned e = 0; e < num_experts_per_token_bound; e++) {
    // ...计算目标地址...
    for (unsigned i = threadIdx.x, s = 0; ...; i += NUM_THREADS, s++) {
        st_global_nc_uint4(&x_token_dst[i], vals[s]);  // 写 K 次，但不需要重读
    }
}
// 总结：读全局内存 1 次，写 K 次；比朴素实现（读 K 次，写 K 次）节省 (K-1)/K 读带宽
```

**为什么 `vals[]` 能放寄存器？**

- `NUM_STEPS` 是 `constexpr`（编译期常量）→ 数组大小编译期已知
- 编译器知道大小的固定数组，默认分配在寄存器（register file）
- 如果 `NUM_STEPS` 是运行时变量，数组必须分配在 local memory（实际是 L2 缓存 spill）

| | `constexpr NUM_STEPS` | 运行时 NUM_STEPS |
|--|--|--|
| `vals[]` 位置 | 寄存器（片上，0 延迟） | local memory（L2 spill，~200 cycle 延迟）|
| `#pragma unroll` | 完全展开，零分支 | 无法展开 |

---

#### 技术点 3：NVLink P2P 直写

```cpp
// 如果目标 GPU 在同一节点内，走 NVLink 直写 recv buffer
if (dst_node == node_rank && dst_rank != rank && route.offset < max_private_tokens) {
    const uint32_t local_peer = dst_rank % NODE_SIZE;
    std::byte *token_ptr = recv_ptrs[local_peer] + (node_group * max_private_tokens + route.offset) * token_stride;
    uint4 *x_token_dst = (uint4*)token_ptr;
    st_global_nc_uint4(&x_token_dst[i], val);  // 直接写到另一块 GPU 的 VRAM
}
```

**`recv_ptrs[local_peer]`** 是什么？

- 通过 `cudaIpcGetMemHandle` 或 NVLink peer mapping 得到的**另一块 GPU 的 VRAM 地址**
- 当前 GPU 可以直接用普通的 store 指令写到这个地址 → NVLink 硬件自动把数据传到对端 GPU
- `st_global_nc_uint4`（Non-Coherent）：跳过 L1 缓存，直接 flush 到 NVLink 总线 → 对端 GPU 立刻可见

**vs 走 send buffer 的路径**：

| 路径 | 目标 | 延迟 | 额外拷贝 |
|------|------|------|---------|
| NVLink 直写 | 同节点内其他 GPU | 低（硬件直连） | 无 |
| send_buffer | 跨节点 / 本地 rank | 高（走 RDMA/PCIe） | 需要额外拷贝操作 |

---

#### TokenDimTy 类型系统：Fixed<N> vs NotFixed

```cpp
// Fixed<N>：编译期已知，忽略传入的运行时值
template<size_t N>
struct Fixed {
    Fixed(size_t) {}                           // 忽略参数
    operator size_t() const { return N; }
    static constexpr size_t Value = N;
};

// NotFixed：运行时读取
struct NotFixed {
    size_t value_;
    NotFixed(size_t v) : value_(v) {}
    operator size_t() const { return value_; }
};
```

**LAUNCH 宏运行时分发**：

```cpp
if (token_dim == 8192) {
    using TokenDim = Fixed<8192>;
    // 启动 kernel，TOKEN_DIM 是编译期常量 → 寄存器数组 + 完全 unroll
} else {
    using TokenDim = NotFixed;
    // 慢路径，运行时判断边界
}
```

#### QUICK vs 非 QUICK

```cpp
if (num_blocks >= num_tokens) {
    // QUICK=true：每个 block 处理 1 个 token，最高并行度，不用 grid-stride loop
} else {
    // QUICK=false：每个 block 循环处理多个 token（grid-stride loop）
}
```

**两个维度叠加 → 4 种 kernel 变体**：

| | QUICK=true | QUICK=false |
|--|--|--|
| Fixed\<N\> | 最快路径（寄存器 + 1:1 并行）| 循环 + 寄存器 |
| NotFixed | 1:1 并行，无 unroll | 慢路径 |

---

#### NUM_STEPS 公式详解

$$\text{NUM\_STEPS} = \left\lceil \frac{\text{TOKEN\_DIM} / \text{sizeof(uint4)}}{\text{NUM\_THREADS}} \right\rceil$$

代码写法（向上取整技巧）：

```cpp
constexpr size_t NUM_STEPS = (TOKEN_DIM + NUM_THREADS - 1) / NUM_THREADS;
// 普通除法 7/3=2（丢余数），向上取整 (7+2)/3=3（正确）
```

示例：
```
TOKEN_DIM=8192, NUM_THREADS=512, sizeof(uint4)=16
uint4 块总数 = 8192/16 = 512
NUM_STEPS = ceil(512/512) = 1    ← 每个线程读 1 个 uint4

TOKEN_DIM=16384, NUM_THREADS=512
uint4 块总数 = 16384/16 = 1024
NUM_STEPS = ceil(1024/512) = 2   ← 每个线程读 2 个 uint4
```

---

### 第三课作业与解答

**Q1：为什么 `vals[]` 数组能放进寄存器，而不是 shared memory 或 global memory？**

A：`NUM_STEPS` 是 `constexpr`（编译期常量），编译器知道数组大小 → 可以把每个元素映射到一个固定的寄存器（就像 `int a, b` 那样展开）。如果 `NUM_STEPS` 是运行时变量，编译器不知道要分配多少个寄存器，只能把数组放在 local memory（实际存在 L2 里，延迟 ~200 cycle）。这就是为什么 `Fixed<N>` 比 `NotFixed` 快：前者的 `vals` 在 0 延迟的寄存器里，后者在高延迟的 local memory 里。

---

**Q2：`st_global_nc_uint4` 里的 `nc` 是什么意思？为什么写 NVLink 时要用这个而不是普通 store？**

A：`nc` = Non-Coherent（非一致性）。普通 store 会先写进本 GPU 的 L1 cache；L1 是 per-SM 的，对端 GPU 看不到 L1 里的数据，只能看到最终 flush 到显存的数据。`st_global_nc` 直接绕过 L1，把数据 flush 到互联总线（NVLink）→ 对端 GPU 能立即看到写入结果，不需要等待额外的缓存一致性操作。

---

## 综合练习 Kernel

### `mini_scatter_kernel.cu`

> 编译：`nvcc -O2 -arch=sm_80 mini_scatter_kernel.cu -o mini_scatter`

融合三课所有核心技术：
- Phase 1（第一课 + 第二课）：Grid-Stride Loop + `__ldg` + `atomicAdd` + Warp Shuffle Prefix Sum
- Phase 2（第三课）：`uint4` 向量化读 + 寄存器复用（Load Once, Reuse）

```cuda
/**
 * mini_scatter_kernel.cu
 * 编译：nvcc -O2 -arch=sm_80 mini_scatter_kernel.cu -o mini_scatter
 */

#include <cuda.h>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cassert>

constexpr int WARP_SIZE      = 32;
constexpr int NUM_EXPERTS    = 8;
constexpr int K              = 2;    // 每 token 路由到 K 个 expert
constexpr int TOKEN_DIM      = 256;  // 每 token 256 个 float = 1024 bytes
constexpr int TOKEN_DIM_U4   = TOKEN_DIM / 4;   // = 64 个 uint4
constexpr int NUM_THREADS    = 32;               // Phase 2 每 Block 线程数
constexpr int NUM_STEPS      = TOKEN_DIM_U4 / NUM_THREADS;  // = 2（每线程处理 2 个 uint4）

// ── Phase 1：计数 + Prefix Sum ──
__global__ void phase1_kernel(
    const int32_t* __restrict__ indices,
    int N,
    uint32_t* __restrict__ token_offset,
    uint32_t* __restrict__ expert_offsets
) {
    extern __shared__ uint32_t tokens_per_expert[];

    // 第一课：Stride Loop 清零
    for (int i = threadIdx.x; i < NUM_EXPERTS; i += blockDim.x)
        tokens_per_expert[i] = 0;
    __syncthreads();

    // 第一课：__ldg + atomicAdd 分配 offset
    for (int i = threadIdx.x; i < N * K; i += blockDim.x) {
        int expert = __ldg(&indices[i]);
        token_offset[i] = atomicAdd(&tokens_per_expert[expert], 1);
    }
    __syncthreads();

    // 第二课：Warp Shuffle Prefix Sum
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = (NUM_EXPERTS + WARP_SIZE - 1) / WARP_SIZE;

    uint32_t val = (threadIdx.x < NUM_EXPERTS) ? tokens_per_expert[threadIdx.x] : 0;

    // Warp 内 inclusive prefix sum
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        uint32_t neighbour = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (lane_id >= offset)
            val += neighbour;
    }

    __shared__ uint32_t warp_sums[32];
    if (lane_id == WARP_SIZE - 1)       // lane 31 持有 Warp 完整总和
        warp_sums[warp_id] = val;
    __syncthreads();

    // Warp 0 对 warp_sums 做跨 Warp prefix sum
    if (warp_id == 0) {
        uint32_t wsum = (lane_id < num_warps) ? warp_sums[lane_id] : 0;
        for (int offset = 1; offset < num_warps; offset <<= 1) {
            uint32_t n = __shfl_up_sync(0xFFFFFFFF, wsum, offset);
            if (lane_id >= offset) wsum += n;
        }
        if (lane_id < num_warps)
            warp_sums[lane_id] = wsum;
    }
    __syncthreads();

    if (threadIdx.x < NUM_EXPERTS) {
        expert_offsets[threadIdx.x] = (warp_id > 0)
            ? warp_sums[warp_id - 1] + val
            : val;
    }
}

// ── Phase 2：uint4 向量化 scatter + 寄存器复用 ──
__global__ void phase2_kernel(
    const uint4*    __restrict__ x,
    const int32_t*  __restrict__ indices,
    const uint32_t* __restrict__ token_offset,
    const uint32_t* __restrict__ expert_offsets,
    uint4*          __restrict__ output,
    int N
) {
    for (int token = blockIdx.x; token < N; token += gridDim.x) {
        const uint4* src = x + (size_t)token * TOKEN_DIM_U4;

        // 第三课：一次性读入寄存器（NUM_STEPS 是 constexpr → 展开为寄存器变量）
        uint4 vals[NUM_STEPS];
        #pragma unroll
        for (int s = 0; s < NUM_STEPS; s++)
            vals[s] = src[threadIdx.x + s * NUM_THREADS];

        // 对 K 个目标写出（从寄存器，不重读全局内存）
        #pragma unroll
        for (int k = 0; k < K; k++) {
            int expert      = __ldg(&indices[token * K + k]);
            uint32_t offset = token_offset[token * K + k];
            uint32_t base   = (expert > 0) ? expert_offsets[expert - 1] : 0;
            uint4* dst      = output + (size_t)(base + offset) * TOKEN_DIM_U4;

            #pragma unroll
            for (int s = 0; s < NUM_STEPS; s++)
                dst[threadIdx.x + s * NUM_THREADS] = vals[s];
        }
        // 总计：全局内存读 1 次，写 K 次（朴素写法：读 K 次，写 K 次）
    }
}

int main() {
    const int N = 64;
    const int total_output_tokens = N * K;

    int32_t h_indices[N * K];
    float   h_x[N * TOKEN_DIM];

    for (int i = 0; i < N; i++) {
        h_indices[i * K + 0] = i % NUM_EXPERTS;
        h_indices[i * K + 1] = (i + 1) % NUM_EXPERTS;
        for (int d = 0; d < TOKEN_DIM; d++)
            h_x[i * TOKEN_DIM + d] = (float)(i * TOKEN_DIM + d);
    }

    int32_t  *d_indices;
    uint4    *d_x, *d_output;
    uint32_t *d_token_offset, *d_expert_offsets;

    cudaMalloc(&d_indices,        N * K * sizeof(int32_t));
    cudaMalloc(&d_x,              N * TOKEN_DIM_U4 * sizeof(uint4));
    cudaMalloc(&d_output,         total_output_tokens * TOKEN_DIM_U4 * sizeof(uint4));
    cudaMalloc(&d_token_offset,   N * K * sizeof(uint32_t));
    cudaMalloc(&d_expert_offsets, NUM_EXPERTS * sizeof(uint32_t));

    cudaMemcpy(d_indices, h_indices, N * K * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * TOKEN_DIM * sizeof(float), cudaMemcpyHostToDevice);

    phase1_kernel<<<1, 128, NUM_EXPERTS * sizeof(uint32_t)>>>(
        d_indices, N, d_token_offset, d_expert_offsets
    );

    phase2_kernel<<<min(N, 256), NUM_THREADS>>>(
        d_x, d_indices, d_token_offset, d_expert_offsets, d_output, N
    );

    cudaDeviceSynchronize();

    uint32_t h_expert_offsets[NUM_EXPERTS];
    cudaMemcpy(h_expert_offsets, d_expert_offsets,
               NUM_EXPERTS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("Expert offsets (prefix sum):\n");
    for (int e = 0; e < NUM_EXPERTS; e++)
        printf("  expert_offsets[%d] = %u\n", e, h_expert_offsets[e]);
    printf("\n✅ Total tokens routed: %u (expected %d)\n",
           h_expert_offsets[NUM_EXPERTS - 1], N * K);

    cudaFree(d_indices);
    cudaFree(d_x);
    cudaFree(d_output);
    cudaFree(d_token_offset);
    cudaFree(d_expert_offsets);
    return 0;
}
```

---

### 练习题（自测用）

1. **改 `K=3`**：验证寄存器复用对 K 的线性扩展性，性能曲线是否保持线性？
2. **把 `NUM_STEPS` 改成运行时变量**：观察 `vals[]` spill 到 local memory，nsight 里 L2 Read 指标上升。
3. **在 Phase 2 加一个 `__syncthreads()`**：思考为什么影响性能但不影响正确性？（提示：Phase 2 每个 block 只处理一个 token，内部无共享状态）

---

## 三课技术速查表

| 技术 | 代码片段 | 目的 |
|------|---------|------|
| Grid-Stride Loop | `for (i = threadIdx.x; i < N; i += blockDim.x)` | 任意大小数组均可并行处理 |
| `__ldg` | `__ldg(&ptr[i])` | 只读缓存，减少 L1 压力 |
| Shared Memory Atomic | `atomicAdd(&smem[expert], 1)` | 分配唯一 offset，比全局 atomic 快 10-20x |
| `__syncthreads()` | 写后同步 | 保证 block 内所有线程看到同一状态 |
| Warp Shuffle (up) | `__shfl_up_sync(0xFFFFFFFF, val, offset)` | O(log 32) warp 内 prefix sum，零 shared memory |
| Lane 31 存 Warp Sum | `if (lane_id == 31) warp_sums[warp_id] = val` | 因为 lane 31 持有 inclusive prefix sum 的最后一项 = 总和 |
| `uint4` 向量化 | `uint4 val = ld_global_nc_uint4(ptr)` | 16B/指令，减少 4x 内存指令数 |
| 寄存器复用 | `uint4 vals[constexpr N]` + `#pragma unroll` | 读 1 次写 K 次，节省 (K-1)/K 读带宽 |
| `st_global_nc` | `st_global_nc_uint4(ptr, val)` | 绕过 L1，直写 NVLink 总线，对端 GPU 立即可见 |
| `Fixed<N>` | `template<size_t N> struct Fixed` | 把运行时值变编译期常量，解锁寄存器数组 + 完全 unroll |

---

*参考文件：[a2a_dispatch_send.cu](../pplx-garden/p2p-all-to-all/a2a-kernels/src/a2a/a2a_dispatch_send.cu) | [mini_scatter_kernel.cu](mini_scatter_kernel.cu)*
