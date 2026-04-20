/**
 * mini_scatter_kernel.cu
 *
 * 教学目标：融合三课所有核心技术
 *   第一课：Grid-Stride Loop, __ldg, atomicAdd (shared memory)
 *   第二课：Warp Shuffle Prefix Sum (__shfl_up_sync)
 *   第三课：uint4 向量化读写, 寄存器复用 (Load Once, Reuse)
 *
 * 任务：
 *   给定 N 个 token，每个 token 路由到 K 个 expert。
 *   把每个 token 的数据（TOKEN_DIM 个 float）
 *   scatter 写入对应 expert 的输出 buffer。
 *
 * 编译：
 *   nvcc -O2 -arch=sm_80 mini_scatter_kernel.cu -o mini_scatter
 */

#include <cuda.h>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cassert>

// ─────────────────────────────────────────────
// 编译期常量（工业级 Kernel 的关键：让编译器展开循环）
// ─────────────────────────────────────────────
constexpr int WARP_SIZE      = 32;
constexpr int NUM_EXPERTS    = 8;    // expert 总数
constexpr int K              = 2;    // 每个 token 路由到 K 个 expert
constexpr int TOKEN_DIM      = 256;  // 每个 token 256 个 float = 1024 bytes
constexpr int TOKEN_DIM_U4   = TOKEN_DIM / 4;  // = 64 个 uint4（每个 uint4 = 16 bytes）
constexpr int NUM_THREADS    = 32;   // Phase 2 每个 Block 的线程数（一个 Warp）
// 每个线程负责的 uint4 块数：64 / 32 = 2
constexpr int NUM_STEPS      = TOKEN_DIM_U4 / NUM_THREADS;  // = 2

// ─────────────────────────────────────────────
// Phase 1 Kernel
// 目标：计算 expert_offsets（prefix sum）和每个 (token, k) 的 offset
// 技术：第一课 + 第二课
// ─────────────────────────────────────────────
__global__ void phase1_kernel(
    const int32_t* __restrict__ indices,    // [N * K] 路由 indices，indices[token*K+k] = expert_id
    int N,
    uint32_t* __restrict__ token_offset,    // 输出 [N * K]：每个 (token,k) 在其 expert 内的编号
    uint32_t* __restrict__ expert_offsets   // 输出 [NUM_EXPERTS]：prefix sum 结果
) {
    // ── 第一课：用 shared memory 计数，避免全局 atomic 竞争 ──
    extern __shared__ uint32_t tokens_per_expert[];  // [NUM_EXPERTS]

    // 初始化 shared memory
    for (int i = threadIdx.x; i < NUM_EXPERTS; i += blockDim.x)
        tokens_per_expert[i] = 0;
    __syncthreads();

    // Grid-Stride Loop：每个线程处理多个 (token, k) 对
    for (int i = threadIdx.x; i < N * K; i += blockDim.x) {
        int expert = __ldg(&indices[i]);              // 第一课：__ldg 只读缓存
        token_offset[i] = atomicAdd(                  // 第一课：原子加，分配唯一 offset
            &tokens_per_expert[expert], 1
        );
    }
    __syncthreads();

    // ── 第二课：Warp Shuffle Prefix Sum ──
    // 目标：把 tokens_per_expert[0..NUM_EXPERTS-1] 变成 prefix sum → expert_offsets

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = (NUM_EXPERTS + WARP_SIZE - 1) / WARP_SIZE;

    // 每个线程负责一个 expert 的计数
    uint32_t val = (threadIdx.x < NUM_EXPERTS) ? tokens_per_expert[threadIdx.x] : 0;

    // Step A：Warp 内 prefix sum（上扫描）
    // offset = 1, 2, 4, 8, 16 → 共 5 轮，O(log32) 完成
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        uint32_t neighbour = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (lane_id >= offset)
            val += neighbour;
    }
    // 此时 val = expert[0..threadIdx.x] 的累计和

    // Step B：保存每个 Warp 的总和（lane 31 持有整个 Warp 的 sum）
    __shared__ uint32_t warp_sums[32];
    if (lane_id == WARP_SIZE - 1)           // 第二课关键：只有 lane 31 持有完整总和
        warp_sums[warp_id] = val;
    __syncthreads();

    // Step C：Warp 0 对 warp_sums 再做一次 prefix sum（跨 Warp）
    if (warp_id == 0) {
        uint32_t wsum = (lane_id < num_warps) ? warp_sums[lane_id] : 0;
        for (int offset = 1; offset < num_warps; offset <<= 1) {
            uint32_t n = __shfl_up_sync(0xFFFFFFFF, wsum, offset);
            if (lane_id >= offset) wsum += n;
        }
        if (lane_id < num_warps)
            warp_sums[lane_id] = wsum;  // 写回：Warp i 之前所有 expert 的总和
    }
    __syncthreads();

    // Step D：每个线程加上前面所有 Warp 的基础偏移 → 最终 prefix sum
    if (threadIdx.x < NUM_EXPERTS) {
        expert_offsets[threadIdx.x] = (warp_id > 0)
            ? warp_sums[warp_id - 1] + val
            : val;
    }
}

// ─────────────────────────────────────────────
// Phase 2 Kernel
// 目标：把每个 token 的数据 scatter 到对应 expert 的位置
// 技术：第三课（uint4 + 寄存器复用）
// ─────────────────────────────────────────────
__global__ void phase2_kernel(
    const uint4*    __restrict__ x,              // 输入 [N, TOKEN_DIM_U4]
    const int32_t*  __restrict__ indices,        // [N * K]
    const uint32_t* __restrict__ token_offset,   // [N * K]
    const uint32_t* __restrict__ expert_offsets, // [NUM_EXPERTS]
    uint4*          __restrict__ output,         // 输出 [total_tokens, TOKEN_DIM_U4]
    int N
) {
    // Grid-Stride Loop：一个 Block 处理一个 token
    for (int token = blockIdx.x; token < N; token += gridDim.x) {

        const uint4* src = x + (size_t)token * TOKEN_DIM_U4;

        // ── 第三课：一次性读入寄存器（Load Once, Reuse in Registers）──
        // NUM_STEPS 是 constexpr → 编译器把 vals[] 展开成寄存器变量
        uint4 vals[NUM_STEPS];

        #pragma unroll
        for (int s = 0; s < NUM_STEPS; s++) {
            // 第三课：uint4 向量化读（16 bytes / 条指令）
            // threadIdx.x + s * NUM_THREADS 保证 coalesced 访问
            vals[s] = src[threadIdx.x + s * NUM_THREADS];
        }
        // 此时整个 token 的数据已在寄存器里，不再需要读全局内存

        // 对 K 个目标 expert 写出（从寄存器写，不需要重读全局内存）
        #pragma unroll
        for (int k = 0; k < K; k++) {
            int expert = __ldg(&indices[token * K + k]);
            uint32_t offset = token_offset[token * K + k];

            // expert 在 output buffer 里的起始位置（用 prefix sum 结果）
            uint32_t base = (expert > 0) ? expert_offsets[expert - 1] : 0;
            uint32_t dst_pos = base + offset;

            uint4* dst = output + (size_t)dst_pos * TOKEN_DIM_U4;

            // 从寄存器 vals[] 写到目标（零额外全局内存读）
            #pragma unroll
            for (int s = 0; s < NUM_STEPS; s++) {
                dst[threadIdx.x + s * NUM_THREADS] = vals[s];
            }
        }
        // 总计：每个 token 读全局内存 1 次，写 K 次
        // 对比朴素写法：读 K 次，写 K 次 → 节省 (K-1)/K 的读带宽
    }
}

// ─────────────────────────────────────────────
// CPU 测试驱动
// ─────────────────────────────────────────────
int main() {
    const int N = 64;  // 64 个 token
    const int total_output_tokens = N * K;  // 每个 token 复制到 K 个地方

    // 分配 host 内存
    int32_t h_indices[N * K];
    float   h_x[N * TOKEN_DIM];

    // 初始化：token i 的路由到 expert (i % NUM_EXPERTS) 和 (i+1) % NUM_EXPERTS
    for (int i = 0; i < N; i++) {
        h_indices[i * K + 0] = i % NUM_EXPERTS;
        h_indices[i * K + 1] = (i + 1) % NUM_EXPERTS;
        for (int d = 0; d < TOKEN_DIM; d++)
            h_x[i * TOKEN_DIM + d] = (float)(i * TOKEN_DIM + d);
    }

    // 分配 device 内存
    int32_t  *d_indices;
    uint4    *d_x, *d_output;
    uint32_t *d_token_offset, *d_expert_offsets;

    cudaMalloc(&d_indices,       N * K * sizeof(int32_t));
    cudaMalloc(&d_x,             N * TOKEN_DIM_U4 * sizeof(uint4));
    cudaMalloc(&d_output,        total_output_tokens * TOKEN_DIM_U4 * sizeof(uint4));
    cudaMalloc(&d_token_offset,  N * K * sizeof(uint32_t));
    cudaMalloc(&d_expert_offsets, NUM_EXPERTS * sizeof(uint32_t));

    cudaMemcpy(d_indices, h_indices, N * K * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * TOKEN_DIM * sizeof(float), cudaMemcpyHostToDevice);

    // Phase 1：计算 prefix sum
    // shared memory 大小 = NUM_EXPERTS 个 uint32_t
    int p1_threads = 128;
    int p1_smem = NUM_EXPERTS * sizeof(uint32_t);
    phase1_kernel<<<1, p1_threads, p1_smem>>>(
        d_indices, N, d_token_offset, d_expert_offsets
    );

    // Phase 2：scatter 写入
    int p2_blocks = min(N, 256);
    phase2_kernel<<<p2_blocks, NUM_THREADS>>>(
        d_x, d_indices, d_token_offset, d_expert_offsets, d_output, N
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // 验证：检查 expert_offsets
    uint32_t h_expert_offsets[NUM_EXPERTS];
    cudaMemcpy(h_expert_offsets, d_expert_offsets,
               NUM_EXPERTS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("Expert offsets (prefix sum):\n");
    for (int e = 0; e < NUM_EXPERTS; e++)
        printf("  expert_offsets[%d] = %u\n", e, h_expert_offsets[e]);

    printf("\n✅ Total tokens routed: %u (expected %d)\n",
           h_expert_offsets[NUM_EXPERTS - 1], N * K);

    // 释放
    cudaFree(d_indices);
    cudaFree(d_x);
    cudaFree(d_output);
    cudaFree(d_token_offset);
    cudaFree(d_expert_offsets);

    return 0;
}
