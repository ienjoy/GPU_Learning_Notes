# Token→Expert 路由循环笔记

## 代码来源

`a2a_dispatch_send.cu`，block 0 第一阶段，统计每个 expert 收到多少 token 并分配排队号：

```cuda
for (uint32_t i = threadIdx.x; i < num_send_tokens * num_experts_per_token_bound; i += blockDim.x) {
    const uint32_t token = i / num_experts_per_token_bound;
    const uint32_t index = i % num_experts_per_token_bound;
    const uint32_t expert = __ldg(&indices[token * indices_stride + index]);

    // Assign an offset to the token within the current rank and expert.
    token_offset[i] = atomicAdd(&tokens_per_expert[expert], 1);
}
```

---

## 第一层：i 是什么

循环下标 `i` 是把二维关系**压成一维**的线性编号：

```
每个 token 有 num_experts_per_token 个 expert 选择（假设=2）：

token 0 → [expert选择0, expert选择1]
token 1 → [expert选择0, expert选择1]
token 2 → [expert选择0, expert选择1]

展开成一维 i：
i=0 → token0 的第0个选择
i=1 → token0 的第1个选择
i=2 → token1 的第0个选择
i=3 → token1 的第1个选择
...
```

总共 `num_send_tokens × num_experts_per_token_bound` 个格子，thread 们用 stride loop 分摊。

---

## 第二层：为什么要算 token 和 index

`i` 是一维编号，但查 `indices` 数组需要知道"是哪个 token 的第几个选择"，即二维坐标。

**一维 → 二维还原公式：**

| 含义                      | 公式                  | 代码 |
|------                     |------                 |------|
| 第几个 token（行号）        | `i / num_experts_per_token` | `const uint32_t token = i / num_experts_per_token_bound;` |
| 该 token 的第几个 expert 选择（列号） | `i % num_experts_per_token` | `const uint32_t index = i % num_experts_per_token_bound;` |

和 Python 的 `np.unravel_index(i, shape)` 是完全相同的思路。

---

## 第三层：indices 数组的内存布局

`indices` 是一个二维张量，在内存里**按行连续存储**：

```
逻辑二维表：
              index=0    index=1
token=0    →  expert_A   expert_B
token=1    →  expert_C   expert_D
token=2    →  expert_E   expert_F

内存一维地址（假设 indices_stride=2，无padding）：
[0]=expert_A  [1]=expert_B  [2]=expert_C  [3]=expert_D  [4]=expert_E  [5]=expert_F
```

**二维坐标 → 一维内存地址公式：**

$$\text{地址} = \text{token} \times \text{indices\_stride} + \text{index}$$

- `token × indices_stride` → 跳到第 token 行的起始位置
- `+ index` → 在这一行取第 index 个元素

> ⚠️ `indices_stride` 不一定等于 `num_experts_per_token`！
> GPU 内存可能为对齐做了 padding，stride 是**实际行间距**，可能比列数大。
> 和 Python 张量的 `.stride()` 概念一致。

---

## 完整例子

假设：`num_send_tokens=3`，`num_experts_per_token=2`，`indices_stride=2`，`blockDim.x=4`

**第1轮（i = threadIdx.x 初始值）：**

| thread | i | token=i/2 | index=i%2 | 内存地址 | 读到 expert |
|--------|---|-----------|-----------|---------|------------|
| 0 | 0 | 0 | 0 | `[0*2+0]=indices[0]` | expert A |
| 1 | 1 | 0 | 1 | `[0*2+1]=indices[1]` | expert B |
| 2 | 2 | 1 | 0 | `[1*2+0]=indices[2]` | expert C |
| 3 | 3 | 1 | 1 | `[1*2+1]=indices[3]` | expert D |

**第2轮（i += 4）：**

| thread | i | token=i/2 | index=i%2 | 内存地址 | 读到 expert |
|--------|---|-----------|-----------|---------|------------|
| 0 | 4 | 2 | 0 | `[2*2+0]=indices[4]` | expert E |
| 1 | 5 | 2 | 1 | `[2*2+1]=indices[5]` | expert F |
| 2 | 6 | ≥6，退出 | — | — | — |
| 3 | 7 | ≥6，退出 | — | — | — |

---

## 第四层：atomicAdd 干了什么

```cuda
token_offset[i] = atomicAdd(&tokens_per_expert[expert], 1);
```

- `tokens_per_expert[expert]`：这个 expert 的计数器（在 shared memory 里）
- `atomicAdd(ptr, 1)`：原子地 +1，返回**加之前的旧值**（即排队号）
- `token_offset[i]`：记录"这次分配的排队号"，后续用于定位在 send buffer 里的写入位置

**类比**：去银行取号机取号，机器自动 +1，你拿到的号就是你在这个 expert 队列里的位置。

---

## 补充：`__ldg` 是什么

```cuda
const uint32_t expert = __ldg(&indices[token * indices_stride + index]);
```

**Q：这行是把2维转1维吗？**

反过来——是把**已知的2维坐标 `(token, index)` 转成1维内存地址**，然后去读值：

```
(token, index)  →  token * indices_stride + index  →  读出 expert 编号
 已知的2D坐标        算出的1D内存偏移量                  读到的数据
```

**`__ldg` = Load through Read-Only Data Cache**

| 对比 | 普通读取 | `__ldg` |
|------|---------|---------|
| 语法 | `indices[addr]` | `__ldg(&indices[addr])` |
| 走的缓存 | L1/L2 data cache | 独立的 read-only texture cache |
| 适用场景 | 可能被写的数据 | 整个 kernel 内只读的数据 |

**为什么用 `__ldg`**：
- `indices` 数组在整个 kernel 里只读不写，走独立只读缓存不会污染 L1 data cache
- 多个 thread 读同一地址时，texture cache 命中率更高
- 相当于给硬件一个提示：这块内存不会被修改，可以积极缓存

类似 C++ 的 `const` + `__restrict__`，但作用在缓存层级上。

---

## 关键点汇总

| 概念 | 说明 |
|------|------|
| `i` 的来历 | 二维(token, expert选择)关系压成一维的线性编号 |
| `token = i / N` | 从线性编号还原行号（哪个 token） |
| `index = i % N` | 从线性编号还原列号（第几个 expert 选择） |
| `indices[token*stride+index]` | 2D坐标→1D地址，从内存取出实际的 expert 编号 |
| `indices_stride` | 实际行间距，≥ 列数（可能有 padding） |
| `__ldg` | 走只读缓存读取，比普通读取更高效，适用于整个 kernel 不写的数组 |
| `atomicAdd` | 给 expert 计数器 +1，同时拿到排队号，保证并发安全 |
| expert 编号的来源 | 运行时由 MoE router 填入 `indices` 数组，kernel 只是读取 |
