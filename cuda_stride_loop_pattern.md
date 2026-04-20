# CUDA Stride Loop 模式笔记

## 代码来源

`a2a_dispatch_send.cu` — block 0 清零 `tokens_per_expert` 数组：

```cuda
uint32_t *tokens_per_expert = (uint32_t*)shared_memory;
for (uint32_t i = threadIdx.x; i < num_experts; i += blockDim.x) {
    tokens_per_expert[i] = 0;
}
```

---

## 核心模式

```
起点 = threadIdx.x      步长 = blockDim.x
```

每个 thread 从自己的编号开始，每次跳一个 block 的宽度，直到超出数组边界。

---

## 例子1：清零 tokens_per_expert（对应图里参数）

图里参数：**4个block，每block 4个thread，`blockDim.x = 4`**，假设 `num_experts = 8`。

第一步：只有 `blockIdx.x == 0` 的 block 进入 if

图里绿色高亮的就是 block 0，其他 block（1、2、3）直接跳过，什么都不干。

第二步：block 0 的 4 个 thread 并行清零

for (uint32_t i = threadIdx.x; i < 8; i += blockDim.x /*4*/) {
        tokens_per_expert[i] = 0;
}

4 个 thread 同时出发，各自负责不同的格子：

| thread | 第1轮 i= | 第2轮 i= | 负责清零的格子 |
|--------|---------|---------|-------------|
| 0（橙色）| 0       | 4 |      `[0]`, `[4]` |
| 1       | 1       | 5 |      `[1]`, `[5]` |
| 2       | 2       | 6 |      `[2]`, `[6]` |
| 3       | 3       | 7 |      `[3]`, `[7]` |

两轮循环，8个格子全部清零，4个thread并行完成，速度是单线程的2倍。

---

## 例子2：更大数组（num_experts = 12）

假设 `num_experts = 12`，`blockDim.x = 4`（block 里有 4 个 thread）：

```
第1轮（i = threadIdx.x 初始值）：
  thread 0 → tokens_per_expert[0]  = 0
  thread 1 → tokens_per_expert[1]  = 0
  thread 2 → tokens_per_expert[2]  = 0
  thread 3 → tokens_per_expert[3]  = 0

第2轮（i += 4）：
  thread 0 → tokens_per_expert[4]  = 0
  thread 1 → tokens_per_expert[5]  = 0
  thread 2 → tokens_per_expert[6]  = 0
  thread 3 → tokens_per_expert[7]  = 0

第3轮（i += 4）：
  thread 0 → tokens_per_expert[8]  = 0
  thread 1 → tokens_per_expert[9]  = 0
  thread 2 → tokens_per_expert[10] = 0
  thread 3 → tokens_per_expert[11] = 0

第4轮（i = 12, 13, 14, 15）：全部 >= 12，退出循环
```

12 个格子清零完毕，每个 thread 跑了 **3 次**。

---

## 为什么用这个模式，不用简单的 `tokens_per_expert[threadIdx.x] = 0`？

| 方式 | 问题 |
|------|------|
| `tokens_per_expert[threadIdx.x] = 0` | 只能处理 `num_experts <= blockDim.x` 的情况，expert 数量一多就溢出 |
| Stride Loop | **无论数组多长都能正确处理**，thread 数不够就多跑几轮 |

实际中 expert 数可能是 64、128、256，而 `blockDim.x` 通常是 128 或 256。Stride Loop 让同一段代码天然适配任意大小。

---

## 关键性质

- **无 warp divergence**：所有 thread 执行相同的循环次数（或差1次），条件判断结果一致
- **内存访问连续**：相邻 thread 访问相邻地址（`[0],[1],[2],[3]`），对 GPU 缓存友好（coalesced access）
- **负载均衡**：每个 thread 承担相同数量的工作，没有人闲置

---

## 这个模式在 a2a_dispatch_send.cu 里出现了很多次

```cuda
// 清零 expert 计数
for (uint32_t i = threadIdx.x; i < num_experts; i += blockDim.x)

// 统计每个 token 发给哪个 expert
for (uint32_t i = threadIdx.x; i < num_send_tokens * num_experts_per_token_bound; i += blockDim.x)

// 搬 token 数据（NotFixed 路径）
for (unsigned i = threadIdx.x; i * sizeof(uint4) < token_dim_bound; i += blockDim.x)
```

**规律**：凡是需要并行处理一个比 `blockDim.x` 大的数组时，就用这个模式。
