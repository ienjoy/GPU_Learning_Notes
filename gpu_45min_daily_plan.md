# GPU 编程 45 分钟/天学习任务表
# 基于 @levidiamode 365天系列 · 为初学者定制

> 每天 45 分钟。分为 3 个时间段，每段约 15 分钟。
> 完成一周后做一次「周复盘」（30分钟，替代当天最后一个任务）。

---

## 第 1 周：搞懂 GPU 是什么（概念扫盲）

### Day 1
- [0-15分] 读：GPU 和 CPU 的核心区别是什么？（搜索"GPU vs CPU architecture"看图）
- [15-30分] 练：用自己的话解释「并行计算」和「串行计算」的区别
- [30-45分] 记：画一张图，把「Thread / Warp / Block / Grid / SM」的层级关系画出来

**一句话目标**：能向朋友解释 GPU 为什么擅长深度学习

---

### Day 2
- [0-15分] 读：GPU 内存层级（Registers → Shared Memory → L1/L2 → HBM/DRAM）
- [15-30分] 练：在纸上写出每层的容量数量级（KB/MB/GB）和大概延迟（cycles）
- [30-45分] 试：打开 Modal GPU Glossary，查 3 个你不认识的词并做笔记

---

### Day 3
- [0-15分] 看：Stanford CS149 第 1 讲（约 15 分钟跳着看重点）
- [15-30分] 读：什么是 CUDA？CUDA kernel 的基本结构是什么？
- [30-45分] 练：抄写一个 Hello GPU / vector add CUDA 程序，注释每一行

---

### Day 4
- [0-15分] 理解：`blockIdx` `threadIdx` `blockDim` 是什么？每个 thread 如何知道自己的"工号"？
- [15-30分] 练：用例子算出：256 个 thread / block，处理 1024 个元素，需要几个 block？
- [30-45分] 记：在你的笔记里写下"内存合并访问"是什么，为什么重要

---

### Day 5
- [0-15分] 读：Shared Memory 是什么？为什么要用它？
- [15-30分] 练：理解一个用 shared memory 做 tiling 的 GEMM 例子
- [30-45分] 看：CS149 第 2 讲里关于内存层级的部分（约 15 分钟）

---

### Day 6
- [0-15分] 读：什么是 Warp Divergence？举一个会造成 divergence 的代码例子
- [15-30分] 练：改写一段有 if-else 的代码，让它避免 warp divergence
- [30-45分] 思考：为什么 GPU 上写 if-else 要特别小心？

---

### Day 7（周复盘，30分钟）
- [0-10分] 回顾本周 6 个笔记，补全缺失的概念
- [10-25分] 尝试回答：GPU 比 CPU 快的根本原因是什么？为什么不是所有程序都能用 GPU 加速？
- [25-30分] 更新你的「会了 / 不会 / 想深入」清单

---

## 第 2 周：会看 kernel，理解性能

### Day 8
- [0-15分] 读：什么是 Roofline 模型？Arithmetic Intensity 怎么算？
- [15-30分] 练：对向量加法（add）和矩阵乘法（GEMM）分别估算 Arithmetic Intensity
- [30-45分] 记：「计算受限」和「内存受限」的 kernel 各该怎么优化？

---

### Day 9
- [0-15分] 读：什么是 Kernel Fusion？举一个简单例子
- [15-30分] 练：画出 LayerNorm + GELU 不 fusion vs fusion 时的内存读写次数对比
- [30-45分] 思考：什么情况下 fusion 可能反而变慢？

---

### Day 10
- [0-15分] 安装：Python 环境 + `pip install triton torch`
- [15-30分] 读：Triton 的设计哲学，和 CUDA 的区别
- [30-45分] 试：运行 Triton 官方 vector add 例子，能跑通即可

---

### Day 11
- [0-15分] 读：Triton 里的 `tl.program_id`、`tl.load`、`tl.store` 是什么
- [15-30分] 练：修改 vector add 例子，改成 vector multiply（向量逐元素相乘）
- [30-45分] 记：Triton 的 constexpr block size 是什么意思，为什么 block size 影响性能

---

### Day 12
- [0-15分] 看：GPU MODE 的 Triton 入门讲座（在 YouTube 搜 "GPU MODE Triton tutorial"）
- [15-30分] 练：写一个 Triton 实现的 softmax（参考官方文档）
- [30-45分] 记：Triton 是怎么处理边界情况（mask）的？

---

### Day 13
- [0-15分] 读：Profiling 工具：`torch.profiler` 和 `ncu`（Nsight Compute）是做什么的
- [15-30分] 试：用 `torch.profiler` 对你的 Triton kernel 做一次 profile，记录关键指标
- [30-45分] 记：哪些指标最容易说明 kernel 有性能问题？

---

### Day 14（周复盘，30分钟）
- [0-10分] 回顾本周内容
- [10-25分] 尝试回答：我怎么判断一个 kernel 是计算受限还是内存受限？我该先优化哪个？
- [25-30分] 更新清单

---

## 第 3 周：深入 Attention 机制

### Day 15
- [0-15分] 读：标准 Self-Attention 的公式和计算流程
- [15-30分] 练：算出长度 1024、head_dim 64 的 attention，$QK^T$ 矩阵有多大？存到显存需要多少 GB？
- [30-45分] 记：为什么长序列的 attention 会 OOM？

---

### Day 16
- [0-15分] 读：Flash Attention 的核心思路（Tiling + Online Softmax）
- [15-30分] 看：Tri Dao 的 Flash Attention 论文简介 / 任意简版讲解视频（约 15 分钟）
- [30-45分] 练：用自己的话解释在 3 个步骤内 Flash Attention 如何省内存

---

### Day 17
- [0-15分] 读：MHA（多头注意力）= 多组独立的 QKV
- [15-30分] 读：MQA（多查询注意力）和 GQA（分组查询注意力），与 MHA 的区别
- [30-45分] 画图：同一张图上画出 3 种机制的 Q/K/V head 数量对比

---

### Day 18
- [0-15分] 读：KV Cache 是什么？为什么推理时需要它？
- [15-30分] 练：对话 10 轮时，不用 KV Cache vs 用 KV Cache，各需要多少次 attention 计算？
- [30-45分] 读：Paged KV Cache 的动机和基本实现思路

---

### Day 19
- [0-15分] 读：MLA（Multi-Head Latent Attention）的动机：压缩 KV Cache
- [15-30分] 看：@rasbt 的「注意力变体视觉指南」（链接在系列主文档）
- [30-45分] 练：用一段话解释「KV 压缩」和「weight absorption」是什么意思

---

### Day 20
- [0-15分] 读：为什么 speculative decoding（投机解码）能提速？
- [15-30分] 练：小模型草稿 + 大模型验证，画个流程图
- [30-45分] 思考：speculative decoding 在什么情况下收益最大？什么情况下没用？

---

### Day 21（周复盘，30分钟）
- [0-10分] 回顾本周
- [10-25分] 尝试回答：从 MHA 到 GQA 到 MLA，每一步解决了什么问题，引入了什么新问题？
- [25-30分] 更新清单

---

## 第 4 周：LLM 推理系统

### Day 22
- [0-15分] 读：TTFT 和 TPOT 是什么？它们分别反映服务的哪个瓶颈？
- [15-30分] 练：假设一个服务 TTFT = 1s，TPOT = 50ms，生成 100 个 token 总共花多少时间？
- [30-45分] 读：Goodput（有效吞吐量）和 Throughput 的区别

---

### Day 23
- [0-15分] 读：Prefill 和 Decode 阶段的差异（计算量、内存量、并行度）
- [15-30分] 练：同样的 batch，prefill 和 decode 哪个更容易成为计算瓶颈？为什么？
- [30-45分] 读：Prefill-Decode Disaggregation（拆分部署）的动机

---

### Day 24
- [0-15分] 读：Data Parallelism (DP) 是什么？
- [15-30分] 读：Tensor Parallelism (TP) 是什么？它把矩阵怎么切分？
- [30-45分] 读：Pipeline Parallelism (PP) 是什么？"Pipeline bubble" 是怎么出现的？

---

### Day 25
- [0-15分] 读：Expert Parallelism (EP) 是什么？为什么 MoE 需要它？
- [15-30分] 读：Context Parallelism (CP) 是什么？处理超长序列时为什么需要它？
- [30-45分] 练：制作一张表格：5 种并行策略 × 3 列（解决什么问题 / 代价 / 适用场景）

---

### Day 26
- [0-15分] 读：MoE 的路由机制（Top-K routing）
- [15-30分] 练：如果一批 token 里 90% 都选同一个专家，会发生什么问题？
- [30-45分] 读：负载均衡（load balancing）损失是怎么解决这个问题的？

---

### Day 27
- [0-15分] 读：什么是 Compute-bound vs Memory-bound Kernel？
- [15-30分] 练：Flash Attention 是哪种？GEMM 是哪种？GEMV（矩阵-向量乘）是哪种？
- [30-45分] 读：Warp Specialization（Hopper 新特性）：producer warp 和 consumer warp 怎么协作？

---

### Day 28（周复盘，30分钟）
- [0-10分] 回顾本周
- [10-25分] 尝试回答：如果我要部署一个大 MoE 模型，我需要考虑哪些并行策略组合？
- [25-30分] 更新清单

---

## 第 5 周：进入竞赛级优化思路

### Day 29-35
这一周进入「学习竞赛解法」模式，每天选一个主题深入：

| 天 | 主题 | 资源 |
|----|------|------|
| Day 29 | 读 Day 95 的 AMD GEMM 冠军解法 | 原帖 + 本系列主文档 |
| Day 30 | 理解 fused quantization 的收益 | 原帖 Day 95 细节 |
| Day 31 | 研究 MLA Decode wrapper bypass | 原帖 Day 95/98 |
| Day 32 | 比较自己的理解和冠军方案的差距 | 原帖 Day 98 |
| Day 33 | HipKittens：XCD swizzle 是什么 | 原帖 Day 96/97 |
| Day 34 | Warp Decode（MoE 低延迟推理）| 原帖 Day 99 / Cursor 博客 |
| Day 35（周复盘） | 总结 5 周学到的"性能因果链" | 综合复盘 |

---

## 每周周复盘模板（30 分钟）

```
本周学了（填写）：
______________

最清楚的概念（填写）：
______________

还是不懂的地方（填写）：
______________

下周想搞定的事（填写）：
______________

用一句话解释：本周最重要的 GPU 编程直觉是（填写）：
______________
```

---

## 注意事项

1. **不要跳步骤**：第 1-2 周的基础概念是后面所有内容的根基
2. **画图优先**：遇到不懂的就画图，GPU 的很多概念本质上是空间问题
3. **用 AI 问问题**：Claude/ChatGPT 对可视化解释非常擅长，养成"不懂就问"的习惯
4. **不要追求全覆盖**：每天搞懂一个概念，远比一天看 5 个都没记住要好
5. **记录比理解更重要（在初期）**：先把概念的名字和大概意思记住，理解是后来的事
