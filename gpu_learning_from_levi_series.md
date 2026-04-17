# GPU 技术学习整理（基于 @levidiamode Day 91-99/365 系列）

更新时间：2026-04-12
来源：公开可访问页面与镜像索引（X 链接 + nitter 索引）
适用人群：希望从 0 到 1 进入 GPU 编程与 LLM 系统优化的学习者

---

## 1. 这组帖子主要讲了什么

这个系列并不是“从 CUDA 语法开始”的入门教程，而是从真实工程问题切入：

1. 如何在 AMD/CDNA 平台优化 LLM 推理内核（GEMM、MLA、MoE）
2. 如何理解推理系统指标（TTFT、TPOT、吞吐/延迟权衡）
3. 如何在算子层、运行时层、系统层同时做优化
4. 如何通过阅读论文、看比赛解法、对照源码建立实战直觉

一句话总结：
这是“面向大模型推理性能优化”的 GPU 学习路线，偏工程实战和性能分析。

---

## 2. 系列时间线（Day 91-99）

1. Day 91：AITER 与 ATOM 的分层关系，理解 AMD 推理栈
2. Day 92：Prefill/Decode Disaggregation，TTFT/TPOT 与 goodput
3. Day 93：TP/DP/PP/EP/CP 并行策略的全局图景
4. Day 94：概念复盘（MLA、KV Cache、MoE、TPOT、TTFT 等）
5. Day 95：AMD Kernel Challenge 获奖解法拆解（GEMM/MLA/MoE）
6. Day 96：HipKittens 讲座学习，AMD 与 Nvidia 编译/寄存器策略差异
7. Day 97：深入 HipKittens 论文，XCD swizzle 与缓存复用
8. Day 98：复盘自己与冠军方案差异（封装层/元数据/预编译）
9. Day 99：MoE Warp Decode 思路，重排并行轴以降低 decode 开销

---

## 3. 核心知识地图

### 3.1 指标层（先知道优化目标）

1. TTFT（Time To First Token）：首 token 延迟
2. TPOT（Time Per Output Token）：每个输出 token 的延迟
3. Throughput：吞吐量（单位时间处理多少请求/Token）
4. Goodput：在满足服务质量约束下的有效吞吐

你需要形成的直觉：
不同业务目标会改变最优策略。低并发、强实时与高并发离线批处理，最优解可能完全不同。

### 3.2 算子/内核层（性能根因）

1. GEMM：是否把量化融合进主循环，避免额外内存往返
2. MLA Decode：是否减少 per-call 元数据开销与 kernel launch 数
3. MoE：是否避免路由/重排带来的布局变换开销，是否充分利用硬件并行

### 3.3 系统层（把内核收益兑现）

1. 是否使用 persistent 路径/预分配中间缓冲
2. 是否提前预编译避免 JIT 超时
3. 是否按序列长度和 batch 做 shape-specific 配置
4. 是否按硬件拓扑做任务映射（例如 XCD/chiplet aware）

---

## 4. 每天你应该学到的“可执行要点”

### Day 91：AITER + ATOM

1. AITER 可理解为高性能算子库层（底下可接 Triton/CK/ASM）
2. ATOM 在更高层，偏“推理执行与调度”
3. 学习重点：分层解耦

行动建议：
画一张 3 层图：Kernel -> Operator -> Serving Runtime，并标注你当前项目在哪一层做优化。

### Day 92：Disaggregation 与 TTFT/TPOT

1. Prefill 和 Decode 负载特性不同，混在一起会互相干扰
2. 分离后可分别优化，但会引入调度与资源耦合新问题

行动建议：
对你的服务记录 4 个指标：TTFT、TPOT、吞吐、P99 延迟。先做基线，再谈优化。

### Day 93：并行策略全景

1. DP/TP/PP/EP/CP 各自优化不同瓶颈
2. 关键不是“懂定义”，而是“知道何时组合使用”

行动建议：
写一页表格：每种并行策略主要节省什么、代价是什么、什么时候值得用。

### Day 95/98：比赛方案复盘最有价值

1. 顶级优化常来自“减少数据路径长度 + 减少 launch + 贴近硬件指令”
2. 只改 kernel 不够，运行时路径（元数据、预分配、预编译）同样关键
3. 环境差异（库版本、runner 限制）能决定能否拿到最终成绩

行动建议：
每次优化实验都记录三类变化：
1) 算子实现变更
2) 运行时路径变更
3) 环境与工具链变更

---

## 5. 面向初学者的 6 周学习计划

### 第 1-2 周：建立概念框架

1. 弄懂 TTFT/TPOT/Throughput/Goodput
2. 弄懂 Prefill vs Decode 的差异
3. 弄懂 KV Cache、GQA、MLA、MoE 的基本动机

产出：
写一份“术语到瓶颈映射表”（每个概念对应它在优化里解决了什么问题）。

### 第 3-4 周：进入 kernel 视角

1. 学会看一次 kernel profiling 结果
2. 学会判断是计算受限还是内存受限
3. 学会识别 kernel launch 过多、元数据计算重复等隐性开销

产出：
对一个小算子做 3 轮优化实验，记录前后指标。

### 第 5-6 周：系统级集成

1. 学会把 kernel 优化放回服务系统中验证（不是只看 microbench）
2. 学会做 shape-specific 配置和预分配
3. 学会写一份复盘：收益来自哪里，副作用是什么

产出：
完成一份“从 baseline 到优化版”的复现报告（含 TTFT/TPOT/吞吐变化）。

---

## 6. 推荐的学习顺序（按难度）

1. 指标与服务目标：TTFT、TPOT、goodput
2. LLM 推理流程：prefill/decode、KV cache
3. 并行策略：DP/TP/PP/EP/CP
4. 算子优化案例：GEMM/Attention/MLA
5. MoE 推理与路由开销
6. 硬件相关细节：寄存器、缓存、chiplet/XCD 映射

如果你是新手，不建议一开始就钻 ISA 细节。先形成“性能因果链”再下潜。

---

## 7. 这组帖子里最值得抄作业的方法

1. 学习驱动方式：每天聚焦一个问题，不求全覆盖
2. 资源组合方式：论文 + 讲座 + 代码 + 竞赛解法对照
3. 复盘方式：不仅记录“快了多少”，还记录“为什么快”
4. 工程方式：同时关注 kernel、runtime、环境三层

---

## 8. 原始帖子入口（便于你继续追）

1. Day 91: https://x.com/levidiamode/status/2040537168020447253
2. Day 92: https://x.com/levidiamode/status/2040938107604742640
3. Day 93: https://x.com/levidiamode/status/2041229052804280811
4. Day 94: https://x.com/levidiamode/status/2041557537162485919
5. Day 95: https://x.com/levidiamode/status/2042021471178850515
6. Day 96: https://x.com/levidiamode/status/2042368949711487143
7. Day 97: https://x.com/levidiamode/status/2042709687712518259
8. Day 98: https://x.com/levidiamode/status/2043111524492066963
9. Day 99: https://x.com/levidiamode/status/2043466778253332655

说明：由于 X 对匿名访问限制较多，整理过程结合了公开镜像索引，建议你打开原帖查看完整上下文和配图。
