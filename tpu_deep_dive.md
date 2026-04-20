# TPU Deep Dive

> 原文链接：https://henryhmko.github.io/posts/tpu/tpu.html  
> 作者：Henry Ko（2025-06-18，曾登上 Hacker News 首页）

---

## 一句话总结

TPU = 极致 matmul 吞吐 + 极致能效，通过 **Systolic Array + XLA 编译器**协同设计实现。

---

## 背景

- 2006 年 Google 评估 GPU / FPGA / ASIC，当时用 CPU 满足需求
- 2013 年语音搜索用上神经网络，算力需求爆炸 → 启动 TPU 研发
- 今天 TPU 驱动 Google 大多数 AI 服务：Gemini、Veo、DLRM 推荐系统

---

## 单芯片层（Single-Chip Level）

### TPUv4 芯片结构

每个芯片有 **2 个 TensorCore**（推理型只有 1 个），共享：
- **CMEM**：128 MiB 片上内存
- **HBM**：32 GiB（注意：远小于 H100 的 80GB）

每个 TensorCore 内部：

| 组件 | 说明 |
|------|------|
| **MXU**（Matrix Multiply Unit） | 128×128 Systolic Array，矩阵乘法核心 |
| **VPU**（Vector Unit） | 逐元素操作（ReLU、加法、乘法、Reduction） |
| **VMEM**（32 MiB） | HBM → VMEM 后才能计算，类似 scratchpad |
| **Scalar Unit + SMEM**（10 MiB） | 控制流、标量运算、内存地址生成 |

### TPU vs GPU 对比

| | TPU | GPU（H100） |
|---|---|---|
| 片上内存 | 大（VMEM 32MiB + CMEM 128MiB） | 小（L1 256KB，L2 50MB） |
| HBM | 小（32GiB） | 大（80GiB） |
| 计算核心数 | 少（1个 MXU per TensorCore） | 多（数万 CUDA Cores） |
| 灵活性 | 低（依赖 AOT 编译） | 高（动态缓存） |

---

## 设计哲学

### 设计选择 #1：Systolic Array + Pipelining

**什么是 Systolic Array？**
- 由处理单元（PE）构成的网格，每个 PE 做乘累加，结果传给相邻 PE
- 一旦数据流入，无需额外控制逻辑
- 足够大时，只有输入输出有内存读写，中间无需 DRAM 访问
- 天然适合矩阵乘法和卷积（固定数据流模式）
- 支持流水线：计算和数据搬运可以重叠

**Systolic Array 的缺点：稀疏矩阵**
- 稀疏矩阵无法获益，零值元素也要走完所有 PE cycle
- MoE 等稀疏模型是潜在瓶颈

### 设计选择 #2：AOT 编译 + 减少 Cache 依赖

**为什么避免 Cache？**
- 内存访问能耗 >> 算术运算能耗（高几个数量级）
- 传统 Cache 是为应对不可预测的内存访问模式而存在的
- 但 DL workload 的内存访问模式高度可预测

**TPU 的解法：**
- XLA 编译器提前分析计算图，生成优化程序
- 硬件只需 scratchpad（VMEM/CMEM），无需复杂 Cache 层级
- 结果：省掉了大量 Cache 控制电路和能耗

**TPUv4 能耗数据：**
- 内存操作比算术运算贵几个数量级
- Scaling law 启示：用更多 FLOPS 换更少内存操作，双重优化（快 + 省电）

**JAX + XLA 的关系：**
- `@jit` 第一次调用时：JAX trace → 静态计算图 → XLA 编译成 TPU 专属二进制
- 等价于 AOT，但触发在运行时
- 注意：不同 input shape 需要重新编译 → 避免动态 padding 和长度不定的 for loop

---

## 多芯片层（Multi-Chip Level）

### 层级结构

```
Thread → Tray → Rack → Superpod → Multi-Pod
```

| 层级 | 规模 | 互联方式 |
|------|------|----------|
| **Tray**（Board） | 4 chips / 8 TensorCores | ICI（片间互联） |
| **Rack**（Cube） | 64 chips（4×4×4 3D torus） | ICI + OCS |
| **Superpod**（Full Pod） | 4096 chips（TPUv4） | ICI + OCS |
| **Multi-Pod**（Multislice） | 4096+ chips | DCN（跨 Pod，带宽低） |

Host ↔ Chip：PCIe  
Chip ↔ Chip：ICI（更高带宽）

### OCS（Optical Circuit Switching）的三大好处

**#1 Wraparound（环形拓扑）**
- 将每个轴变成 ring（1D torus）
- 最坏情况跳数从 N-1 降为 (N-1)/2
- 规模越大收益越明显

**#2 非连续多节点切片**
- TPU Slice 不必是物理上连续的 Rack
- OCS 是可重配置的交换机，不是硬连线
- 整个 Pod 可以被视为"节点包"，提高利用率
- 节点故障：blast radius 小，不影响整个 Pod

**#3 Twisted Torus（扭曲环形拓扑）**
- 固定 (x,y,z) 维度，改变连线方式
- 扭曲后某平面内芯片间通信更快
- 尤其适合加速 Tensor Parallel 的 all-to-all 通信

### TPU Slice 拓扑选择

同样 512 chips，可选：

| 拓扑 | 形状 | 适合场景 |
|------|------|----------|
| 8×8×8 Cube | 方形 | DP / TP（高 bisection bandwidth） |
| 4×4×32 Cigar | 长条 | PP（顺序层间通信） |
| 4×8×16 Rectangle | 矩形 | 混合策略 |

### 术语辨析

| 术语 | 定义 |
|------|------|
| TPU Rack | 物理单元，64 chips |
| TPU Pod / Superpod | 最大 ICI 互联单元（TPUv4: 4096 chips）|
| TPU Slice | 抽象概念，Pod 内任意 chip 子集 |

### Multi-Pod（Multislice）

- 多个 Pod 通过 **DCN（数据中心网络）** 连接，带宽低于 ICI
- XLA 负责跨 Pod 通信协调（GSPMD）：
  - 片内：ICI 处理 activation 通信（TP/PP）
  - 跨 Pod：DCN 处理梯度通信（DP）
- PaLM 训练案例：6144 个 TPUv4（2 个 Pod），耗时 56 天

---

## 参考资料

- [Google Blog: TPU Multi-Slice Training](https://cloud.google.com/blog/products/compute/using-cloud-tpu-multislice-to-scale-ai-workloads)
- [TPUv4 论文](https://arxiv.org/abs/2304.01433)
- [How to Scale Your Model - TPUs](https://jax-ml.github.io/scaling-book/tpus/)
- [GSPMD 论文](https://arxiv.org/pdf/2105.04663)
- [Google Ironwood 发布博客](https://blog.google/products/google-cloud/ironwood-tpu-age-of-inference/)
