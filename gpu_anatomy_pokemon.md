# Understanding the Anatomy of GPUs using Pokémon

> 原文链接：https://blog.ovhcloud.com/understanding-the-anatomy-of-gpus-using-pokemon/  
> 作者：Jean-Louis Queguiner（OVHcloud，2019-03-13，更新于 2020-05-14）

---

## 核心结论

GPU 结构看似复杂，其实就像 Pokémon 卡牌游戏一样简单——学会四个"族"的概念就能看懂任何 GPU。

---

## Fact 1：GPU 天生适合矩阵运算

Deep Learning 的本质 = 大规模并行矩阵乘法和加法（+ 和 ×）。  
GPU（GPGPU）正好就是为大规模并行矩阵乘法和加法设计的。完美契合。

---

## Fact 2：GPU 早期只是图形处理器

- 90 年代的 GPU 是线性设计的，每个阶段有专用硬件：
  - Vertex shaders → Tessellation → Geometry shaders → Rendering
- 固定管线，每步都有专属芯片，无法通用。

---

## Fact 3：CPU 是跑车，GPU 是卡车

| | CPU | GPU |
|---|---|---|
| 特点 | 顺序执行，速度快，延迟低 | 并行执行，吞吐大，延迟高 |
| 比喻 | 跑车 | 货车 |
| 适合 | 串行逻辑任务 | 大规模并行计算（如矩阵运算）|

图像渲染需要"一次性"算出所有像素，天然要求大规模并行 → GPU 架构由此而来。

---

## Fact 4：2006 年 NVIDIA 引入 CUDA Core，终结了流水线专用硬件

- 引入 **CUDA Cores**（ALU，算术逻辑单元）：通用计算单元，可执行任意运算
- 现代 GPU 仍保留少量专用核心：
  - **SFU（Special Function Units）**：sin、cos、倒数、开方等高性能数学运算
  - **TMU（Texture Mapping Units）**：高维矩阵纹理映射

架构演进：Fermi → Kepler → Maxwell → Pascal → Volta → Turing → Ampere

---

## Fact 5：用 Pokémon 卡牌理解 GPU 结构

### 四个"族"

| 族 | 对应概念 | 说明 |
|---|---|---|
| **Micro-Architecture Family** | 微架构代际 | Fermi / Kepler / Maxwell / Pascal / Volta / Turing / Ampere |
| **Architecture Family** | GPC / SM 架构 | 调度、缓存、编排，GPU 的"大脑" |
| **Multi-Core Units Family** | CUDA Cores / Tensor Cores / SFU / TMU | 实际做数学运算的物理核心 |
| **Programming Model Family** | Grid / Block / Warp / Thread | 编程抽象层，让代码跨架构可移植 |

### 玩法（理解层次）

1. 选一张 **Micro-Architecture** 卡（比如 Volta）
2. 找到对应的 **Architecture** 卡（GV100 包含多少个 GPC、SM）
3. 在 Architecture 下放对应的 **Multi-Core Units**（每个 SM 有多少 CUDA/Tensor/SFU/TMU）
4. 用 **Programming Model** 卡覆盖在上面（Grid → Block → Warp → Thread）
5. 完整结构图就出来了

### 各代架构卡牌配置

| 架构 | 年份 | 特点 |
|---|---|---|
| Fermi | 2010 | 首个 GPGPU 架构，引入 CUDA Core |
| Kepler | 2012 | 动态并行，Hyper-Q |
| Maxwell | 2014 | 能效大幅提升 |
| Pascal | 2016 | NVLink，HBM2 |
| Volta | 2017 | 引入 Tensor Core（用于 AI） |
| Turing | 2018 | RT Core（光线追踪）+ Tensor Core |
| Ampere | 2020 | 第三代 Tensor Core，稀疏性支持 |

---

## 资源

- 可打印的 GPU Pokédex 卡牌 PDF：[GPU Cards Game](https://www.ovh.com/blog/wp-content/uploads/2020/05/GPU-Cards-1.pdf)
- 前置阅读：[Deep Learning Explained（OVHcloud）](https://www.ovh.com/fr/blog/deep-learning-explained-to-my-8-year-old-daughter/)
