# Inside a Volta SM

> 原文链接：https://cvw.cac.cornell.edu/gpu-architecture/gpu-example-tesla-v100/volta_sm
> 来源：Cornell University Center for Advanced Computing (CVW)

---

## 每个 Volta SM 的计算资源

每个 Volta SM 的算力来自以下硬件：

- **CUDA Cores（按数据类型分）**
  - 64 个 FP32 CUDA cores
  - 64 个 INT32 CUDA cores
  - 32 个 FP64 CUDA cores
- **8 个 Tensor Cores**
- **16 个 Special Function Units（SFU）**
- **4 个 Texture Units**

---

## SM 内部结构

每个 SM 被细分为 **4 个 Processing Block**，提高调度灵活性。

CUDA cores 的调度规则：
- 每个时钟周期，最多可处理 **2 个 FP32 或 INT32 的 warp**（可以各1个）
- 或者处理 **1 个 FP64 的 warp**

数据通过每个 processing block 底部的 **32 个 Load/Store Units** 供给主硬件。

![Volta SM Block Diagram](https://cvw.cac.cornell.edu/gpu-architecture/gpu-example-tesla-v100/GV100SMDiagram.png)

*NVIDIA Volta Streaming Multiprocessor (SM) 结构图*

---

## 上下文导航

- 上一页：[Volta Block Diagram](https://cvw.cac.cornell.edu/gpu-architecture/gpu-example-tesla-v100/volta_block)
- 下一页：[Tensor Cores](https://cvw.cac.cornell.edu/gpu-architecture/gpu-example-tesla-v100/tensor_cores)
- 所属课程：[Understanding GPU Architecture](https://cvw.cac.cornell.edu/gpu-architecture)
