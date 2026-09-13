# Triton Hardware Kernels 🚀

Welcome! This repository is my personal "learning by doing" space for mastering GPU architecture, memory hierarchies, and low-level optimization using OpenAI Triton.

A collection of high-performance, fused GPU kernels written in OpenAI Triton. 

This repository focuses on overcoming the "Memory Wall" in modern Generative AI workloads. While frameworks like PyTorch are highly optimized for dense matrix multiplications (Compute-bound), they heavily fragment memory and introduce massive kernel launch overheads for element-wise operations and routing logic (Memory-bound).

These implementations bridge the gap between high-level Python and bare-metal GPU architecture by fusing operations in SRAM, maximizing Compute (SM) utilization, and minimizing High Bandwidth Memory (HBM) roundtrips.

## 🛠️ Implemented Kernels


| Kernel                | Description                                                | Hardware Optimization                                                                                                | Link                                         |
| --------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| **Fused SwiGLU**      | Element-wise activation used in Llama 3 & Mistral.         | Eliminates intermediate HBM tensor materialization. Slashes execution time by 40% and triples SM compute efficiency. | `[fused_swiglu](./02_fused_swiglu)`          |
| **Fused MoE Routing** | Token routing and expert selection for Mixture of Experts. | Mitigates memory bottlenecks during sparse expert assignment.                                                        | `[fused_moe_router](./01_fused_moe_router/)` |


*Click into the subdirectories for deep-dive hardware teardowns, architecture diagrams, and raw profiling data.*

## 🔬 Profiling & Hardware Analysis

Writing the kernel is only half the battle. Every kernel in this repository is heavily profiled to prove its hardware efficiency using **NVIDIA Nsight Compute (**`ncu`**)** and the **PyTorch Profiler**. 

Optimization metrics focus on:

- **Memory Throughput vs. Compute (SM) Throughput:** Shifting workloads from the memory bus to the math cores.
- **Kernel Launch Overhead:** Collapsing multi-kernel PyTorch native implementations into single, continuous blocks.
- **Warp Occupancy & L2 Cache Hit Rates:** Ensuring efficient memory coalescing and thread execution.



## 🚀 Roadmap / Upcoming Implementations

- **PagedAttention (vLLM style):** Non-contiguous KV-cache memory management via block tables.
- **Fused Linear Cross-Entropy Loss:** Chunked loss calculation to prevent massive logits tensor materialization.



## 💻 Getting Started

All kernels are designed to be standalone and can be tested locally or in Google Colab (T4/L4 GPUs).

```bash
# Clone the repository
git clone [https://github.com/Deepak9242/triton-hardware-kernels.git](https://github.com/Deepak9242/triton-hardware-kernels.git)
cd triton-hardware-kernels

# Install requirements
pip install -qU torch triton matplotlib pandas
```

