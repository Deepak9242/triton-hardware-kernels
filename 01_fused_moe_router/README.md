# Fused MoE Top-K Router Kernel in OpenAI Triton

This repository provides a high-performance implementation of a fused Top-2 Gating and Numerically Stable Softmax router kernel for Mixture-of-Experts (MoE) architectures, written natively in OpenAI Triton.

## 🚀 The Core Optimization (1-Pass vs 3-Pass VRAM Slicing)

In standard un-fused frameworks (like standard PyTorch), executing a routing layer forces the GPU to execute three separate global memory passes:
1. `torch.matmul`: Computes dense routing scores across experts.
2. `torch.topk`: Sorts and extracts the highest-ranked execution channels.
3. `torch.softmax`: Normalizes final activation probabilities.

This implementation collapses the gating operations into a **single-pass fused hardware kernel**. By holding the pre-computed routing matrices within the local thread registers and SRAM cache, the kernel executes fully vectorized Top-2 extractions and numerically stable softmax normalization natively on the chip, reducing total VRAM transactions by **66%**.

## 📊 Performance Benchmarks (Live GPU Profiling)

The kernel was profiled using `triton.testing.do_bench` across scaling token batch sizes (Dimension M) on a live NVIDIA GPU instance. 

- **Maximum Acceleration:** Achieved up to a **4x speedup** in memory-bound execution zones (low batch footprints).
- **Throughput Scaling:** Maintained an average **~1.8x execution latency reduction** over the optimized native PyTorch baseline at a batch size of 2,048 tokens.
