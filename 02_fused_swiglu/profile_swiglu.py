import torch
import triton
import triton.language as tl


@triton.jit
def fused_swiglu_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)
    
    sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate))
    silu_gate = gate * sigmoid_gate
    result = silu_gate * up
    
    tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)

def triton_swiglu(gate, up):
    out = torch.empty_like(gate)
    n_elements = gate.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    fused_swiglu_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE=1024)
    return out

# --- 2. Profile Execution ---
def main():
    gate = torch.randn(8, 2048, 4096, device='cuda', dtype=torch.float16)
    up = torch.randn(8, 2048, 4096, device='cuda', dtype=torch.float16)
    
    # Warmup
    _ = torch.nn.functional.silu(gate) * up
    _ = triton_swiglu(gate, up)
    torch.cuda.synchronize()

    # PyTorch Native execution
    out_torch = torch.nn.functional.silu(gate) * up
    torch.cuda.synchronize()

    # Triton execution
    out_triton = triton_swiglu(gate, up)
    torch.cuda.synchronize()

if __name__ == "__main__":
    main()
