import triton
import triton.language as tl
import torch

"""
Swiglu(x) = Swish(x) + Glu(x) = x.σ(x) + Glu(x)
Glu(x) = σ(x).x

---

Swiglu(x)=x⋅σ(x)+σ(x)⋅x=σ(x)⋅(x+x)=2⋅σ(x)⋅x
"""
# copied from unsloth: https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/swiglu.py:_fg_kernel
@triton.jit
def _fg_kernel(e, g, h, n_elements, BLOCK_SIZE : tl.constexpr,):
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask = mask, other = 0).to(tl.float32)
    g_row = tl.load(g + offsets, mask = mask, other = 0)#.to(tl.float32)

    # f = e * sigmoid(e)
    f_row = e_row * tl.sigmoid(e_row) # e_row / (1 + tl.exp(-e_row))
    f_row = f_row.to(g_row.dtype) # Exact copy from HF
    # h = f * g
    h_row = f_row * g_row

    # Store h
    tl.store(h + offsets, h_row, mask = mask)
pass

# copied from unsloth: https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/swiglu.py:swiglu_fg_kernel
def swiglu_fg_kernel(e, g):
    batch, seq_len, hd = e.shape
    n_elements = e.numel()
    h = torch.empty((batch, seq_len, hd), dtype = e.dtype, device = "cuda:0")
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _fg_kernel[grid](e, g, h, n_elements, BLOCK_SIZE = 1024,)
    return h
pass
