"""
Fast NF4 dequantize for ROCm gfx1201.

Strategy: two-pass — Triton kernel dequantizes NF4 → bfloat16 (16MB write
instead of BNB's 32MB float32), then delegate GEMM to rocBLAS via F.linear.
Total bandwidth: 8MB read + 16MB write vs BNB's 8MB read + 32MB write + 16MB
format conversion. The rocBLAS GEMM then reads 16MB bf16 instead of 32MB f32.

Drop-in patch: call `patch_bnb_linear4bit()` after importing bitsandbytes.
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Fast absmax dequantization (double-quantized uint8 → float32)
# ---------------------------------------------------------------------------

@triton.jit
def _dequant_absmax_kernel(
    absmax_u8_ptr,
    code2_ptr,       # [256] float32
    outer_ptr,       # [N_OUTER] float32
    out_ptr,         # [N_BLOCKS] float32
    N_BLOCKS,
    OUTER_BS: tl.constexpr,  # = 256
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_BLOCKS
    a8 = tl.load(absmax_u8_ptr + offs, mask=mask, other=0).to(tl.int32)
    c2 = tl.load(code2_ptr + a8, mask=mask, other=0.0)
    scale = tl.load(outer_ptr + offs // OUTER_BS, mask=mask, other=1.0)
    tl.store(out_ptr + offs, c2 * scale, mask=mask)


def dequant_absmax(quant_state: object) -> torch.Tensor:
    """Double-quantized uint8 absmax → float32. Cached on quant_state."""
    if hasattr(quant_state, "_absmax_f32_cache"):
        return quant_state._absmax_f32_cache

    if quant_state.state2 is None:
        result = quant_state.absmax.float()
    else:
        n = quant_state.absmax.numel()
        out = torch.empty(n, dtype=torch.float32, device=quant_state.absmax.device)
        _dequant_absmax_kernel[((n + 511) // 512,)](
            quant_state.absmax,
            quant_state.state2.code.float(),
            quant_state.state2.absmax.float(),
            out,
            n,
            OUTER_BS=256,
            BLOCK=512,
        )
        result = out

    if getattr(quant_state, 'offset', None) is not None:
        result = result + quant_state.offset

    quant_state._absmax_f32_cache = result
    return result


# ---------------------------------------------------------------------------
# NF4 → bfloat16 dequantization kernel
#
# Writes [N, K] bfloat16 in stride-2 pairs: hi nibble (even K) and lo nibble
# (odd K) both write to the same cache lines, so write-combining fills lines
# completely — no partial-cache-line penalty.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BN": 16, "BK_HALF": 64},  num_warps=4),
        triton.Config({"BN": 32, "BK_HALF": 64},  num_warps=4),
        triton.Config({"BN": 64, "BK_HALF": 64},  num_warps=4),
        triton.Config({"BN": 16, "BK_HALF": 128}, num_warps=4),
        triton.Config({"BN": 32, "BK_HALF": 128}, num_warps=8),
        triton.Config({"BN": 64, "BK_HALF": 128}, num_warps=8),
    ],
    key=["N", "K"],
)
@triton.jit
def _nf4_to_bf16_kernel(
    w_ptr,      # [N, K//2] uint8 packed
    am_ptr,     # [N, K//BSIZE] float32 absmax
    code_ptr,   # [16] float32 NF4 table
    out_ptr,    # [N, K] bfloat16
    N, K,
    stride_wn,   # = K // 2
    stride_amn,  # = K // BSIZE
    stride_on,   # = K
    BSIZE: tl.constexpr,
    BN: tl.constexpr,
    BK_HALF: tl.constexpr,  # bytes processed per tile = K elements / 2
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    n0 = pid_n * BN
    kb0 = pid_k * BK_HALF  # first byte index in K//2 space

    offs_n = n0 + tl.arange(0, BN)
    offs_kb = kb0 + tl.arange(0, BK_HALF)

    # Packed weight bytes [BN, BK_HALF]
    w_bytes = tl.load(
        w_ptr + offs_n[:, None] * stride_wn + offs_kb[None, :],
        mask=(offs_n[:, None] < N) & (offs_kb[None, :] < K // 2),
        other=0,
    ).to(tl.int32)

    # Absmax: one value per BSIZE bytes = per 2*BSIZE K elements
    # Each byte covers 2 K elements; BSIZE bytes = BSIZE*2 K elements per block.
    # But standard blocksize=64 means 64 K elements per absmax block → BSIZE//2 bytes per block.
    k_block = kb0 * 2 // BSIZE  # block index for the first K element in this tile
    absmax = tl.load(
        am_ptr + offs_n * stride_amn + k_block,
        mask=offs_n < N,
        other=1.0,
    )  # [BN]

    # NF4 decode via table gather (code table is 64 bytes, lives in L1 after first access)
    hi = tl.load(code_ptr + ((w_bytes >> 4) & 0xF)) * absmax[:, None]  # [BN, BK_HALF]
    lo = tl.load(code_ptr + (w_bytes & 0xF))         * absmax[:, None]

    # Write to [N, K] bfloat16 with stride-2 stores.
    # hi → even K positions (k0, k0+2, ...), lo → odd K positions (k0+1, k0+3, ...)
    # Both store into the same cache lines → write-combining fills lines completely.
    offs_ke = kb0 * 2 + tl.arange(0, BK_HALF) * 2      # even K offsets
    offs_ko = offs_ke + 1                                  # odd K offsets

    tl.store(
        out_ptr + offs_n[:, None] * stride_on + offs_ke[None, :],
        hi.to(tl.bfloat16),
        mask=(offs_n[:, None] < N) & (offs_ke[None, :] < K),
    )
    tl.store(
        out_ptr + offs_n[:, None] * stride_on + offs_ko[None, :],
        lo.to(tl.bfloat16),
        mask=(offs_n[:, None] < N) & (offs_ko[None, :] < K),
    )


def nf4_dequant_bf16(w: torch.Tensor, quant_state) -> torch.Tensor:
    """Dequantize NF4 packed weight to bfloat16 [N, K]."""
    N = quant_state.shape[0]
    K = quant_state.shape[1]
    BSIZE = quant_state.blocksize

    w_2d = w.data.view(N, K // 2)
    absmax_flat = dequant_absmax(quant_state)
    absmax_2d = absmax_flat.view(N, K // BSIZE)

    out = torch.empty(N, K, dtype=torch.bfloat16, device=w.device)
    code_f32 = quant_state.code.float().contiguous()

    grid = lambda meta: (
        triton.cdiv(N, meta["BN"]),
        triton.cdiv(K // 2, meta["BK_HALF"]),
    )

    _nf4_to_bf16_kernel[grid](
        w_2d, absmax_2d, code_f32, out,
        N, K,
        w_2d.stride(0),
        absmax_2d.stride(0),
        out.stride(0),
        BSIZE=BSIZE,
    )
    return out


def nf4_matmul(x: torch.Tensor, w: torch.Tensor, quant_state, bias=None) -> torch.Tensor:
    """
    NF4 matmul via fast dequant-to-bf16 + rocBLAS GEMM.
    x: [M, K] bfloat16
    w: bitsandbytes Params4bit
    Returns: [M, N] bfloat16
    """
    assert quant_state.quant_type == "nf4"
    assert x.dtype == torch.bfloat16

    M = x.shape[0] if x.dim() == 2 else x.view(-1, x.shape[-1]).shape[0]
    x_2d = x.view(M, -1)

    w_bf16 = nf4_dequant_bf16(w, quant_state)   # [N, K] bf16
    out = F.linear(x_2d, w_bf16, bias)           # rocBLAS handles the GEMM

    return out.view(*x.shape[:-1], quant_state.shape[0])


# ---------------------------------------------------------------------------
# Patch bitsandbytes Linear4bit to use the fast dequant path on ROCm
# ---------------------------------------------------------------------------

_patched = False


def patch_bnb_linear4bit():
    global _patched
    if _patched:
        return
    try:
        if not (torch.cuda.is_available() and torch.version.hip):
            return

        import bitsandbytes.nn.modules as _m
        _orig_forward = _m.Linear4bit.forward

        def _fast_forward(self, x: torch.Tensor):
            if (
                self.weight.quant_state is not None
                and self.weight.quant_state.quant_type == "nf4"
                and x.is_cuda
                and x.dtype == torch.bfloat16
                and not self.training
            ):
                bias = self.bias.to(x.dtype) if self.bias is not None else None
                return nf4_matmul(x, self.weight, self.weight.quant_state, bias=bias)
            return _orig_forward(self, x)

        _m.Linear4bit.forward = _fast_forward
        _patched = True
        print("[apostate] triton NF4 fast-dequant path active (ROCm)", flush=True)
    except Exception as e:
        print(f"[apostate] triton NF4 patch skipped: {e}", flush=True)
