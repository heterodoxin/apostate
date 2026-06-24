"""
Fused NF4 dequantize + GEMM Triton kernel for ROCm gfx1201.

Key insight: x and decoded weights must stay bfloat16 so tl.dot maps to
RDNA4 WMMA instructions. The previous attempt cast both to float32, which
has no WMMA support and fell back to scalar FMAs (10x slower).

With bf16 WMMA: reads 8MB NF4 weight vs bf16's 32MB → should be ~2-4x
faster than bf16 inference.

Drop-in patch: call `patch_bnb_linear4bit()` after importing bitsandbytes.
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl
import torch.nn.functional as F


def _install_do_bench_guard():
    # ROCm (gfx1201) timer occasionally measures estimate_ms=0 for a very fast kernel, and triton's
    # do_bench then divides by it -> ZeroDivisionError that crashes the whole forward mid-autotune.
    # Treat a glitched config as worst (inf) so a working config is chosen instead of crashing.
    import triton.testing as _tt
    if getattr(_tt.do_bench, "_rocm_guarded", False):
        return
    _orig = _tt.do_bench
    def _guarded(*args, **kwargs):
        try:
            return _orig(*args, **kwargs)
        except ZeroDivisionError:
            return float("inf")
    _guarded._rocm_guarded = True
    _tt.do_bench = _guarded
    try:
        import triton.runtime.autotuner as _at
        _at.do_bench = _guarded
    except Exception:
        pass

_install_do_bench_guard()

# ---------------------------------------------------------------------------
# Double-quantized absmax dequantization (uint8 → float32)
# ---------------------------------------------------------------------------

@triton.jit
def _dequant_absmax_kernel(
    absmax_u8_ptr,
    code2_ptr,       # [256] float32
    outer_ptr,       # [N_OUTER] float32
    out_ptr,         # [N_BLOCKS] float32
    N_BLOCKS,
    OUTER_BS: tl.constexpr,
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
# Fused NF4 + GEMM kernel — bfloat16 inputs for RDNA4 WMMA tensor cores
#
# Both x (activations) and decoded w must be bfloat16 so tl.dot compiles
# to WMMA instructions instead of scalar FMAs. f32@f32 has no WMMA on RDNA4.
#
# BK = BSIZE (= 64) so each K-tile has exactly one absmax value per N-row.
# Weight packing: HIGH nibble = k=2b (even K), LOW nibble = k=2b+1 (odd K).
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BM": 16, "BN": 64},  num_warps=4),
        triton.Config({"BM": 16, "BN": 128}, num_warps=4),
        triton.Config({"BM": 16, "BN": 256}, num_warps=8),
        triton.Config({"BM": 32, "BN": 64},  num_warps=4),
        triton.Config({"BM": 32, "BN": 128}, num_warps=4),
        triton.Config({"BM": 32, "BN": 256}, num_warps=8),
        triton.Config({"BM": 64, "BN": 64},  num_warps=4),
        triton.Config({"BM": 64, "BN": 128}, num_warps=8),
        triton.Config({"BM": 64, "BN": 256}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _nf4_matmul_kernel(
    x_ptr,       # [M, K] bfloat16
    w_ptr,       # [N, K//2] uint8
    am_ptr,      # [N, K//BSIZE] float32 absmax
    code_ptr,    # [16] float32 NF4 table
    out_ptr,     # [M, N] bfloat16
    M, N, K,
    stride_xm, stride_xk,
    stride_wn,
    stride_amn,
    stride_om, stride_on,
    BSIZE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    BK: tl.constexpr = BSIZE  # one absmax per K-tile

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k0 in tl.range(0, K, BK):
        offs_ke = k0 + tl.arange(0, BK // 2) * 2   # even K indices
        offs_ko = offs_ke + 1                         # odd  K indices
        mask_m = offs_m[:, None] < M

        # Load activations as bfloat16 — keeps WMMA compatibility
        x_even = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_ke[None, :] * stride_xk,
            mask=mask_m & (offs_ke[None, :] < K), other=0.0,
        )  # [BM, BK//2] bf16
        x_odd = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_ko[None, :] * stride_xk,
            mask=mask_m & (offs_ko[None, :] < K), other=0.0,
        )  # [BM, BK//2] bf16

        # Absmax for this N-slice × K-block  [BN]
        k_block = k0 // BSIZE
        absmax = tl.load(am_ptr + offs_n * stride_amn + k_block, mask=offs_n < N, other=1.0)

        # Packed weight bytes  [BN, BK//2]
        offs_kb = k0 // 2 + tl.arange(0, BK // 2)
        w_bytes = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_kb[None, :],
            mask=(offs_n[:, None] < N) & (offs_kb[None, :] < K // 2), other=0,
        ).to(tl.int32)

        # NF4 decode: gather from 64-byte table (L1-resident after first hit),
        # scale by absmax, cast to bfloat16 for WMMA
        w_even = (tl.load(code_ptr + ((w_bytes >> 4) & 0xF)) * absmax[:, None]).to(tl.bfloat16)
        w_odd  = (tl.load(code_ptr + (w_bytes & 0xF))         * absmax[:, None]).to(tl.bfloat16)

        # bf16 @ bf16.T → f32 accumulation using RDNA4 WMMA hardware
        acc += tl.dot(x_even, tl.trans(w_even), allow_tf32=True)
        acc += tl.dot(x_odd,  tl.trans(w_odd),  allow_tf32=True)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def nf4_matmul(x: torch.Tensor, w: torch.Tensor, quant_state, bias=None) -> torch.Tensor:
    assert quant_state.quant_type == "nf4"
    assert x.dtype == torch.bfloat16

    M = x.shape[0] if x.dim() == 2 else x.view(-1, x.shape[-1]).shape[0]
    x_2d = x.view(M, -1)
    K = x_2d.shape[1]
    N = quant_state.shape[0]

    w_2d = w.data.view(N, K // 2)
    absmax_flat = dequant_absmax(quant_state)
    absmax_2d = absmax_flat.view(N, K // quant_state.blocksize)
    code_f32 = quant_state.code.float().contiguous()

    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)

    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))

    _nf4_matmul_kernel[grid](
        x_2d, w_2d, absmax_2d, code_f32, out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        w_2d.stride(0),
        absmax_2d.stride(0),
        out.stride(0), out.stride(1),
        BSIZE=quant_state.blocksize,
    )

    if bias is not None:
        out = out + bias
    return out.view(*x.shape[:-1], N)


# ---------------------------------------------------------------------------
# Patch bitsandbytes Linear4bit on ROCm
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
        print("[apostate] triton NF4 kernel active (ROCm WMMA path)", flush=True)
    except Exception as e:
        print(f"[apostate] triton NF4 patch skipped: {e}", flush=True)
