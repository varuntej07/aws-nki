"""
Decode (flash-decoding) attention kernel.

Decode step of autoregressive attention: a single query position (seqlen_q = 1) attending over a cached K/V of length seqlen_kv. 
This is the memory-bound complement to the compute-bound prefill kernel in `pipelined_attention.py`.

Two kernels live here. 
`decode_attention_fwd` is the simplest correct version: one head, one KV tile, seqlen_kv <= 128. 
`decode_attention_gqa_fwd` lifts that length limit with KV tiling and an online softmax, and 
adds grouped-query attention (GQA) so query heads sharing a KV head also share its K/V loads.

Author: Varun (varuntej.dev@gmail.com)

Validation:
   - Numerics are checked against the NumPy references in this file, via
     check_correct / check_correct_gqa. The same checks run two ways: on CPU
     through nki.simulate_kernel, which needs no device, and on a Neuron
     device through nki.baremetal. Both fp32 and bf16.
   - Both kernels were also validated on Trn2 during upstream review.
   - benchmark_sweep() measures latency with nki.benchmark and reports achieved
     HBM bandwidth against the traffic model in _hbm_bytes(). Run it on your own
     hardware rather than trusting numbers measured on someone else's.

   Run `python decode_attention.py` for the numeric checks (device auto-detected),
   or `python decode_attention.py --sweep` to add the bandwidth sweep.

WARNING: These kernels:
   - Have not been tested across all input configurations
   - Carry no compatibility guarantees
   - May change without prior notice

Status:
   - [A] done: single-head, single-tile decode (MHA, seqlen_kv <= 128)
   - [B] done: KV tiling + online softmax + GQA (decode_attention_gqa_fwd)
   - [C] planned: flash-decoding split-KV for long context

"""
import argparse
import math
import os
import sys

import numpy as np

import nki
# nisa - Neuron Instruction Set Architecture. This is the low-level API to Neuron hardware.
import nki.isa as nisa
import nki.language as nl

# bf16 inputs need ml_dtypes for the NumPy side. Optional: without it the file
# still runs, it just skips the bf16 cases.
try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None

# =====================================================================
# Milestone A: single-head, single-tile decode (MHA).
# Adapted from `attn_fwd_v1` in the attention_fwd_performance tutorial,
# specialized to seqlen_q = 1 and with the softmax scale applied.

@nki.jit
def decode_attention_fwd(q, k, v, softmax_scale=None):
    """
    Bird's Eye View: The model has already processed the prompt; Keys/Values are cached in HBM.
    Now this is the kernel for generating tokens one at a time.
    This kernel computes the attention for one new token, attending over the entire cached KV.
    
    IO tensor layouts (d on the partition axis, matching attn_fwd_v1):
      - q: (d, seqlen_q)     with seqlen_q == 1   (one new query vector)
      - k: (d, seqlen_kv)                         (cached keys, d-major)
      - v: (d, seqlen_kv)                         (cached values, d-major)
      - returns o: (seqlen_q, d) == (1, d)

    Compile-time constant: softmax_scale (defaults to 1/sqrt(d)).

    Assumptions (Milestone A):
      - d <= 128          (head dim fits the partition axis)
      - seqlen_q == 1     (decode: a single query position)
      - seqlen_kv <= 128  (single tile; the P@V contraction axis must fit the 128-wide partition dimension. 
                           Lifting this is Milestone B: KV tiling + online softmax.)
    """
    d, seqlen_q = q.shape
    d_k, seqlen_kv = k.shape
    d_v, seqlen_kv_v = v.shape

    assert d == d_k == d_v, "q, k, v must share head dim d"
    assert seqlen_kv == seqlen_kv_v, "k and v must share seqlen_kv"
    assert d <= 128, "head dim d must fit the 128-wide partition axis"
    assert seqlen_q == 1, "decode kernel expects a single query position"
    assert seqlen_kv <= 128, "Milestone A is single-tile; tile KV in Milestone B"

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    out = nl.ndarray((seqlen_q, d), dtype=q.dtype, buffer=nl.shared_hbm)

    # --- load inputs (q, k, v) from HBM -> copy to SBUF ---
    q_sbuf = nl.load(q)   # [d, 1] one new query vector
    k_sbuf = nl.load(k)   # [d, seqlen_kv]  cached keys
    v_sbuf = nl.load(v)   # [d, seqlen_kv]  cached values

    # --- logits: s = scale * (qᵀ @ k), contract over d (the partition axis) ---
    # matmul lands in PSUM (the only exit door from the tensor engine).
    qk_psum = nl.matmul(q_sbuf, k_sbuf, transpose_x=True)   # [seqlen_q, seqlen_kv]

    # The vector/scalar engines that run the softmax *can* read PSUM, but PSUM is tiny 
    # and is meant to hold tensor-engine matmul outputs, so the recommended practice is 
    # to evict to SBUF as soon as possible and free the bank for the next matmul. 
    # nc_matmul already accumulates in fp32; keeping it fp32 here keeps the softmax numerically stable.
    qk_sbuf = nl.ndarray(qk_psum.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(qk_sbuf, qk_psum)

    # (seqlen_q, seqlen_kv) = (1, seqlen_kv); tensor_scalar writes the scaled tile
    qk_scaled = nl.ndarray(qk_sbuf.shape, dtype=qk_sbuf.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(qk_scaled, qk_sbuf, op0=nl.multiply, operand0=softmax_scale)

    # softmax over seqlen_kv (the cached tokens). Reduce along axis=1 with keepdims
    # collapses seqlen_kv -> 1, so row_max has shape (seqlen_q, 1) = (1, 1).
    row_max = nl.max(qk_scaled, axis=1, keepdims=True)      # find max (stability)
    norm = nl.ndarray(qk_scaled.shape, dtype=qk_scaled.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(norm, qk_scaled, op0=nl.subtract, operand0=row_max)   # subtract max

    # softmax(x) = exp(x) / Σexp(x); scores = softmax(qk_scaled)
    exp_row = nl.exp(norm)                                 # exponentiate [seqlen_q, seqlen_kv]
    sum_row = nl.sum(exp_row, axis=1, keepdims=True)       # denominator [seqlen_q, 1]
    inv_sum = nl.reciprocal(sum_row)                 # 1 / denominator

    scores = nl.ndarray(exp_row.shape, dtype=exp_row.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(scores, exp_row, op0=nl.multiply, operand0=inv_sum)

    # output = Σⱼ scoreⱼ · vⱼ
    v_t_psum = nl.transpose(v_sbuf)           # (d, N) -> (seqlen_kv, d) = [N, d]

    # nl.transpose runs on the Tensor Engine, so its result lands in PSUM. 
    # nc_matmul must read its inputs from SBUF, so we evacuate the transposed result 
    # from PSUM to SBUF before the final matmul. Hence, tensor_copy.
    v_t = nl.ndarray(v_t_psum.shape, dtype=v_sbuf.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(v_t, v_t_psum)

    scores_t_psum = nl.transpose(scores)           # [seqlen_kv, seqlen_q]
    scores_t = nl.ndarray(scores_t_psum.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(scores_t, scores_t_psum)

    attn_psum = nl.matmul(scores_t, v_t, transpose_x=True)        # [seqlen_q, d] = (1, d)
    attn_sbuf = nl.ndarray(attn_psum.shape, dtype=q.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(attn_sbuf, attn_psum)      # PSUM -> SBUF

    nl.store(out, value=attn_sbuf)      # copy output from SBUF -> HBM
    return out


# =====================================================================
# Milestone B: KV tiling + online softmax + grouped-query attention (GQA).
# =====================================================================
# Single batch element. Builds directly on Milestone A:
#   * same QK -> scale -> softmax -> PV pipeline, but
#   * seqlen_kv is streamed in tiles of TILE_KV with a running online-softmax
#     state (m, l, acc) carried across tiles, so we never need all logits at
#     once (this is what lifts A's 'seqlen_kv <= 128' wall), and
#   * 'group' query heads share ONE KV head -> load K/V once per group (GQA win).

TILE_KV = 128   # KV chunk width. Must be <= 128: it becomes the partition axis of the P@V matmul


@nki.jit
def decode_attention_gqa_fwd(q, k, v, n_q_heads, n_kv_heads, softmax_scale=None):
    """
    GQA decode attention, single batch element, online softmax over KV tiles.

    IO tensor layouts (d on the partition axis):
      - q: (d, n_q_heads)
      - k: (n_kv_heads, d, seqlen_kv)
      - v: (n_kv_heads, d, seqlen_kv)
      - returns o: (n_q_heads, d)

    Compile-time constants: n_q_heads, n_kv_heads, softmax_scale (default 1/sqrt(d)).

    Assumptions (Milestone B v1):
      - d <= 128
      - n_q_heads % n_kv_heads == 0       (group = n_q_heads // n_kv_heads)
      - seqlen_kv % TILE_KV == 0          (no padding yet -> Future Work)
    """
    d, n_q = q.shape
    n_kv, d_k, seqlen_kv = k.shape
    n_kv_v, d_v, seqlen_kv_v = v.shape

    assert d == d_k == d_v, "q, k, v must share head dim d"
    assert n_q == n_q_heads, "q head count must match n_q_heads"
    assert n_kv == n_kv_v == n_kv_heads, "k, v head count must match n_kv_heads"
    assert seqlen_kv == seqlen_kv_v, "k and v must share seqlen_kv"
    assert d <= 128, "head dim d must fit the 128-wide partition axis"
    assert n_q_heads % n_kv_heads == 0, "n_q_heads must be a multiple of n_kv_heads"
    assert seqlen_kv % TILE_KV == 0, "seqlen_kv must be a multiple of TILE_KV (v1)"

    group = n_q_heads // n_kv_heads
    num_tiles = seqlen_kv // TILE_KV

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    out = nl.ndarray((n_q_heads, d), dtype=q.dtype, buffer=nl.shared_hbm)

    # One KV head at a time; its 'group' query heads ride together on the free axis 
    # so the shared K/V tile is loaded ONCE per group.
    for i_kv in nl.affine_range(n_kv_heads):
        # grouping the query heads: slice grabs the group w.r.t. n_kv_heads, then load from HBM -> SBUF.
        q_group = nl.load(q[:, i_kv * group:(i_kv + 1) * group])

        # running online-softmax state for the 'group' rows (lives across tiles)
        # SOFTMAX = final o = (Σ exp(logit - m)·v) /  Σ exp(logit - m) 
        # Keeping numerator and denominator UNNORMALIZED here and divide by l once at the end. 
        m_state = nl.full((group, 1), -np.inf, dtype=nl.float32, buffer=nl.sbuf)   # running max per query head, initialized to -inf
        acc = nl.zeros((group, d), dtype=nl.float32, buffer=nl.sbuf)              # acc = running Σ exp(logit - m)·v (numerator)
        
        l_state = nl.zeros((group, 1), dtype=nl.float32, buffer=nl.sbuf)        # running sum of exp(logits - m) -> denominator (normalizer)

        # sequential_range cuz tile i_t depends on tile i_t-1's (m, l, acc).
        for i_t in nl.sequential_range(num_tiles):
            kv_lo = i_t * TILE_KV

            k_tile = nl.load(k[i_kv, :, kv_lo:kv_lo + TILE_KV])   # [d, TILE_KV] since NKI wants d on the partition axis
            v_tile = nl.load(v[i_kv, :, kv_lo:kv_lo + TILE_KV])   # [d, TILE_KV]

            # logits  qk = scale * (q_groupᵀ @ k_tile)
            # contract over d (the partition axis) -> [group, TILE_KV]
            qk_psum = nl.matmul(q_group, k_tile, transpose_x=True)    # PSUM [group, TILE_KV]
            qk_unscaled = nl.ndarray(qk_psum.shape, dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(qk_unscaled, qk_psum)                    # PSUM -> SBUF
            qk = nl.ndarray(qk_unscaled.shape, dtype=qk_unscaled.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(qk, qk_unscaled, op0=nl.multiply, operand0=softmax_scale)

            # online-softmax update
            tile_max = nl.max(qk, axis=1, keepdims=True)             # [group, 1] max logit in THIS tile per query head
            new_m = nl.maximum(m_state, tile_max)                    # [group, 1] update running max

            # acc and l_state were built using m_old as the reference point, every exp was exp(logit - m_old).
            # The new tile uses m_new. Two different reference points can't be added directly.
            # rebase_factor = exp(m_old - m_new) converts the old running state into m_new's units by:
            #   exp(logit - m_old) * exp(m_old - m_new) => exp(logit - m_new)
            # Always in (0, 1] because m_new >= m_old, so the exponent is always <= 0.
            # First tile: m_old = -inf -> rebase_factor = 0 (wipes the empty state cleanly).
            rebase_exp_in = nl.ndarray(m_state.shape, dtype=m_state.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(rebase_exp_in, m_state, op0=nl.subtract, operand0=new_m)
            rebase_factor = nl.exp(rebase_exp_in)

            # p = exp(qk - new_m); new_m (a per-row scalar) broadcasts on the free axis
            norm = nl.ndarray(qk.shape, dtype=qk.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(norm, qk, op0=nl.subtract, operand0=new_m)
            p = nl.exp(norm)                                         # [group, TILE_KV]
            tile_l = nl.sum(p, axis=1, keepdims=True)               # [group,1]

            # l_state = l_state*rebase_factor + tile_l    (the denominator)
            l_scaled = nl.ndarray(l_state.shape, dtype=l_state.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(l_scaled, l_state, op0=nl.multiply, operand0=rebase_factor)
            new_l = nl.add(l_scaled, tile_l)                        # [group,1]

            # P @ V, contracting over TILE_KV
            # the contraction axis must sit on partition, so transpose both
            # operands to [TILE_KV, *] and evacuate (PSUM can't feed a matmul).
            p_t_psum = nl.transpose(p)                              # PSUM [TILE_KV, group]
            p_t = nl.ndarray(p_t_psum.shape, dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(p_t, p_t_psum)

            v_t_psum = nl.transpose(v_tile)                        # PSUM [TILE_KV, d]
            v_t = nl.ndarray(v_t_psum.shape, dtype=v_tile.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(v_t, v_t_psum)

            pv_psum = nl.matmul(p_t, v_t, transpose_x=True)        # PSUM [group, d]
            pv = nl.ndarray(pv_psum.shape, dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(pv, pv_psum)

            # fold this tile into the accumulator: acc = acc*rebase_factor + pv
            acc_scaled = nl.ndarray(acc.shape, dtype=acc.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(acc_scaled, acc, op0=nl.multiply, operand0=rebase_factor)
            new_acc = nl.add(acc_scaled, pv)                       # [group, d]

            # commit the loop-carried state in place (all olds were read above).
            m_state[...] = new_m
            l_state[...] = new_l
            acc[...] = new_acc

        # finalize this group: o = acc / l
        inv_l = nl.reciprocal(l_state)               # [group, 1]
        o_group = nl.ndarray(acc.shape, dtype=q.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(o_group, acc, op0=nl.multiply, operand0=inv_l)

        nl.store(out[i_kv * group:(i_kv + 1) * group, :], value=o_group)

    return out


# =====================================================================
# Reference math implementations in NumPy, for testing the kernels above.
# =====================================================================
def numpy_decode_reference(q, k_cache, v_cache, scale):
    """
    Natural (math) layout, single head:
      - q: (d,)
      - k_cache: (seqlen_kv, d)
      - v_cache: (seqlen_kv, d)
      - returns o: (d,)
    """
    s = scale * (k_cache @ q)        # (seqlen_kv,)  logits
    s = s - s.max()                  # online-softmax max (stability)
    p = np.exp(s)                    # (seqlen_kv,)  unnormalized weights
    p = p / p.sum()                  # normalize
    o = p @ v_cache                  # (d,)  attention output
    return o


def numpy_decode_gqa_reference(q, k_cache, v_cache, n_q_heads, n_kv_heads, scale):
    """
    Natural (math) layout, GQA, single batch element. The oracle for decode_attention_gqa_fwd.
      - q: (n_q_heads, d)
      - k_cache: (n_kv_heads, seqlen_kv, d)
      - v_cache: (n_kv_heads, seqlen_kv, d)
      - returns o: (n_q_heads, d)
    Query head h is served by KV head (h // group), group = n_q_heads // n_kv_heads.
    This is exactly `repeat_kv` + per-head softmax attention, done the slow, obvious way.
    """
    group = n_q_heads // n_kv_heads
    d = q.shape[1]
    o = np.empty((n_q_heads, d), dtype=np.float32)
    for h in range(n_q_heads):
        kv = h // group                      # which KV head this query head shares
        s = scale * (k_cache[kv] @ q[h])     # (seqlen_kv,)  logits
        s = s - s.max()                      # stability
        p = np.exp(s)
        p = p / p.sum()                      # softmax over cached tokens
        o[h] = p @ v_cache[kv]               # (d,)  blend of values
    return o


# =====================================================================
# Test harness.
# =====================================================================
# Three ways to run a kernel, and they are not interchangeable:
#
#   simulate   nki.simulate_kernel   CPU, no device. Real outputs. Slow.
#   baremetal  nki.baremetal         Device. Real outputs. The numeric check.
#   benchmark  nki.benchmark         Device. Latency only.
#
# The third one is the trap: nki.benchmark does not feed the real input values
# through the NEFF, so whatever comes back is undefined. Latency comes from
# benchmark, numbers come from baremetal, and the two cannot be one pass.

# Inferentia2 HBM bandwidth, per CHIP (inf2.xlarge has one chip).
# nki.benchmark runs on a single NeuronCore-v2 and a chip has two, so a
# single-core kernel may not be able to reach this. Treat "% of chip peak"
# as a floor on how well we are doing, not as a utilization figure.
PEAK_BW_GIB_S = 820.0


def _itemsize(dtype):
    return np.dtype(dtype).itemsize


def _dtype_name(dtype):
    return np.dtype(dtype).name


def _quantize(x, dtype):
    """Round fp32 data to the kernel's input dtype, then back to fp32.

    The kernel gets the low-precision values; the NumPy reference gets the
    *same* values widened back to fp32. That isolates what we actually want
    to measure (kernel error given bf16 inputs) from NumPy's own bf16
    arithmetic, which is a different question.
    """
    narrowed = x.astype(dtype)
    return narrowed, narrowed.astype(np.float32)


def _make_mha_inputs(d=128, seqlen_kv=128, dtype=np.float32, seed=42):
    """Build inputs for decode_attention_fwd. Returns (args, ref, meta)."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(d)

    q = rng.standard_normal(d).astype(np.float32)
    k_cache = rng.standard_normal((seqlen_kv, d)).astype(np.float32)
    v_cache = rng.standard_normal((seqlen_kv, d)).astype(np.float32)

    q, q_ref = _quantize(q, dtype)
    k_cache, k_ref = _quantize(k_cache, dtype)
    v_cache, v_ref = _quantize(v_cache, dtype)

    ref = numpy_decode_reference(q_ref, k_ref, v_ref, scale)        # (d,)

    # kernel layout: d on the partition axis -> transpose K, V.
    # ascontiguousarray is load-bearing: nl.load slices assume a C-contiguous
    # HBM tensor in exactly this layout, and a bare .T is only a view.
    q_t = q.reshape(d, 1)                                          # (d, 1)
    k_t = np.ascontiguousarray(k_cache.T)                          # (d, seqlen_kv)
    v_t = np.ascontiguousarray(v_cache.T)                          # (d, seqlen_kv)

    meta = dict(kernel="mha", d=d, seqlen_kv=seqlen_kv, n_q_heads=1,
                n_kv_heads=1, group=1, dtype=dtype)
    return (q_t, k_t, v_t, scale), ref, meta


def _make_gqa_inputs(d=128, seqlen_kv=512, n_q_heads=8, n_kv_heads=2,
                     dtype=np.float32, seed=42):
    """Build inputs for decode_attention_gqa_fwd. Returns (args, ref, meta)."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(d)

    q = rng.standard_normal((n_q_heads, d)).astype(np.float32)
    k_cache = rng.standard_normal((n_kv_heads, seqlen_kv, d)).astype(np.float32)
    v_cache = rng.standard_normal((n_kv_heads, seqlen_kv, d)).astype(np.float32)

    q, q_ref = _quantize(q, dtype)
    k_cache, k_ref = _quantize(k_cache, dtype)
    v_cache, v_ref = _quantize(v_cache, dtype)

    ref = numpy_decode_gqa_reference(q_ref, k_ref, v_ref,
                                     n_q_heads, n_kv_heads, scale)

    # kernel layout: d on the partition axis -> move d to the front.
    q_t = np.ascontiguousarray(q.T)                                # (d, n_q_heads)
    k_t = np.ascontiguousarray(k_cache.transpose(0, 2, 1))         # (n_kv, d, seqlen_kv)
    v_t = np.ascontiguousarray(v_cache.transpose(0, 2, 1))         # (n_kv, d, seqlen_kv)

    meta = dict(kernel="gqa", d=d, seqlen_kv=seqlen_kv, n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads, group=n_q_heads // n_kv_heads, dtype=dtype)
    return (q_t, k_t, v_t, n_q_heads, n_kv_heads, scale), ref, meta


# ---------------------------------------------------------------------
# Traffic model.
# ---------------------------------------------------------------------
# Counting only what the kernel's nl.load / nl.store actually touch:
#
#   K and V tiles   2 * n_kv_heads * seqlen_kv * d     loaded once per KV head,
#                                                      NOT once per query head.
#                                                      That reuse is the GQA win.
#   q_group         n_q_heads * d
#   output store    n_q_heads * d
#
# The q and output terms are noise for any realistic seqlen_kv; they are
# included so the number is honest rather than convenient.

def _hbm_bytes(meta):
    s = _itemsize(meta["dtype"])
    d, n = meta["d"], meta["seqlen_kv"]
    return s * d * (2 * meta["n_kv_heads"] * n + 2 * meta["n_q_heads"])


def _bandwidth_gib_s(total_bytes, latency_us):
    return total_bytes / (latency_us * 1e-6) / (1024 ** 3)


def _roofline_us(meta):
    """Time this config would take if it ran at full chip bandwidth."""
    return _hbm_bytes(meta) / (PEAK_BW_GIB_S * (1024 ** 3)) * 1e6


def _arithmetic_intensity(meta):
    """FLOP per byte. QK and PV are 2*n_q_heads*seqlen_kv*d FLOPs each, and
    bytes are dominated by 2*n_kv_heads*seqlen_kv*d*itemsize, so this reduces
    to 2*group/itemsize. Raising `group` is the only lever the kernel has."""
    return 2.0 * meta["group"] / _itemsize(meta["dtype"])


# ---------------------------------------------------------------------
# Backend dispatch.
# ---------------------------------------------------------------------

def _run(kernel, args, backend="simulate"):
    """Run kernel(*args) and return its output as an ndarray.

    Only the two backends that produce real outputs live here. Benchmarking is
    a separate function on purpose: nki.benchmark does not feed the real inputs
    through the NEFF, so there is deliberately no way to ask it for numbers.
    """
    if backend == "simulate":
        return np.asarray(nki.simulate_kernel(kernel, *args))
    if backend == "baremetal":
        # No artifacts_dir: it errors if the directory is non-empty, and we
        # have no use for the artifacts here.
        return np.asarray(nki.baremetal()(kernel)(*args))
    raise ValueError(f"unknown backend: {backend!r}")


def _latency_us(kernel, args, warmup=10, iters=100, neff_name=None):
    """Benchmark kernel(*args) on device. Returns {50: us, 99: us}."""
    # Decorate inside the call, not at module scope. n_q_heads, n_kv_heads and
    # softmax_scale are compile-time constants, so every config is its own
    # compilation and needs its own warmup.
    bench = nki.benchmark(warmup=warmup, iters=iters,
                          save_neff_name=neff_name)(kernel)
    bench(*args)
    nc = bench.benchmark_result.nc_latency
    # Available percentiles are exactly [0, 1, 10, 25, 50, 90, 99, 100].
    return {50: nc.get_latency_percentile(50),
            99: nc.get_latency_percentile(99)}


# ---------------------------------------------------------------------
# Correctness.
# ---------------------------------------------------------------------

def check_correct(backend="simulate", dtype=np.float32, d=128, seqlen_kv=128):
    """Milestone A: single head, single KV tile."""
    if backend == "benchmark":
        raise ValueError("benchmark outputs are undefined; use simulate or baremetal")

    args, ref, _ = _make_mha_inputs(d=d, seqlen_kv=seqlen_kv, dtype=dtype)
    out = _run(decode_attention_fwd, args, backend=backend)
    out = out.reshape(-1).astype(np.float32)                       # (d,)

    max_diff = float(np.abs(out - ref).max())
    ok = np.allclose(out, ref, atol=1e-2, rtol=1e-2)
    print(f"[check_correct]     {backend:9s} {_dtype_name(dtype):8s} "
          f"d={d} seqlen_kv={seqlen_kv}  max|diff|={max_diff:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_correct_gqa(backend="simulate", dtype=np.float32, d=128,
                      seqlen_kv=512, n_q_heads=8, n_kv_heads=2):
    """Milestone B: KV tiling + online softmax + GQA.

    seqlen_kv=512 is four TILE_KV tiles, so the online-softmax rescale path
    actually runs. group=4 makes it real GQA rather than the degenerate case.
    """
    if backend == "benchmark":
        raise ValueError("benchmark outputs are undefined; use simulate or baremetal")

    args, ref, meta = _make_gqa_inputs(d=d, seqlen_kv=seqlen_kv,
                                       n_q_heads=n_q_heads, n_kv_heads=n_kv_heads,
                                       dtype=dtype)
    out = _run(decode_attention_gqa_fwd, args, backend=backend)
    out = out.astype(np.float32)                                   # (n_q_heads, d)

    max_diff = float(np.abs(out - ref).max())
    ok = np.allclose(out, ref, atol=1e-2, rtol=1e-2)
    print(f"[check_correct_gqa] {backend:9s} {_dtype_name(dtype):8s} "
          f"d={d} seqlen_kv={seqlen_kv} group={meta['group']}  "
          f"max|diff|={max_diff:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def check_all(backend="simulate"):
    """Both kernels, both dtypes."""
    dtypes = [np.float32] + ([bfloat16] if bfloat16 is not None else [])
    if bfloat16 is None:
        print("note: ml_dtypes not installed, skipping bf16 checks "
              "(pip install ml_dtypes)")

    results = []
    for dtype in dtypes:
        results.append(check_correct(backend=backend, dtype=dtype))
        results.append(check_correct_gqa(backend=backend, dtype=dtype))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return all(results)


# ---------------------------------------------------------------------
# Performance.
# ---------------------------------------------------------------------

# Calibrated bound for the optional regression assertion. The multiplicative
# term tracks the traffic model so one constant covers every sweep point; the
# additive term keeps short sequences, which are launch-bound rather than
# bandwidth-bound, from tripping it.
#
# TODO: set these from the first real hardware run (benchmark_sweep prints a
# roofline_x column for exactly this), and record instance type + SDK version
# + date here. Until then assert_perf stays off.
ROOFLINE_SLACK = None
FIXED_OVERHEAD_US = None


def benchmark_kernel(save_artifacts=False):
    """The canonical single-config benchmark, in the shape the rest of
    contributed/ uses. Saves NEFF/NTFF only when asked."""
    args, _, _ = _make_gqa_inputs(seqlen_kv=2048, n_q_heads=8, n_kv_heads=2)
    lat = _latency_us(decode_attention_gqa_fwd, args, warmup=10, iters=100,
                      neff_name="decode_attention.neff" if save_artifacts else None)

    print(f"Latency (P50): {lat[50]:.2f} us")
    print(f"Latency (P99): {lat[99]:.2f} us")
    return lat


_COLS = ("kernel  d    N      Hq  Hkv  grp  dtype     "
         "p50_us    p99_us    MiB      GiB/s   %chip  AI     roofline_x")


def _bench_row(meta, warmup, iters):
    """Benchmark one config and return a row dict."""
    if meta["kernel"] == "mha":
        args, _, _ = _make_mha_inputs(d=meta["d"], seqlen_kv=meta["seqlen_kv"],
                                      dtype=meta["dtype"])
        kernel = decode_attention_fwd
    else:
        args, _, _ = _make_gqa_inputs(d=meta["d"], seqlen_kv=meta["seqlen_kv"],
                                      n_q_heads=meta["n_q_heads"],
                                      n_kv_heads=meta["n_kv_heads"],
                                      dtype=meta["dtype"])
        kernel = decode_attention_gqa_fwd

    # neff_name=None on purpose: a sweep with artifacts on would drop dozens
    # of NEFF/NTFF files into the working directory.
    lat = _latency_us(kernel, args, warmup=warmup, iters=iters, neff_name=None)

    nbytes = _hbm_bytes(meta)
    row = dict(meta)
    row.update(p50_us=lat[50], p99_us=lat[99], nbytes=nbytes,
               gib_s=_bandwidth_gib_s(nbytes, lat[50]),
               ai=_arithmetic_intensity(meta),
               roofline_x=lat[99] / _roofline_us(meta))
    row["pct_chip"] = 100.0 * row["gib_s"] / PEAK_BW_GIB_S
    return row


def _print_row(r):
    print(f"{r['kernel']:<7s} {r['d']:<4d} {r['seqlen_kv']:<6d} "
          f"{r['n_q_heads']:<3d} {r['n_kv_heads']:<4d} {r['group']:<4d} "
          f"{_dtype_name(r['dtype']):<9s} "
          f"{r['p50_us']:<9.2f} {r['p99_us']:<9.2f} "
          f"{r['nbytes'] / 1024 ** 2:<8.2f} {r['gib_s']:<7.1f} "
          f"{r['pct_chip']:<6.1f} {r['ai']:<6.2f} {r['roofline_x']:<.1f}")
    # grep-able duplicate: `... --sweep | grep ^CSV > results.csv`
    print(f"CSV,{r['kernel']},{r['d']},{r['seqlen_kv']},{r['n_q_heads']},"
          f"{r['n_kv_heads']},{r['group']},{_dtype_name(r['dtype'])},"
          f"{r['p50_us']:.3f},{r['p99_us']:.3f},{r['nbytes']},"
          f"{r['gib_s']:.3f},{r['pct_chip']:.3f},{r['ai']:.3f}")


def _fit_overhead(rows):
    """Fit latency_us(N) = a + b*N over one config's length sweep.

    `a` is the fixed launch/sync floor. `b` gives the asymptotic bandwidth with
    that floor removed, which is the number that answers "how close to the roof
    are we". The short-sequence points are almost pure overhead, so their raw
    GiB/s figure means nothing on its own.
    """
    if len(rows) < 2:
        return None, None
    ns = np.array([r["seqlen_kv"] for r in rows], dtype=np.float64)
    us = np.array([r["p50_us"] for r in rows], dtype=np.float64)
    b, a = np.polyfit(ns, us, 1)
    if b <= 0:
        return a, None
    cfg = rows[0]
    # K and V, one element each per token per KV head.
    bytes_per_token = _itemsize(cfg["dtype"]) * cfg["d"] * 2 * cfg["n_kv_heads"]
    return a, _bandwidth_gib_s(bytes_per_token, b)


def benchmark_sweep(warmup=10, iters=100, assert_perf=False):
    """Two experiments, deliberately separated.

    Experiment 1 isolates the GQA effect: n_q_heads and seqlen_kv are pinned,
    so FLOPs and output size are constant while K/V bytes fall 8x. If latency
    tracks bytes rather than FLOPs, the shared K/V loads are real.

    Experiment 2 scales seqlen_kv. The tile loop is nl.sequential_range, so
    latency should grow linearly with the tile count. Departure from linear is
    what would motivate split-KV.
    """
    # Check calibration before burning a few dozen compilations, not after.
    if assert_perf and (ROOFLINE_SLACK is None or FIXED_OVERHEAD_US is None):
        raise RuntimeError(
            "assert_perf=True but ROOFLINE_SLACK / FIXED_OVERHEAD_US are "
            "uncalibrated. Run the sweep once without it and set them from "
            "the roofline_x column.")

    dtypes = [np.float32] + ([bfloat16] if bfloat16 is not None else [])

    print("\n=== Experiment 1: GQA isolation "
          "(n_q_heads=8, seqlen_kv=2048 fixed; n_kv_heads varies) ===")
    print(_COLS)
    exp1 = []
    for dtype in dtypes:
        for n_kv in (8, 4, 2, 1):
            meta = dict(kernel="gqa", d=128, seqlen_kv=2048, n_q_heads=8,
                        n_kv_heads=n_kv, group=8 // n_kv, dtype=dtype)
            r = _bench_row(meta, warmup, iters)
            exp1.append(r)
            _print_row(r)

    print("\n=== Experiment 2: length scaling "
          "(n_q_heads=8, n_kv_heads=2 fixed; seqlen_kv varies) ===")
    print(_COLS)
    exp2 = []
    for dtype in dtypes:
        per_dtype = []
        for n in (128, 512, 1024, 2048, 4096, 8192):
            meta = dict(kernel="gqa", d=128, seqlen_kv=n, n_q_heads=8,
                        n_kv_heads=2, group=4, dtype=dtype)
            r = _bench_row(meta, warmup, iters)
            per_dtype.append(r)
            _print_row(r)
        a, asymptotic = _fit_overhead(per_dtype)
        if a is not None:
            tail = (f"asymptotic BW {asymptotic:.1f} GiB/s "
                    f"({100.0 * asymptotic / PEAK_BW_GIB_S:.1f}% of chip peak)"
                    if asymptotic else "slope non-positive, refit needed")
            print(f"  fit[{_dtype_name(dtype)}]: fixed overhead {a:.2f} us, {tail}")
        exp2.extend(per_dtype)

    print("\n=== Milestone A (single tile, seqlen_kv=128) ===")
    print(_COLS)
    mha = []
    for dtype in dtypes:
        meta = dict(kernel="mha", d=128, seqlen_kv=128, n_q_heads=1,
                    n_kv_heads=1, group=1, dtype=dtype)
        r = _bench_row(meta, warmup, iters)
        mha.append(r)
        _print_row(r)

    print(f"\npeak reference: {PEAK_BW_GIB_S:.0f} GiB/s per Inferentia2 CHIP.")
    print("nki.benchmark runs on ONE NeuronCore-v2 and a chip has two, so "
          "%chip is a lower bound on utilization, not a utilization figure.")
    print(f"warmup={warmup} iters={iters}; GiB/s is computed from p50.")

    rows = exp1 + exp2 + mha
    if assert_perf:
        for r in rows:
            bound = ROOFLINE_SLACK * _roofline_us(r) + FIXED_OVERHEAD_US
            assert r["p99_us"] <= bound, (
                f"p99 {r['p99_us']:.2f}us exceeds {bound:.2f}us for "
                f"N={r['seqlen_kv']} Hkv={r['n_kv_heads']} "
                f"{_dtype_name(r['dtype'])}")
        print(f"perf assertion passed on all {len(rows)} configs")
    return rows


# =====================================================================

def _auto_backend():
    """Use the device if there is one, otherwise fall back to CPU simulation,
    so `python decode_attention.py` does the right thing either way."""
    return "baremetal" if os.path.exists("/dev/neuron0") else "simulate"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Decode (flash-decoding) attention kernels: checks and benchmarks.")
    parser.add_argument("--backend", choices=("simulate", "baremetal"),
                        default=None,
                        help="default: baremetal if a Neuron device is present")
    parser.add_argument("--sweep", action="store_true",
                        help="run the bandwidth sweep (requires a device)")
    parser.add_argument("--assert-perf", action="store_true",
                        help="fail if any config misses the calibrated roofline bound")
    parser.add_argument("--save-artifacts", action="store_true",
                        help="keep the NEFF/NTFF from benchmark_kernel()")
    args = parser.parse_args(argv)

    backend = args.backend or _auto_backend()
    ok = check_all(backend=backend)

    if args.sweep:
        if backend != "baremetal":
            print("\n--sweep needs a Neuron device; skipping.")
        else:
            benchmark_kernel(save_artifacts=args.save_artifacts)
            benchmark_sweep(assert_perf=args.assert_perf)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
