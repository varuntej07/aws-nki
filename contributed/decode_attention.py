"""
Decode (flash-decoding) attention kernel.

Decode step of autoregressive attention: a single query position
(seqlen_q = 1) attending over a cached K/V of length seqlen_kv. This is the
memory-bound complement to the compute-bound prefill kernel in
`pipelined_attention.py`.

Current state is Milestone A — plain multi-head attention (MHA), one head,
one KV tile. Grouped-query attention (GQA) and flash-decoding split-KV are
on the roadmap below (Milestones B and C), not yet implemented here.

Author: Varun (varuntej.dev@gmail.com)

WARNING: These kernels:
   - Are tested only against internal nightly builds
   - May not be compatible with public NeuronSDK releases
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice

Milestones:
   - [A] DONE here: correct single-head, single-tile decode (MHA, seqlen_kv <= 128)
   - [B] TODO: KV tiling + online softmax + grouped-query attention (GQA)
   - [C] TODO: flash-decoding split-KV for long context

"""
import math
import numpy as np

# nisa - Neuron Instruction Set Architecture. This is the low-level API to Neuron hardware.
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc import nki


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

    # --- load inputs (q, k, v) from HBM -> copy to SBUF (on chip memory) ---
    q_sbuf = nl.load(q)   # [d, 1] one new query vector
    k_sbuf = nl.load(k)   # [d, seqlen_kv]  cached keys
    v_sbuf = nl.load(v)   # [d, seqlen_kv]  cached values

    # --- logits: s = scale * (qᵀ @ k), contract over d (the partition axis) ---
    # matmul lands in PSUM (the only exit door from the tensor engine).
    qk_psum = nl.matmul(q_sbuf, k_sbuf, transpose_x=True)   # [seqlen_q, seqlen_kv]

    # Evacuate PSUM -> SBUF, keeping fp32. Two reasons:
    #   1) free the PSUM bank for the next matmul,
    #   2) the softmax runs on the vector/scalar engines, which read SBUF (not PSUM).
    # fp32 here keeps the softmax numerically stable even if q/k were bf16.
    qk_sbuf = nl.ndarray(qk_psum.shape, dtype=nl.float32, buffer=nl.sbuf)   # lives in SBUF
    nisa.tensor_copy(dst=qk_sbuf, src=qk_psum)

    qk_scaled = nl.ndarray(qk_psum.shape, dtype=nl.float32, buffer=nl.sbuf)     # (seqlen_q, seqlen_kv) = (1, seqlen_kv)
    nisa.tensor_scalar(dst=qk_scaled, data=qk_sbuf, op0=nl.multiply, operand0=softmax_scale)

    # softmax over seqlen_kv (the cached tokens). Reduce along axis=1 with keepdims
    # collapses seqlen_kv -> 1, so row_max has shape (seqlen_q, 1) = (1, 1).
    row_max = nl.max(qk_scaled, axis=1, keepdims=True)      # find max (stability)
    norm = nl.ndarray(qk_scaled.shape, dtype=nl.float32, buffer=nl.sbuf)    # subtract max
    nisa.tensor_scalar(dst=norm, data=qk_scaled, op0=nl.subtract, operand0=row_max)

    # softmax(x) = exp(x) / Σexp(x); scores = softmax(qk_scaled)
    exp_row = nl.exp(norm)                                 # exponentiate [seqlen_q, seqlen_kv]
    sum_row = nl.sum(exp_row, axis=1, keepdims=True)       # denominator [seqlen_q, 1]
    inv_sum = nl.reciprocal(sum_row)                 # 1 / denominator

    scores = nl.ndarray(exp_row.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=scores, data=exp_row, op0=nl.multiply, operand0=inv_sum)

    # output = Σⱼ scoreⱼ · vⱼ
    v_t_psum = nl.transpose(v_sbuf)           # (d, N) -> (seqlen_kv, d) = [N, d]

    # now the result should be in the PSUM. PSUM tensors can't be fed back in as a matmul input on gen3+ hardware,
    # so we must evacuate the transposed result from PSUM to SBUF before the final matmul. Hence, tensor_copy.
    v_t = nl.ndarray(v_t_psum.shape, dtype=v_sbuf.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=v_t, src=v_t_psum)

    scores_t_psum = nl.transpose(scores)           # [seqlen_kv, seqlen_q]
    scores_t = nl.ndarray(scores_t_psum.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=scores_t, src=scores_t_psum)

    attn_psum = nl.matmul(scores_t, v_t, transpose_x=True)        # [seqlen_q, d] = (1, d)
    attn_sbuf = nl.ndarray(attn_psum.shape, dtype=q.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=attn_sbuf, src=attn_psum)      # PSUM -> SBUF

    nl.store(out, value=attn_sbuf)      # copy output from SBUF -> HBM
    return out


# =====================================================================
# Reference math (pure NumPy) — ground truth. Runs anywhere.
# =====================================================================
def numpy_decode_reference(q, k_cache, v_cache, scale):
    """
    Natural (math) layout, single head:
      - q:       (d,)
      - k_cache: (seqlen_kv, d)
      - v_cache: (seqlen_kv, d)
      - returns o: (d,)
    """
    s = scale * (k_cache @ q)        # (seqlen_kv,)  logits
    s = s - s.max()                  # online-softmax max (stability)
    p = np.exp(s)                    # (seqlen_kv,)  unnormalized weights
    p = p / p.sum()                  # normalize
    o = p @ v_cache                  # (d,)          attention output
    return o


# =====================================================================
# Local correctness check — runs on CPU via nki.simulate_kernel (no device).
# On a real Neuron instance, swap to:
#   out = nki.baremetal()(decode_attention_fwd)(q_t, k_t, v_t, scale)
# =====================================================================
def check_correct():
    np.random.seed(42)
    d, seqlen_kv = 128, 128
    scale = 1.0 / math.sqrt(d)

    # natural-layout random inputs (where the numbers come from is irrelevant)
    q = np.random.randn(d).astype(np.float32)
    k_cache = np.random.randn(seqlen_kv, d).astype(np.float32)
    v_cache = np.random.randn(seqlen_kv, d).astype(np.float32)

    ref = numpy_decode_reference(q, k_cache, v_cache, scale)        # (d,)

    # kernel layout: d on the partition axis -> transpose K, V
    q_t = q.reshape(d, 1).astype(np.float32)                       # (d, 1)
    k_t = np.ascontiguousarray(k_cache.T)                         # (d, seqlen_kv)
    v_t = np.ascontiguousarray(v_cache.T)                         # (d, seqlen_kv)

    out = nki.simulate_kernel(decode_attention_fwd, q_t, k_t, v_t, scale)
    out = np.asarray(out).reshape(-1)                             # (d,)

    max_diff = float(np.abs(out - ref).max())
    ok = np.allclose(out, ref, atol=1e-2, rtol=1e-2)
    print(f"[check_correct] d={d} seqlen_kv={seqlen_kv}  max|diff|={max_diff:.3e}")
    print("PASS" if ok else "FAIL")
    return ok


def main():
    check_correct()


if __name__ == "__main__":
    main()
