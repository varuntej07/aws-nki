"""Get real device latency. Temporary; delete before the PR.

Strategy 1: compile_kernel() -> CompiledKernel.benchmark().
  Blocked last time by `AssertionError: Unknown dtype: float32`, because the
  builder hands the kernel a tracer tensor whose .dtype is a NUMPY dtype while
  nki.language._dtypes.dtype_size() only knows NKI dtypes. That is a missing
  table entry, not a wall, so normalise it here in the harness rather than
  touching the kernel.

Strategy 2: if 1 still fails, dump the NEFF/NTFF and read device-side metrics
  out of the neuron-profile CLI, which is fully supported.
"""
import os
import sys
import time
import json
import tempfile
import traceback
import subprocess

import numpy as np

sys.path.insert(0, "contributed")
from decode_attention import (            # noqa: E402
    decode_attention_gqa_fwd as K,
    _make_gqa_inputs,
)

PEAK_HBM_BYTES_S = 410e9                  # NeuronCore-v2, from _PEAK_HBM_BW


def hbm_bytes(meta):
    s = np.dtype(meta["dtype"]).itemsize
    d, n = meta["d"], meta["seqlen_kv"]
    return s * d * (2 * meta["n_kv_heads"] * n + 2 * meta["n_q_heads"])


def patch_dtype_size():
    """Teach dtype_size() about numpy dtypes.

    tensor.py does `from ._dtypes import dtype_size`, so the name is bound in
    that module too. Patch every module that holds a reference.
    """
    import nki.language._dtypes as D

    original = D.dtype_size

    def tolerant(t):
        try:
            return original(t)
        except Exception:
            return np.dtype(t).itemsize          # numpy fallback

    patched = []
    for name in ("nki.language._dtypes", "nki.language.tensor",
                 "nki.language._core", "nki.language._nki_base"):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "dtype_size"):
            mod.dtype_size = tolerant
            patched.append(name)
    return patched


def annotate(func):
    from nki.compiler.kernel_builder import builder as B
    Tensor = getattr(B, "Tensor", None)
    if Tensor is None:
        import nki.typing as nt
        Tensor = nt.Tensor
    func.__annotations__ = {"q": Tensor, "k": Tensor, "v": Tensor}
    return Tensor


def bench_one(seqlen_kv, n_q_heads, n_kv_heads, warmup=5, iters=50):
    from nki.compiler.kernel_builder.builder import compile_kernel
    from nki.compiler.ncc_driver import CompileOptions

    args, ref, meta = _make_gqa_inputs(seqlen_kv=seqlen_kv,
                                       n_q_heads=n_q_heads,
                                       n_kv_heads=n_kv_heads)
    q_t, k_t, v_t, nq, nkv, scale = args

    with tempfile.TemporaryDirectory(prefix="nki_bench_") as wd:
        opts = CompileOptions(target="trn1", artifacts_dir=wd,
                              output_path=os.path.join(wd, "kernel.neff"))
        t0 = time.time()
        compiled = compile_kernel(
            K.func,
            inputs={"q": q_t, "k": k_t, "v": v_t},
            compile_opts=opts,
            return_outputs=True,
            n_q_heads=nq, n_kv_heads=nkv, softmax_scale=scale,
        )
        compile_s = time.time() - t0

        t0 = time.time()
        res = compiled.benchmark(warmup=warmup, iterations=iters,
                                 q=q_t, k=k_t, v=v_t)
        wall = time.time() - t0

    nbytes = hbm_bytes(meta)
    row = dict(meta, nbytes=nbytes, compile_s=compile_s, wall_s=wall,
               per_run_ms=wall / (warmup + iters) * 1e3,
               latency_us=(res.latency or 0) * 1e6,
               min_us=(res.latency_min or 0) * 1e6,
               max_us=(res.latency_max or 0) * 1e6,
               std_us=(res.latency_std or 0) * 1e6,
               mbu=res.mbu, hfu=res.hfu)
    if res.latency:
        row["gbs"] = nbytes / res.latency / 1e9
        row["pct_peak"] = 100.0 * nbytes / res.latency / PEAK_HBM_BYTES_S

    # outputs come back real, so this validates numerics through this path too
    row["max_diff"] = None
    for name, arr in (res.outputs or {}).items():
        a = np.asarray(arr)
        if a.shape == ref.shape:
            row["max_diff"] = float(np.abs(a.astype(np.float32) - ref).max())
    return row


HDR = ("N      Hq  Hkv  grp  MiB      lat_us     min_us     std_us   "
       "GB/s    %peak  mbu     max|diff|")


def show(r):
    mbu = "-" if r.get("mbu") is None else f"{r['mbu']:.3f}"
    md = "-" if r.get("max_diff") is None else f"{r['max_diff']:.2e}"
    print(f"{r['seqlen_kv']:<6d} {r['n_q_heads']:<3d} {r['n_kv_heads']:<4d} "
          f"{r['group']:<4d} {r['nbytes'] / 1024 ** 2:<8.2f} "
          f"{r['latency_us']:<10.2f} {r['min_us']:<10.2f} {r['std_us']:<8.2f} "
          f"{r.get('gbs', 0):<7.1f} {r.get('pct_peak', 0):<6.1f} {mbu:<7s} {md}",
          flush=True)
    print(f"CSV,{r['seqlen_kv']},{r['n_q_heads']},{r['n_kv_heads']},{r['group']},"
          f"{r['nbytes']},{r['latency_us']:.3f},{r['min_us']:.3f},"
          f"{r['std_us']:.3f},{r.get('gbs', 0):.3f},{r.get('pct_peak', 0):.3f},"
          f"{r.get('mbu')}", flush=True)


def main():
    print("patched dtype_size in:", patch_dtype_size(), flush=True)
    print("Tensor:", annotate(K.func), flush=True)

    print("\n--- smoke test: one config ---", flush=True)
    r = bench_one(512, 8, 2)
    print(f"compile {r['compile_s']:.1f}s, 55 runs in {r['wall_s']:.2f}s "
          f"({r['per_run_ms']:.1f} ms/run)", flush=True)
    print(HDR, flush=True)
    show(r)

    if r["per_run_ms"] > 200:
        print("\nSTOP: still ~recompiling per run; these are not execution times.")
        return
    if r["max_diff"] is not None and r["max_diff"] > 1e-2:
        print("\nSTOP: outputs do not match the reference; timings are moot.")
        return

    print("\n=== Experiment 1: GQA isolation (Hq=8, N=2048; Hkv varies) ===",
          flush=True)
    print(HDR, flush=True)
    for nkv in (8, 4, 2, 1):
        show(bench_one(2048, 8, nkv))

    print("\n=== Experiment 2: length scaling (Hq=8, Hkv=2) ===", flush=True)
    print(HDR, flush=True)
    for n in (128, 512, 1024, 2048, 4096, 8192):
        show(bench_one(n, 8, 2))

    print(f"\npeak {PEAK_HBM_BYTES_S / 1e9:.0f} GB/s per NeuronCore-v2 "
          f"(one core of two on an inf2 chip).")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
