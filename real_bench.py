"""Benchmark via the parser frontend: the path @nki.jit actually takes.

nir_probe explained the whole chase. compile_kernel (nki.compiler.kernel_builder)
is a DIFFERENT authoring API, for kernels written against the builder's TileView.
decode_attention.py is written against nl/nisa, which is the PARSER frontend, and
CompileKernel._frontend_cls is ParserFrontend. Every missing attribute came from
feeding a parser-frontend kernel through the builder's front door. No number
produced that way would have been worth printing.

The parser route is public end to end:

    ParserFrontend().compile(...)     -> CompilationResult   (frontend.py:132)
    CompiledKernel.from_frontend(...) -> NEFF                (ncc_driver.py:264)
    CompiledKernel.benchmark(...)     -> latency, MBU, HFU

ParserFrontend and CompilationResult are both exported (nki.compiler.frontend,
nki.compiler). No monkeypatches, and the kernel is untouched, so the thing
measured is the thing that ships.

Because the exact compile() signature has not been read yet, this script prints
it first, then builds the call from the signature's own parameter names rather
than guessing. If a parameter is unexpected, the printed signature and source
say exactly what to write next.

Run: python real_bench.py
"""
import inspect
import math
import os
import sys
import tempfile
import time
import traceback

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


def kernel_inputs(args):
    """The kernel's arguments keyed by name; compile() wants a dict, not a tuple."""
    names = list(inspect.signature(K.func).parameters)
    return dict(zip(names, args))


def show_api():
    """Print the machinery around compile_kernel_to_nir, so a miss self-explains."""
    from nki.compiler.ncc_driver import CompileOptions
    from nki.framework import compiled as FC

    print("=== the route: compile_kernel_to_nir -> NirResult -> CompiledKernel ===")
    for label, obj in (("compile_kernel_to_nir", getattr(FC, "compile_kernel_to_nir", None)),
                       ("compile_to_bir", getattr(FC, "compile_to_bir", None)),
                       ("run_from_bir", getattr(FC, "run_from_bir", None))):
        print(f"\n  {label}:")
        if obj is None:
            print("    <missing>")
            continue
        try:
            print("    signature:", inspect.signature(obj))
        except Exception as e:
            print("    signature unavailable:", e)

    # compile_to_bir is where the IR context gets built; that is the part
    # frontend.compile() wanted and we could not supply.
    src_target = getattr(FC, "compile_to_bir", None)
    if src_target is not None:
        print("\n  source: compile_to_bir")
        try:
            for line in inspect.getsource(src_target).splitlines()[:70]:
                print("   ", line)
        except Exception as e:
            print("    unavailable:", e)

    NR = getattr(FC, "NirResult", None)
    print("\n  NirResult:", NR)
    if NR is not None:
        print("    fields:", sorted(n for n in dir(NR) if not n.startswith("__")))

    print("\n  CompileOptions fields:")
    fields = getattr(CompileOptions, "__dataclass_fields__", {})
    print("   ", sorted(fields) or inspect.signature(CompileOptions))


def find_compiled_kernel(nir, verbose=False):
    """Walk a NirResult for a CompiledKernel or a NEFF path, two levels deep."""
    from nki.compiler.ncc_driver import CompiledKernel

    seen = []
    for name in sorted(n for n in dir(nir) if not n.startswith("__")):
        try:
            val = getattr(nir, name)
        except Exception as e:
            seen.append((name, f"<raised {type(e).__name__}: {e}>"))
            continue
        if callable(val) and not isinstance(val, CompiledKernel):
            seen.append((name, f"<callable {type(val).__name__}>"))
            continue
        seen.append((name, repr(val)[:150]))
        if isinstance(val, CompiledKernel):
            print(f"  found CompiledKernel at nir.{name}")
            return val
        if isinstance(val, str) and val.endswith(".neff") and os.path.exists(val):
            print(f"  found NEFF at nir.{name}: {val}")
        for sub in sorted(n for n in dir(val) if not n.startswith("_")):
            try:
                inner = getattr(val, sub)
            except Exception:
                continue
            if isinstance(inner, CompiledKernel):
                print(f"  found CompiledKernel at nir.{name}.{sub}")
                return inner
    if verbose:
        print("  NirResult contents:")
        for name, val in seen:
            print(f"    {name} = {val}")
    return None


def bench_one(seqlen_kv, n_q_heads, n_kv_heads, warmup=5, iters=50, verbose=False):
    from nki.compiler.ncc_driver import CompileOptions
    from nki.framework.compiled import compile_kernel_to_nir

    args, ref, meta = _make_gqa_inputs(seqlen_kv=seqlen_kv,
                                       n_q_heads=n_q_heads,
                                       n_kv_heads=n_kv_heads)
    q_t, k_t, v_t = args[0], args[1], args[2]
    inputs = kernel_inputs(args)
    if verbose:
        print("  inputs:", {k: getattr(v, "shape", v) for k, v in inputs.items()})

    with tempfile.TemporaryDirectory(prefix="nki_pf_") as wd:
        opts = CompileOptions(target="trn1", artifacts_dir=wd,
                              output_path=os.path.join(wd, "kernel.neff"))

        t0 = time.time()
        nir = compile_kernel_to_nir(K, inputs=inputs, compile_opts=opts)
        frontend_s = time.time() - t0
        if verbose:
            print(f"  NirResult: {type(nir).__name__}")

        t0 = time.time()
        compiled = find_compiled_kernel(nir, verbose=verbose)
        if compiled is None:
            raise RuntimeError(
                "no CompiledKernel reachable from NirResult; see the dump above")
        compile_s = time.time() - t0
        if verbose:
            print(f"  CompiledKernel artifacts="
                  f"{getattr(compiled, 'artifacts_dir', '?')}")

        t0 = time.time()
        res = compiled.benchmark(warmup=warmup, iterations=iters,
                                 q=q_t, k=k_t, v=v_t)
        wall = time.time() - t0

    nbytes = hbm_bytes(meta)
    lat = getattr(res, "latency", None) or 0.0
    row = dict(meta, nbytes=nbytes, frontend_s=frontend_s, compile_s=compile_s,
               wall_s=wall, per_run_ms=wall / (warmup + iters) * 1e3,
               latency_us=lat * 1e6,
               min_us=(getattr(res, "latency_min", None) or 0) * 1e6,
               max_us=(getattr(res, "latency_max", None) or 0) * 1e6,
               std_us=(getattr(res, "latency_std", None) or 0) * 1e6,
               mbu=getattr(res, "mbu", None), hfu=getattr(res, "hfu", None))
    if lat:
        row["gbs"] = nbytes / lat / 1e9
        row["pct_peak"] = 100.0 * nbytes / lat / PEAK_HBM_BYTES_S

    row["max_diff"] = None
    outputs = getattr(res, "outputs", None) or {}
    for _name, arr in outputs.items():
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
    show_api()

    print("\n\n--- smoke test: one config ---", flush=True)
    r = bench_one(512, 8, 2, verbose=True)
    print(f"frontend {r['frontend_s']:.1f}s, neff {r['compile_s']:.1f}s, "
          f"55 runs in {r['wall_s']:.2f}s ({r['per_run_ms']:.1f} ms/run)")
    print(HDR, flush=True)
    show(r)

    if r["max_diff"] is None:
        print("\nNOTE: benchmark() returned no matching output; numerics "
              "unverified on this path. Treat timings as provisional.")
    elif r["max_diff"] > 1e-2:
        print("\nSTOP: outputs do not match the reference; timings are moot.")
        return
    if r["latency_us"] <= 0:
        print("\nSTOP: benchmark() reported no latency.")
        return

    print("\n=== Experiment 1: GQA isolation (Hq=8, N=2048; Hkv varies) ===")
    print(HDR, flush=True)
    for nkv in (8, 4, 2, 1):
        show(bench_one(2048, 8, nkv))

    print("\n=== Experiment 2: length scaling (Hq=8, Hkv=2) ===")
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
