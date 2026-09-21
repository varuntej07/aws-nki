"""Probe the compile_kernel path before patching it a third time. Temporary.

Two monkeypatches deep and still not at the device. Before adding a third,
answer the two questions that decide whether this path can work at all, in
cost order:

  A. Does CompiledKernel.benchmark() exist on NKI 0.6.0 and report latency?
     If it does not, Strategy 1 is dead and no amount of TileView patching
     gets us a number. Costs nothing: pure introspection, no compile.

  B. What is the TileView that `q[:, a:b]` returns under the builder, and does
     it carry a route back to the HBM tensor that nl.load() wants? If it holds
     a parent tensor with a .buffer, patch 3 is a one-liner. If the builder
     hands the kernel tile views that are already resident, then the kernel
     body cannot be shared between the @nki.jit path and this one, and
     Strategy 2 (NEFF + neuron-profile) is the honest route.

Run: python builder_probe.py
"""
import os
import sys
import tempfile
import traceback

import numpy as np

sys.path.insert(0, "contributed")
from decode_attention import (            # noqa: E402
    decode_attention_gqa_fwd as K,
    _make_gqa_inputs,
)
from bench_probe import patch_dtype_size, annotate   # noqa: E402


def public(o):
    return sorted(n for n in dir(o) if not n.startswith("__"))


def safe(o, name):
    try:
        v = getattr(o, name)
    except Exception as e:
        return f"<raised {type(e).__name__}: {e}>"
    try:
        return repr(v)[:200]
    except Exception:
        return f"<{type(v).__name__} unreprable>"


# ---------------------------------------------------------------- A
def probe_benchmark_api():
    import inspect
    from nki.compiler.kernel_builder import builder as B

    print("=== A. does the benchmark API exist? ===", flush=True)
    print("builder exports:", [n for n in dir(B) if not n.startswith("_")])

    CK = getattr(B, "CompiledKernel", None)
    print("CompiledKernel:", CK)
    if CK is None:
        print("  NO CompiledKernel class -> Strategy 1 is dead")
        return False

    print("  methods:", public(CK))
    bm = getattr(CK, "benchmark", None)
    if bm is None:
        print("  NO .benchmark() -> Strategy 1 is dead, go to Strategy 2")
        return False
    try:
        print("  benchmark signature:", inspect.signature(bm))
    except Exception as e:
        print("  benchmark signature unavailable:", e)
    doc = (getattr(bm, "__doc__", "") or "").strip().splitlines()
    print("  benchmark doc:", doc[0] if doc else "<none>")
    return True


# ---------------------------------------------------------------- B
def probe_tileview():
    from nki.compiler.kernel_builder.builder import compile_kernel
    from nki.compiler.ncc_driver import CompileOptions
    import nki.language as nl

    print("\n=== B. what does the builder actually hand the kernel? ===",
          flush=True)

    args, _ref, _meta = _make_gqa_inputs(seqlen_kv=512, n_q_heads=8,
                                         n_kv_heads=2)
    q_t, k_t, v_t = args[0], args[1], args[2]

    def probe(q, k, v):
        print("\n-- the input view --")
        print("q type mro:", type(q).__mro__)
        print("q attrs:", public(q))
        for a in ("buffer", "shape", "dtype", "tensor", "base", "parent"):
            print(f"  q.{a} = {safe(q, a)}")

        print("\n-- nl.load on the whole tensor --")
        try:
            whole = nl.load(q)
            print("  OK:", type(whole).__name__, getattr(whole, "shape", "?"))
        except Exception as e:
            print(f"  FAILED {type(e).__name__}: {e}")

        print("\n-- the slice --")
        sub = q[:, 0:4]
        print("sub type mro:", type(sub).__mro__)
        print("sub attrs:", public(sub))
        for a in public(sub):
            print(f"  sub.{a} = {safe(sub, a)}")
        print("sub private:", [n for n in dir(sub)
                               if n.startswith("_") and not n.startswith("__")])
        for a in (n for n in dir(sub)
                  if n.startswith("_") and not n.startswith("__")):
            print(f"  sub.{a} = {safe(sub, a)}")

        print("\n-- nl.load on the slice --")
        try:
            got = nl.load(sub)
            print("  OK:", type(got).__name__, getattr(got, "shape", "?"))
        except Exception as e:
            print(f"  FAILED {type(e).__name__}: {e}")

        raise SystemExit("PROBE_DONE")

    annotate(probe)
    with tempfile.TemporaryDirectory(prefix="nki_probe_") as wd:
        opts = CompileOptions(target="trn1", artifacts_dir=wd,
                              output_path=os.path.join(wd, "probe.neff"))
        try:
            compile_kernel(probe, inputs={"q": q_t, "k": k_t, "v": v_t},
                           compile_opts=opts)
        except SystemExit as e:
            print(f"\n({e})")


if __name__ == "__main__":
    print("patched dtype_size in:", patch_dtype_size(), flush=True)
    alive = probe_benchmark_api()
    try:
        probe_tileview()
    except Exception:
        traceback.print_exc()
    if not alive:
        print("\nVERDICT: no benchmark API. Stop patching; use Strategy 2.")
