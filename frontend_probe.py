"""Is there a supported bridge from @nki.jit to CompiledKernel? Temporary.

builder_probe.py answered two things:
  A. CompiledKernel.benchmark() exists and returns outputs + latency. The
     endgame is real.
  B. The builder hands the kernel a TileView, not an HBM tensor. nl.load()
     asserts on src.buffer; TileView calls the same thing memspace. Patchable,
     but it is the third patch on a path the kernel was never written for.

The exports list named CompiledKernel.from_frontend. The @nki.jit kernel
ALREADY runs correctly on device as a plain call. If from_frontend turns that
frontend into a CompiledKernel, we get compile-once plus .benchmark() with the
kernel untouched and both monkeypatches deleted.

So, in cost order:
  C. What do from_frontend / compile_and_execute / build_kernel take, and does
     the @nki.jit object expose a compiled form itself?
  D. Fallback if C is a dead end: what exactly does nl.load's is_hbm() accept,
     and does MemSpace.SharedHbm satisfy it? That sizes patch 3.

Run: python frontend_probe.py
"""
import inspect
import sys
import traceback

sys.path.insert(0, "contributed")
from decode_attention import decode_attention_gqa_fwd as K   # noqa: E402
from bench_probe import patch_dtype_size                     # noqa: E402


def public(o):
    return sorted(n for n in dir(o) if not n.startswith("__"))


def describe(obj, label):
    print(f"\n--- {label} ---")
    if obj is None:
        print("  <missing>")
        return
    try:
        print("  signature:", inspect.signature(obj))
    except Exception as e:
        print("  signature unavailable:", e)
    doc = inspect.getdoc(obj)
    print("  doc:", ("\n       ".join(doc.splitlines()[:12]) if doc else "<none>"))


def source_of(obj, label, limit=40):
    print(f"\n--- source: {label} ---")
    try:
        src = inspect.getsource(obj).splitlines()
        for line in src[:limit]:
            print("   ", line)
        if len(src) > limit:
            print(f"    ... ({len(src) - limit} more lines)")
    except Exception as e:
        print("  unavailable:", e)


# ---------------------------------------------------------------- C
def probe_frontend_bridge():
    import nki
    from nki.compiler.kernel_builder import builder as B
    from nki.compiler.ncc_driver import CompiledKernel

    print("=== C. is there a supported frontend -> CompiledKernel bridge? ===")

    describe(getattr(CompiledKernel, "from_frontend", None),
             "CompiledKernel.from_frontend")
    source_of(getattr(CompiledKernel, "from_frontend", None),
              "CompiledKernel.from_frontend")

    for name in ("compile_kernel", "build_kernel", "compile_and_execute"):
        describe(getattr(B, name, None), f"builder.{name}")

    describe(getattr(CompiledKernel, "execute", None), "CompiledKernel.execute")
    describe(getattr(CompiledKernel, "run", None), "CompiledKernel.run")

    print("\n--- the @nki.jit object ---")
    print("  type:", type(K), type(K).__mro__)
    print("  attrs:", public(K))
    for a in public(K):
        if a.startswith("_"):
            continue
        try:
            v = getattr(K, a)
        except Exception as e:
            print(f"  K.{a} = <raised {type(e).__name__}: {e}>")
            continue
        print(f"  K.{a} = {repr(v)[:160]}")
    describe(getattr(nki, "jit", None), "nki.jit")

    print("\n--- ExecutionResult ---")
    ER = None
    for modname in ("nki.compiler.ncc_driver",):
        mod = sys.modules.get(modname) or __import__(modname, fromlist=["x"])
        ER = getattr(mod, "ExecutionResult", None)
        if ER is not None:
            break
    if ER is None:
        print("  <not found by name>")
    else:
        print("  fields:", public(ER))
        doc = inspect.getdoc(ER)
        print("  doc:", (doc.splitlines()[0] if doc else "<none>"))


# ---------------------------------------------------------------- D
def probe_is_hbm():
    print("\n\n=== D. fallback: what does is_hbm() accept? ===")
    import nki.language as nl
    import nki.language._core as C
    from nki.compiler.kernel_builder.builder import MemSpace

    is_hbm = getattr(C, "is_hbm", None)
    source_of(is_hbm, "nki.language._core.is_hbm", limit=25)

    print("\n--- the buffer objects nl exposes ---")
    for name in ("shared_hbm", "hbm", "private_hbm", "sbuf", "psum"):
        b = getattr(nl, name, None)
        print(f"  nl.{name} = {repr(b)[:120]}  type={type(b).__name__}")

    print("\n--- MemSpace members ---")
    try:
        print("  ", list(MemSpace))
    except Exception as e:
        print("  unavailable:", e)

    if is_hbm is None:
        print("\n  is_hbm not found; cannot test")
        return
    print("\n--- does is_hbm accept each? ---")
    cands = [("nl.shared_hbm", getattr(nl, "shared_hbm", None)),
             ("nl.hbm", getattr(nl, "hbm", None)),
             ("MemSpace.SharedHbm", getattr(MemSpace, "SharedHbm", None))]
    for label, val in cands:
        if val is None:
            print(f"  {label:22s} <missing>")
            continue
        try:
            print(f"  {label:22s} -> {is_hbm(val)}")
        except Exception as e:
            print(f"  {label:22s} -> raised {type(e).__name__}: {e}")

    source_of(getattr(C, "load", None), "nki.language._core.load", limit=45)


if __name__ == "__main__":
    print("patched dtype_size in:", patch_dtype_size(), flush=True)
    for fn in (probe_frontend_bridge, probe_is_hbm):
        try:
            fn()
        except Exception:
            traceback.print_exc()
