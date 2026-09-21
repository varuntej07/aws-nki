"""Find the supported route to a CompiledKernel, or failing that, a NEFF.

compile_kernel is finished. The fifth missing attribute was src._vector_offset,
a private NkiTensor field for indirect DMA that TileView has no concept of.
Setting it to None would be asserting "this is not a gather" on the library's
behalf: true for this kernel, but emulation rather than translation, and in
private internals, where a wrong guess compiles cleanly and reports a number
that is wrong.

jit_probe settled two things. neuron-profile is deprecated and removed in
favour of neuron-explorer, so the Strategy 2 in bench_probe's docstring names
a tool that no longer exists. And 7b33760 holds: calls 2..6 sit flat at
~1.54 s against 10.1 s for the first, so there is a 6.6x cache but 1.54 s is
not kernel latency for N=512.

Two questions left, best payoff first:

  E. CompiledKernel.from_frontend(frontend_result, compile_opts) wants a
     frontend CompilationResult holding an MLIR module. SOMETHING in the
     package builds one, because that is how @nki.jit reaches the device.
     Find it and we get .benchmark() with latency, MBU and HFU on the
     supported path, with all four monkeypatches deleted and the kernel
     untouched. That is strictly the best outcome available, so it is worth
     one grep before settling for the profiler.

  F. If E is a dead end: get the NEFF on disk. It is not in the usual cache
     roots, so try NEURON_FRAMEWORK_DEBUG and an explicit compile cache dir,
     then hand it to neuron-explorer.

Read-only apart from the kernel's own artifacts. Run: python neff_probe.py
"""
import os

# Must precede any nki import: these are read at framework init.
DUMP = os.path.abspath("_neff_dump")
os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")
os.environ.setdefault("NEURON_COMPILE_CACHE_URL", DUMP)
os.makedirs(DUMP, exist_ok=True)

import importlib.util                     # noqa: E402
import subprocess                         # noqa: E402
import sys                                # noqa: E402
import time                               # noqa: E402
import traceback                          # noqa: E402

sys.path.insert(0, "contributed")


def sh(cmd, limit=60):
    print(f"\n$ {' '.join(cmd)}")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as e:
        print("  failed:", e)
        return ""
    text = (out.stdout or "") + (out.stderr or "")
    lines = text.splitlines()
    for line in lines[:limit]:
        print("   ", line)
    if len(lines) > limit:
        print(f"    ... ({len(lines) - limit} more lines)")
    return text


# ---------------------------------------------------------------- E
def probe_compilation_result():
    print("=== E. who builds a frontend CompilationResult? ===")
    spec = importlib.util.find_spec("nki")
    root = list(spec.submodule_search_locations)[0]
    print("  nki package:", root)

    sh(["grep", "-rn", "--include=*.py", "-e", "class CompilationResult",
        "-e", "CompilationResult(", root], limit=40)
    sh(["grep", "-rn", "--include=*.py", "from_frontend", root], limit=25)

    print("\n--- nki.framework contents ---")
    fw = os.path.join(root, "framework")
    if os.path.isdir(fw):
        for dirpath, _dirnames, filenames in os.walk(fw):
            rel = os.path.relpath(dirpath, root)
            for fn in sorted(filenames):
                if fn.endswith(".py"):
                    print("   ", os.path.join(rel, fn))
    else:
        print("    <no framework dir>")

    # the numpy standalone path is what a plain K(*args) call takes
    sh(["grep", "-rn", "--include=*.py", "-e", "def compile", "-e", "def trace",
        "-e", "def _compile", fw], limit=40)


# ---------------------------------------------------------------- F
def probe_neff_on_disk():
    print("\n\n=== F. get a NEFF on disk ===")
    from decode_attention import (            # noqa: E402
        decode_attention_gqa_fwd as K,
        _make_gqa_inputs,
    )
    from jit_probe import newest_neffs        # noqa: E402

    args, _ref, _meta = _make_gqa_inputs(seqlen_kv=512, n_q_heads=8,
                                         n_kv_heads=2)
    since = time.time()
    print(f"  NEURON_FRAMEWORK_DEBUG={os.environ['NEURON_FRAMEWORK_DEBUG']}")
    print(f"  NEURON_COMPILE_CACHE_URL={os.environ['NEURON_COMPILE_CACHE_URL']}")
    t0 = time.perf_counter()
    K(*args)
    print(f"  one call: {(time.perf_counter() - t0) * 1e3:.1f} ms")

    roots = [DUMP, os.getcwd(), "/tmp", "/var/tmp",
             os.path.expanduser("~")]
    hits = newest_neffs(roots, since, limit=20)
    if not hits:
        print("\n  still no .neff/.ntff under", roots)
        return None
    print()
    for mtime, size, path in hits:
        stamp = time.strftime("%H:%M:%S", time.localtime(mtime))
        print(f"  {size / 1024:9.1f} KiB  {stamp}  {path}")
    neffs = [p for _m, _s, p in hits if p.endswith(".neff")]
    return neffs[0] if neffs else None


def probe_explorer(neff):
    print("\n\n=== neuron-explorer ===")
    sh(["neuron-explorer", "--help"], limit=30)
    sh(["neuron-explorer", "capture", "--help"], limit=40)
    if neff:
        print(f"\n  a NEFF to capture against: {neff}")
    else:
        print("\n  no NEFF found; capture not attempted")


if __name__ == "__main__":
    neff = None
    for fn in (probe_compilation_result,):
        try:
            fn()
        except Exception:
            traceback.print_exc()
    try:
        neff = probe_neff_on_disk()
    except Exception:
        traceback.print_exc()
    try:
        probe_explorer(neff)
    except Exception:
        traceback.print_exc()
