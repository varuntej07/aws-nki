"""Strategy 2 groundwork: can the @nki.jit path itself give us a number?

bench_probe is four translations deep into the compile_kernel path. Each one
was real (TileView genuinely knows the thing, it just spells it differently),
but there is no bound on how many remain, so build the alternative now rather
than after the next AttributeError.

Strategy 2 needs a NEFF, and compile_kernel is the path that is broken, so the
NEFF has to come from the jit path that already runs correctly on device.
Three questions:

  1. Does call N cost less than call 1? Commit 7b33760 dropped the latency
     benchmark because the standalone path looked like it recompiled every
     call. If a compile cache actually kicks in after the first call, the
     cheapest honest benchmark is just timing calls 2..N and no NEFF spelunking
     is needed at all.
  2. Where does the NEFF land? /var/tmp/neuron-compile-cache is the Neuron
     default. If it is there, neuron-profile can read device-side metrics off
     it without any of the builder machinery.
  3. Is the neuron-profile CLI actually installed on this box?

Read-only apart from whatever the kernel itself writes. Run: python jit_probe.py
"""
import os
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, "contributed")
from decode_attention import (            # noqa: E402
    decode_attention_gqa_fwd as K,
    _make_gqa_inputs,
)

CACHE_ROOTS = ["/var/tmp/neuron-compile-cache",
               os.path.expanduser("~/.cache/neuron"),
               "/tmp"]


def newest_neffs(roots, since, limit=10):
    """NEFF/NTFF files modified after `since`, newest first."""
    hits = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath.count(os.sep) - root.count(os.sep) > 6:
                dirnames[:] = []
                continue
            for fn in filenames:
                if not fn.endswith((".neff", ".ntff")):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if st.st_mtime >= since:
                    hits.append((st.st_mtime, st.st_size, p))
    hits.sort(reverse=True)
    return hits[:limit]


def probe_env():
    print("=== env ===")
    for k in sorted(os.environ):
        if k.startswith("NEURON") or k.startswith("NKI"):
            print(f"  {k}={os.environ[k]}")
    print("\n=== tooling ===")
    for tool in ("neuron-profile", "neuron-ls", "neuron-monitor"):
        path = shutil.which(tool)
        print(f"  {tool:16s} {path or '<not installed>'}")
        if path and tool == "neuron-profile":
            try:
                out = subprocess.run([path, "--help"], capture_output=True,
                                     text=True, timeout=30)
                head = (out.stdout or out.stderr).splitlines()[:15]
                print("    " + "\n    ".join(head))
            except Exception as e:
                print("    --help failed:", e)

    print("\n=== cache roots ===")
    for root in CACHE_ROOTS:
        print(f"  {root:36s} {'exists' if os.path.isdir(root) else '<absent>'}")


def probe_call_timing(calls=6):
    print("\n=== does call N get cheaper than call 1? ===", flush=True)
    args, ref, meta = _make_gqa_inputs(seqlen_kv=512, n_q_heads=8, n_kv_heads=2)
    print(f"  config: {meta['seqlen_kv']=} {meta['n_q_heads']=} "
          f"{meta['n_kv_heads']=}", flush=True)

    t_start = time.time()
    times = []
    out = None
    for i in range(calls):
        t0 = time.perf_counter()
        out = K(*args)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  call {i + 1}: {dt * 1e3:9.2f} ms", flush=True)

    out = np.asarray(out).astype(np.float32)
    print(f"  max|diff| vs reference: {np.abs(out - ref).max():.3e}")

    first, rest = times[0], times[1:]
    if rest:
        best = min(rest)
        print(f"\n  first {first * 1e3:.1f} ms, best of rest {best * 1e3:.1f} ms, "
              f"ratio {first / best:.1f}x")
        if best < first / 10:
            print("  -> a cache IS kicking in. Timing calls 2..N is the "
                  "cheapest honest benchmark; no NEFF needed.")
        elif best > 0.2:
            print("  -> still ~recompiling per call, which is what 7b33760 "
                  "found. Need the NEFF route.")
    return t_start


def probe_artifacts(since):
    print("\n=== where did the artifacts land? ===")
    hits = newest_neffs(CACHE_ROOTS, since)
    if not hits:
        print("  no .neff/.ntff written under", CACHE_ROOTS)
        print("  -> try NEURON_FRAMEWORK_DEBUG=1 or a NEURON_CC_FLAGS dump dir")
        return
    for mtime, size, path in hits:
        print(f"  {size / 1024:9.1f} KiB  {time.strftime('%H:%M:%S', time.localtime(mtime))}  {path}")


if __name__ == "__main__":
    try:
        probe_env()
        since = probe_call_timing()
        probe_artifacts(since)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
