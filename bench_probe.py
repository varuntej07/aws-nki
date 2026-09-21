"""Probe: can we compile once and benchmark many times?

Temporary. Delete before opening the PR.

The public path (calling a @nki.jit kernel with numpy arrays) recompiles on
every single call: nki/framework/compiled.py passes enable_cache=False to
compile_kernel_to_nir. That made "latency" come out at ~2 seconds for a 1 MB
kernel, which is compile time, not execution time.

This probe tests the level below: compile_kernel() -> CompiledKernel once,
then CompiledKernel.benchmark() many times. If it works we get real device
latency plus the SDK's own mbu (memory bandwidth utilization), which also
cross-checks the traffic model in _hbm_bytes().
"""
import sys
import time
import tempfile
import traceback

import numpy as np

sys.path.insert(0, "contributed")
from decode_attention import (            # noqa: E402
    decode_attention_gqa_fwd as K,
    _make_gqa_inputs,
    _hbm_bytes,
    PEAK_HBM_BYTES_S,
)


def main():
    args, ref, meta = _make_gqa_inputs(seqlen_kv=512, n_q_heads=8, n_kv_heads=2)
    q_t, k_t, v_t, nq, nkv, scale = args
    print("shapes", q_t.shape, k_t.shape, v_t.shape)
    print("bytes per call", _hbm_bytes(meta))

    from nki.compiler.kernel_builder.builder import compile_kernel
    from nki.compiler.ncc_driver import CompileOptions

    with tempfile.TemporaryDirectory(prefix="nki_bench_") as wd:
        opts = CompileOptions(
            target="trn1",                 # gen2: inf2 and trn1
            artifacts_dir=wd,
            output_path=wd + "/kernel.neff",
        )

        t0 = time.time()
        compiled = compile_kernel(
            K.func,
            inputs={"q": q_t, "k": k_t, "v": v_t},
            compile_opts=opts,
            return_outputs=True,
            n_q_heads=nq,
            n_kv_heads=nkv,
            softmax_scale=scale,
        )
        print("COMPILED once in %.1fs -> %s"
              % (time.time() - t0, type(compiled).__name__))

        # The decisive number. 55 runs should take about a second if the
        # compile is genuinely amortised, and about two minutes if it is not.
        t0 = time.time()
        res = compiled.benchmark(warmup=5, iterations=50,
                                 q=q_t, k=k_t, v=v_t)
        wall = time.time() - t0
        print("benchmark wall %.2fs for 55 runs" % wall)
        print("  -> %.1f ms per run" % (wall / 55 * 1e3))

        def us(x):
            return None if x is None else x * 1e6

        print("latency_us  ", us(res.latency))
        print("latency_min ", us(res.latency_min))
        print("latency_max ", us(res.latency_max))
        print("latency_std ", us(res.latency_std))
        print("mbu         ", res.mbu)
        print("hfu         ", res.hfu)
        print("outputs     ", list(res.outputs.keys()))

        if res.latency:
            gbs = _hbm_bytes(meta) / res.latency / 1e9
            print("GB/s        %.1f" % gbs)
            print("%% of core peak %.1f"
                  % (100.0 * gbs * 1e9 / PEAK_HBM_BYTES_S))

        # If the outputs come back real, this path also validates numerics.
        for name, arr in res.outputs.items():
            a = np.asarray(arr)
            if a.shape == ref.shape:
                d = float(np.abs(a.astype(np.float32) - ref).max())
                print("max|diff| on %s = %.3e  %s"
                      % (name, d, "PASS" if d < 1e-2 else "FAIL"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
