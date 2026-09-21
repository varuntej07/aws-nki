"""Read the two functions that stand between us and a CompiledKernel.

neff_probe found the producers. nki/compiler/frontend.py defines
CompilationResult at line 48 and builds one at 332 and 536, which is exactly
what CompiledKernel.from_frontend wants. nki/framework/compiled.py has
compile_kernel_to_nir at 49 and a compile() at 354 returning
Tuple[FrameworkConfig, str], and that str is plausibly a NEFF path, which
would answer both open questions at once.

It also found that no NEFF reaches disk even with NEURON_FRAMEWORK_DEBUG=1
and an explicit NEURON_COMPILE_CACHE_URL, so neuron-explorer capture has
nothing to chew on. Going through compiled.py is now the only live route as
well as the best one.

This probe only reads source. No compiling, no device. It prints the API
surface of both modules and the specific regions around each producer, so the
next step is written against what the code does rather than its name.

Run: python nir_probe.py
"""
import importlib.util
import os
import subprocess
import sys
import traceback

ROOT = None


def sh(cmd, limit=80):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as e:
        print("  failed:", e)
        return
    text = (out.stdout or "") + (out.stderr or "")
    lines = text.splitlines()
    for line in lines[:limit]:
        print("   ", line)
    if len(lines) > limit:
        print(f"    ... ({len(lines) - limit} more lines)")


def surface(path, label):
    print(f"\n\n########## {label}: API surface ##########")
    sh(["grep", "-n", "-e", "^class ", "-e", "^def ", "-e", "^    def ",
        "-e", "^@", path], limit=90)


def region(path, spec, label):
    print(f"\n----- {label} -----")
    sh(["sed", "-n", spec, path], limit=110)


def main():
    spec = importlib.util.find_spec("nki")
    root = list(spec.submodule_search_locations)[0]
    print("nki package:", root)

    frontend = os.path.join(root, "compiler", "frontend.py")
    compiled = os.path.join(root, "framework", "compiled.py")
    driver = os.path.join(root, "compiler", "ncc_driver.py")

    # --- the thing from_frontend consumes -----------------------------
    surface(frontend, "nki/compiler/frontend.py")
    region(frontend, "40,80p", "CompilationResult definition (~48)")
    region(frontend, "290,345p", "producer 1 (~332)")
    region(frontend, "495,545p", "producer 2 (~536)")

    # --- the framework path a plain K(*args) call takes ----------------
    surface(compiled, "nki/framework/compiled.py")
    region(compiled, "40,120p", "compile_kernel_to_nir (~49)")
    region(compiled, "340,400p", "compile() -> (FrameworkConfig, str)  (~354)")
    region(compiled, "480,530p", "_compile_and_run (~489)")

    # --- and what from_frontend expects on the other side --------------
    region(driver, "255,300p", "CompiledKernel.from_frontend (~264)")

    print("\n\n########## public exports ##########")
    for mod in ("nki.compiler", "nki.compiler.frontend", "nki.framework.compiled"):
        try:
            m = __import__(mod, fromlist=["x"])
        except Exception as e:
            print(f"  {mod}: import failed: {type(e).__name__}: {e}")
            continue
        names = sorted(n for n in dir(m) if not n.startswith("_"))
        print(f"  {mod}: {names}")


if __name__ == "__main__":
    sys.path.insert(0, "contributed")
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
