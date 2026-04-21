import pathlib
import os
import sys
import argparse
import platform
import shlex
import subprocess

def _openblas_flags():
    candidates = []

    prefix_lib = pathlib.Path(sys.prefix) / "lib"
    candidates.append(prefix_lib)

    brew = subprocess.run(
        ["brew", "--prefix", "openblas"],
        capture_output=True,
        check=False,
        text=True,
    )
    if brew.returncode == 0:
        candidates.append(pathlib.Path(brew.stdout.strip()) / "lib")

    for lib_dir in candidates:
        if any(lib_dir.glob("libopenblas*")):
            return f"-L{lib_dir} -lopenblas"

    return "-lopenblas"

def main(argv=None):
    root = pathlib.Path(__file__).resolve().parent
    src = os.path.join(root, "symMatmulRoutines.f90")
    module_name = "symMatmulRoutines"

    # Optimization flags
    flags = {
        "safe": "-O3 -fno-unsafe-math-optimizations -fno-fast-math -funroll-loops -fomit-frame-pointer -flto -fstrict-aliasing",
        "fast": "-Ofast -ffast-math -fno-math-errno -funroll-loops -fomit-frame-pointer -flto -fstrict-aliasing -fno-trapping-math -fno-signaling-nans",
        "ultimate": "-Ofast -funroll-all-loops -ffast-math -fassociative-math -freciprocal-math -fno-signed-zeros -fno-trapping-math -fno-math-errno -fno-protect-parens -fno-rounding-math -fno-signaling-nans -fomit-frame-pointer -frename-registers -fvect-cost-model=cheap -ftree-vectorize -falign-loops=64 -falign-functions=64 -fno-stack-protector -fno-semantic-interposition -flto -fno-plt -Wno-maybe-uninitialized"
    }

    parser = argparse.ArgumentParser(
        description="Compile the Fortran extension used by fast mPOD."
    )
    parser.add_argument(
        "--blas",
        default="auto",
        help=(
            "BLAS/LAPACK linker flags passed to f2py. "
            "Default: auto-detect OpenBLAS, falling back to -lopenblas"
        ),
    )
    parser.add_argument(
        "--opt",
        default="safe",
        choices=("safe", "fast", "ultimate"),
        help="Fortran optimization level. Default: safe",
    )
    args = parser.parse_args(argv)

    level = args.opt

    env = os.environ.copy()
    env["FFLAGS"] = flags[level]
    env["FCFLAGS"] = flags[level]
    python_bin = pathlib.Path(sys.executable).parent
    env["PATH"] = f"{python_bin}{os.pathsep}{env.get('PATH', '')}"
    if platform.system() == "Darwin" and "SDKROOT" not in env:
        sdk = subprocess.run(
            ["xcrun", "--show-sdk-path"],
            capture_output=True,
            check=False,
            text=True,
        ).stdout.strip()
        if sdk:
            env["SDKROOT"] = sdk

    print(f"Using {level} optimization flags:")
    print("FFLAGS:", env["FFLAGS"])
    print("FCFLAGS:", env["FCFLAGS"])
    if "SDKROOT" in env:
        print("SDKROOT:", env["SDKROOT"])

    blas = _openblas_flags() if args.blas == "auto" else args.blas

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "numpy.f2py",
        "-c",
        "-m",
        module_name,
        src,
        *shlex.split(blas),
    ]

    print("Running:", " ".join(shlex.quote(part) for part in cmd))

    # Run inside the folder so the .so lands here
    subprocess.run(cmd, cwd=root, env=env, check=True)

    print("\nFortran extension built successfully.")
    print(f"Stored in: {root}")

if __name__ == "__main__":
    main()
