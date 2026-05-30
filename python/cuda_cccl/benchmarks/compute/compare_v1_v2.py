#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Compare cuda.compute v1 vs v2 Python benchmarks.

v1 vs v2 is a build-time choice of the cuda-cccl package (CCCL_PYTHON_USE_V2),
not two source trees -- so this runs the same benchmark scripts under two Python
environments (one built as v1, one as v2) and diffs the JSON with nvbench-compare.

Supply the two environments one of two ways:

  --build
      Use uv to create fresh venvs under --venv-root (default /tmp/cccl-bench)
      and editable-install cuda-cccl from this repo into each -- the v2 venv with
      CMAKE_ARGS=-DCCCL_PYTHON_USE_V2=ON. uv fetches Python --python-version
      (default 3.13) automatically. Requires uv on PATH, plus the CUDA build
      toolchain (nvcc, nvrtc-dev, nvjitlink-dev, nvfatbin-dev) in the environment.

  --v1-python PATH --v2-python PATH
      Use interpreters you already built. Each must be the python from a venv
      whose cuda-cccl was built as v1 / v2 respectively.

The script verifies each interpreter's USING_V2 flag before running so a
mis-built environment fails fast instead of silently comparing v1 against v1.

Examples:
  # Build both envs and compare the quick subset of every benchmark
  python compare_v1_v2.py --build --quick

  # One benchmark, against pre-built envs
  python compare_v1_v2.py --v1-python /tmp/v1/bin/python \\
                          --v2-python /tmp/v2/bin/python \\
                          -b reduce/sum

  # Full sweep, cu12 toolkit, device 1
  python compare_v1_v2.py --build --cuda cu12 -d 1
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

# Reuse the axis/quick-config machinery from the py-vs-cpp runner.
from run_benchmarks import (
    SCRIPT_DIR,
    SUPPORTED_BENCHMARKS,
    build_axis_args_from_config,
    get_py_script,
    get_quick_config_entry,
    load_quick_configs,
    print_banner,
    print_section,
)

CCCL_CCCL_DIR = SCRIPT_DIR.parents[1]  # python/cuda_cccl (has pyproject.toml)
RESULTS_DIR = SCRIPT_DIR / "results_v1v2"


def run_streamed(cmd: list, log_path: Path, env: dict | None = None) -> dict:
    """Run a command, streaming combined stdout/stderr to the console and a log.

    Unlike run_benchmarks.run_and_log (which only writes to the file), this echoes
    output live so long steps like ``uv pip install`` and the benchmark runs aren't
    silent.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  $ {shlex.join(cmd)}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Command: {shlex.join(cmd)}\n\n")
        log_file.flush()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
            )
        except Exception as exc:  # noqa: BLE001
            log_file.write(f"\nERROR: failed to execute command.\n{exc}\n")
            print(f"  ERROR: failed to execute command: {exc}", flush=True)
            return {"status": "error", "returncode": None, "error": str(exc)}
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
        proc.wait()

    status = "ok" if proc.returncode == 0 else "failed"
    return {"status": status, "returncode": proc.returncode}


def _apply_elements_override(axis_config: dict, elements_pow2) -> dict:
    """Return a copy of axis_config with any 'Elements' axis set to elements_pow2.

    The value is a pow2 exponent (e.g. 28 => 2**28), matching how Elements axes
    are specified in quick_configs.yaml.
    """
    if elements_pow2 is None:
        return axis_config
    out = dict(axis_config)
    for name in list(out):
        base = name.rsplit("{", 1)[0] if name.endswith("}") else name
        if base == "Elements":
            out[name] = elements_pow2
    return out


def build_py_axis_args(benchmark: str, quick_configs: dict, elements_pow2=None) -> list:
    """Build the per-benchmark nvbench axis args (quick mode only).

    Mirrors run_benchmarks.run_benchmark, but only the Python side: handles both
    the flat axis form and the nested ``benchmarks:`` form in quick_configs.yaml.
    ``elements_pow2`` overrides the Elements axis exponent (e.g. 28) for runs at
    realistic sizes.
    """
    config_entry = get_quick_config_entry(benchmark, quick_configs)
    args = []
    if "benchmarks" in config_entry:
        for bench_name, axis_config in config_entry["benchmarks"].items():
            args.extend(["--benchmark", bench_name])
            args.extend(
                build_axis_args_from_config(
                    _apply_elements_override(axis_config, elements_pow2),
                    for_python=True,
                )
            )
    else:
        args = build_axis_args_from_config(
            _apply_elements_override(config_entry, elements_pow2), for_python=True
        )
    return args


def query_using_v2(python: str) -> bool | None:
    """Return the USING_V2 flag of a cuda-cccl install, or None if unknown."""
    code = "from cuda.compute._build_info import USING_V2; print(int(bool(USING_V2)))"
    proc = subprocess.run(
        [python, "-c", code], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        # _build_info absent on some installs => treated as v1 by the package.
        return None
    out = proc.stdout.strip()
    if out in ("0", "1"):
        return bool(int(out))
    return None


def verify_side(python: str, side: str, want_v2: bool) -> None:
    """Fail fast if an interpreter isn't the expected v1/v2 build."""
    if not Path(python).exists():
        sys.exit(f"ERROR: {side} python not found: {python}")
    got = query_using_v2(python)
    got_label = {True: "v2", False: "v1", None: "unknown (no _build_info)"}[got]
    print(f"  {side}: {python}  -> USING_V2={got_label}")
    # None means _build_info missing, which the package treats as v1.
    effective_v2 = got is True
    if effective_v2 != want_v2:
        sys.exit(
            f"ERROR: {side} interpreter reports {got_label}, expected "
            f"{'v2' if want_v2 else 'v1'}. Build it correctly first "
            f"(v2 needs CMAKE_ARGS=-DCCCL_PYTHON_USE_V2=ON)."
        )


def build_venv(
    venv_path: Path,
    cuda: str,
    use_v2: bool,
    py_version: str,
    jobs: int,
    rebuild: bool,
    log_path: Path,
) -> str:
    """Create a uv venv and editable-install cuda-cccl (v1 or v2). Returns python path.

    Uses uv so the right Python (cuda-bench/cupy top out at 3.13) is fetched
    automatically instead of relying on whatever interpreter launched this script.
    Reuses an existing venv whose cuda-cccl already matches the requested v1/v2
    build unless ``rebuild`` is set.
    """
    side = "v2" if use_v2 else "v1"
    py = str(venv_path / "bin" / "python")

    # Reuse an already-built env unless asked to rebuild.
    if not rebuild and Path(py).exists() and query_using_v2(py) is use_v2:
        print(
            f"Reusing existing {side} venv at {venv_path} "
            f"(USING_V2={use_v2}); pass --rebuild to force."
        )
        return py

    # Stable, per-side build dir. v1 and v2 MUST NOT share one: the wheel tag
    # doesn't encode the v2 flag, so a shared dir reconfigures (and rebuilds
    # LLVM/Clang) every time you switch sides. Keeping it persistent across runs
    # lets a failed build resume incrementally instead of refetching/rebuilding
    # LLVM from scratch. --rebuild wipes it for a clean build.
    build_dir = venv_path.parent / f"build-{side}"
    if rebuild and build_dir.exists():
        print(f"--rebuild: removing {build_dir}")
        shutil.rmtree(build_dir, ignore_errors=True)

    print(
        f"Building {side} venv at {venv_path} "
        f"(python {py_version}, {jobs} jobs, build dir {build_dir}, via uv) ..."
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Only (re)create the venv if it's missing -- don't clobber an existing one
    # (and its installed deps) just because the cuda-cccl build failed last time.
    # uv venv --python X auto-downloads a managed CPython X if none is installed.
    steps = []
    if not Path(py).exists():
        steps.append((["uv", "venv", "--python", py_version, str(venv_path)], None))
    # nvbench-compare runtime deps (until cuda-bench declares them); no-op if present.
    steps.append(
        (
            [
                "uv",
                "pip",
                "install",
                "--python",
                py,
                "colorama",
                "jsondiff",
                "tabulate",
            ],
            None,
        )
    )
    # Editable install carries the v2 flag through scikit-build-core via CMAKE_ARGS.
    # --verbose makes uv stream the build backend (CMake/nvcc/ninja) output instead
    # of hiding it behind a spinner; build.verbose surfaces the per-file compile
    # commands (renamed from cmake.verbose in scikit-build-core >=0.10). SKBUILD
    # logging.level=INFO keeps scikit-build-core chatty too.
    # CMAKE_BUILD_PARALLEL_LEVEL caps ninja's parallelism at `jobs`.
    install_env = os.environ.copy()
    install_env["SKBUILD_LOGGING_LEVEL"] = "INFO"
    install_env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(jobs)
    if use_v2:
        install_env["CMAKE_ARGS"] = (
            install_env.get("CMAKE_ARGS", "") + " -DCCCL_PYTHON_USE_V2=ON"
        ).strip()
    steps.append(
        (
            [
                "uv",
                "pip",
                "install",
                "--verbose",
                "--python",
                py,
                "--config-settings=build.verbose=true",
                # {wheel_tag} is expanded by scikit-build-core (literal here).
                f"--config-settings=build-dir={build_dir}/{{wheel_tag}}",
                "-e",
                f"{CCCL_CCCL_DIR}[bench-{cuda}]",
            ],
            install_env,
        )
    )

    for cmd, env in steps:
        status = run_streamed(cmd, log_path, env=env)
        if status["status"] != "ok":
            sys.exit(
                f"ERROR: {side} venv setup failed (exit {status['returncode']}). "
                f"See {log_path}"
            )
    return py


def run_one_side(
    python: str,
    script: Path,
    json_out: Path,
    device: str,
    axis_args: list,
    log_path: Path,
) -> dict:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [python, str(script), "--json", str(json_out), "--devices", device]
    cmd.extend(axis_args)
    return run_streamed(cmd, log_path)


def nvbench_compare(
    python: str, base_json: Path, test_json: Path, log_path: Path
) -> int:
    """Run nvbench-compare (base=v1, test=v2); stream to console and tee to log."""
    compare_bin = Path(python).parent / "nvbench-compare"
    cmd = [str(compare_bin), "--no-color", str(base_json), str(test_json)]
    status = run_streamed(cmd, log_path)
    return status["returncode"] if status["returncode"] is not None else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cuda.compute v1 vs v2 Python benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported benchmarks:
{chr(10).join(f"  {b}" for b in SUPPORTED_BENCHMARKS)}
""",
    )
    parser.add_argument(
        "-d", "--device", default="0", help="GPU device ID [default: 0]"
    )
    parser.add_argument(
        "-b", "--benchmark", help="Run one benchmark only [default: all]"
    )
    parser.add_argument(
        "-q",
        "--quick",
        action="store_true",
        help="Reduced parameter set (quick_configs.yaml)",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Create and install v1/v2 venvs automatically",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=4,
        help="Parallel build jobs for the native build [default: 4]",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild even if a matching venv already exists",
    )
    parser.add_argument(
        "--python-version",
        default="3.13",
        help="Python version for --build venvs [default: 3.13]. "
        "uv fetches it automatically; keep <=3.13 since "
        "cuda-bench/cupy have no 3.14 wheels.",
    )
    parser.add_argument(
        "--cuda",
        default="cu13",
        help="CUDA suffix for the bench extra [default: cu13] "
        "(cu12/cu13/sysctk12/sysctk13)",
    )
    parser.add_argument(
        "--venv-root",
        default="/tmp/cccl-bench",
        help="Where --build creates venvs [default: /tmp/cccl-bench]",
    )
    parser.add_argument("--v1-python", help="Path to a v1-built python interpreter")
    parser.add_argument("--v2-python", help="Path to a v2-built python interpreter")
    parser.add_argument(
        "--elements",
        type=int,
        default=None,
        help="Override the Elements axis (pow2 exponent, e.g. 28 "
        "for 2**28). Uses quick-config single points for the "
        "other axes.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Output dir for JSON/logs [default: results_v1v2]",
    )
    args = parser.parse_args()

    if not args.build and not (args.v1_python and args.v2_python):
        parser.error("provide --build, or both --v1-python and --v2-python")

    global RESULTS_DIR
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
    # --elements reuses the quick-config single points as a base for other axes.
    use_quick = args.quick or args.elements is not None

    if args.benchmark:
        if args.benchmark not in SUPPORTED_BENCHMARKS:
            print(f"ERROR: Benchmark '{args.benchmark}' not supported.\n")
            for b in SUPPORTED_BENCHMARKS:
                print(f"  {b}")
            sys.exit(1)
        benchmarks = [args.benchmark]
    else:
        benchmarks = SUPPORTED_BENCHMARKS

    quick_configs = load_quick_configs() if use_quick else {}

    print_banner("CCCL v1-vs-v2 Benchmark Comparison")

    # Resolve the two interpreters.
    if args.build:
        venv_root = Path(args.venv_root)
        v1_python = build_venv(
            venv_root / "v1",
            args.cuda,
            False,
            args.python_version,
            args.jobs,
            args.rebuild,
            RESULTS_DIR / "logs" / "venv_v1.log",
        )
        v2_python = build_venv(
            venv_root / "v2",
            args.cuda,
            True,
            args.python_version,
            args.jobs,
            args.rebuild,
            RESULTS_DIR / "logs" / "venv_v2.log",
        )
    else:
        v1_python = args.v1_python
        v2_python = args.v2_python

    print("Verifying environments:")
    verify_side(v1_python, "v1", want_v2=False)
    verify_side(v2_python, "v2", want_v2=True)
    print(f"\nResults dir: {RESULTS_DIR}\n")

    summary = {}
    for bench in benchmarks:
        print_section(f"Benchmark: {bench}")
        script = get_py_script(bench)
        if not script.exists():
            print(f"  SKIP: script not found: {script}")
            summary[bench] = "skipped (no script)"
            continue

        axis_args = (
            build_py_axis_args(bench, quick_configs, args.elements) if use_quick else []
        )
        bench_path = Path(bench)
        v1_json = RESULTS_DIR / "v1" / bench_path.parent / f"{bench_path.name}.json"
        v2_json = RESULTS_DIR / "v2" / bench_path.parent / f"{bench_path.name}.json"
        logs = RESULTS_DIR / "logs" / bench_path.parent

        s1 = run_one_side(
            v1_python,
            script,
            v1_json,
            args.device,
            axis_args,
            logs / f"{bench_path.name}_v1.log",
        )
        s2 = run_one_side(
            v2_python,
            script,
            v2_json,
            args.device,
            axis_args,
            logs / f"{bench_path.name}_v2.log",
        )
        if s1["status"] != "ok" or s2["status"] != "ok":
            print(
                f"  WARNING: run failed (v1={s1['status']}, v2={s2['status']}); "
                f"see logs in {logs}"
            )
            summary[bench] = f"run failed (v1={s1['status']}, v2={s2['status']})"
            continue

        print("  nvbench-compare (base=v1, test=v2):")
        rc = nvbench_compare(
            v1_python, v1_json, v2_json, logs / f"{bench_path.name}_compare.log"
        )
        summary[bench] = "compared" if rc == 0 else f"compare exit {rc}"

    print_banner("Summary")
    for bench in benchmarks:
        print(f"  {bench}: {summary.get(bench, 'not run')}")
    print(f"\nJSON + logs under: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
