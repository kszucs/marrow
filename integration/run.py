#!/usr/bin/env python3
"""Run archery integration tests with Marrow.

Usage (from the repo root):

    # C Data + IPC, Marrow-only:
    PYTHONPATH=python pixi run -e dev python -m integration.run --run-c-data --run-ipc

    # Cross-test against arrow-rs (build first with build_arrow_rust task):
    pixi run -e integration run_integration -- --run-c-data --run-ipc --with-rust

    # With C++ Arrow for cross-implementation tests:
    ARROW_CPP_EXE_PATH=/path/to/arrow/build/debug \\
    PYTHONPATH=python pixi run -e integration python -m integration.run \\
        --run-c-data --run-ipc --with-cpp

    # With golden files:
    pixi run -e integration run_integration -- --run-ipc \\
        --gold-dirs /path/to/arrow/testing/data/arrow-ipc-stream/integration/cpp-21.0.0

Prerequisites:
    pixi install -e integration  (installs archery from apache/arrow)
"""

import argparse
import io
import sys

from archery.integration import datagen as _datagen
from archery.integration.runner import run_all_tests
from integration.reporter import Tee, report
from integration.tester import MarrowTester

# Types not yet implemented in Marrow — skip these test cases rather than
# failing.  We monkey-patch datagen so archery marks them as skipped without
# modifying the upstream archery source.
_UNSUPPORTED = {
    'large_binary',       # largebinary / largeutf8
    'binary',             # contains fixed_size_binary fields
    'binary_no_batches',  # contains fixed_size_binary fields
    'binary_zerolength',  # contains fixed_size_binary fields
    'null',               # null type
    'null_trivial',       # null-only schema (no supported columns remain)
    'decimal',            # decimal128
    'decimal256',
    'decimal32',
    'decimal64',
    'datetime',           # date / time / timestamp
    'duration',
    'interval',
    'interval_mdn',       # month_day_nano_interval
    'map',
    'map_non_canonical',
    'nested_large_offsets',   # large list
    'union',
    'dictionary',
    'dictionary_unsigned',
    'nested_dictionary',
    'binary_view',
    'list_view',
    'extension',
    'run_end_encoded',
    'recursive_nested',   # tester layer cannot preserve custom inner-field names
    'custom_metadata',    # schema/field metadata not preserved end-to-end
}

_orig_get_generated = _datagen.get_generated_json_files


def _patched_get_generated_json_files(tempdir=None):
    files = _orig_get_generated(tempdir)
    for f in files:
        if f.name in _UNSUPPORTED:
            f.skip_tester('Mojo')
    return files


_datagen.get_generated_json_files = _patched_get_generated_json_files


def main():
    parser = argparse.ArgumentParser(
        description="Run archery integration tests with Marrow"
    )
    parser.add_argument("--run-ipc", action="store_true",
                        help="Run IPC producer/consumer tests")
    parser.add_argument("--run-c-data", action="store_true",
                        help="Run C Data Interface tests")
    parser.add_argument("--with-cpp", action="store_true",
                        help="Cross-test against C++ Arrow (needs ARROW_CPP_EXE_PATH)")
    parser.add_argument("--with-rust", action="store_true",
                        help="Cross-test against arrow-rs (needs ARROW_RUST_EXE_PATH)")
    parser.add_argument("--with-go", action="store_true",
                        help="Cross-test against arrow-go (needs GOBIN + ARROW_ROOT)")
    parser.add_argument("--no-stop-on-error", action="store_true",
                        help="Continue after failures (default: stop on first error)")
    parser.add_argument("--match", default=None, metavar="PATTERN",
                        help="Only run test cases whose name contains PATTERN")
    parser.add_argument("--gold-dirs", nargs="*", default=None, metavar="DIR",
                        help="Paths to golden .arrow_file directories")
    args = parser.parse_args()

    if not args.run_ipc and not args.run_c_data:
        parser.error("Specify at least one of --run-ipc or --run-c-data")

    testers = [MarrowTester()]
    other_testers = []

    if args.with_cpp:
        try:
            from archery.integration.tester_cpp import CppTester
            other_testers.append(CppTester())
        except Exception as e:
            print(f"Warning: could not load CppTester: {e}", file=sys.stderr)

    if args.with_rust:
        try:
            from archery.integration.tester_rust import RustTester
            other_testers.append(RustTester())
        except Exception as e:
            print(f"Warning: could not load RustTester: {e}", file=sys.stderr)

    if args.with_go:
        try:
            from archery.integration.tester_go import GoTester
            other_testers.append(GoTester())
        except Exception as e:
            print(f"Warning: could not load GoTester: {e}", file=sys.stderr)

    # Tee stdout to a buffer so report() can summarise without blocking the
    # live stream the user sees.
    buf = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, buf)
    try:
        run_all_tests(
            testers=testers,
            other_testers=other_testers,
            run_ipc=args.run_ipc,
            run_c_data=args.run_c_data,
            stop_on_error=not args.no_stop_on_error,
            match=args.match,
            gold_dirs=args.gold_dirs,
        )
    finally:
        sys.stdout = original_stdout
        report(buf.getvalue())


if __name__ == "__main__":
    main()
