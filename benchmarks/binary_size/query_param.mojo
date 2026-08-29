"""Binary-size gate: what a late-bound query parameter costs.

Nearly `query_scan.mojo`'s query and plan — same fused predicate, same
projection, same Parquet leaf. The differences are the late-bound values:

    scan(args.get("src"), sch)                      # path, from argv
    Gt(col("a", int64), param("min-a", int64))      # predicate, from argv

so the binary links the `Param[Int64Type]` node, the `Bindings` map the value
travels in, and `marrow.utils.argparse` — the parser, the `--help` renderer and
the string-to-scalar conversion at the boundary. The delta against
`query_scan` is what "this value arrives at run time" costs an AOT query.

    pixi run binary_size query_param query_scan

**Ported from the old expression package on 2026-08-29, and it measures
materially less than it did. The recorded baseline is stale and not
comparable.**

The old package ended this program with `plan.execute_cli()`, and the
measurement was mostly about that tail:

    query_scan_typed                              2,025,432
    query_param, writers gated out                2,222,132   +196,700
    query_param, writers linked (default build)   2,794,420   +768,988

(all three pre-port, and none of them reproducible on this tree)

**None of that layer exists in `marrow.expr`**: no `execute_cli`, no
`parse_params`, no `--describe` JSON, no `-o` / `--format` output-writer
dispatch, no `PathSpec`, no `register_param` registry, and therefore no
`CLI_WRITERS_ENABLED` gating the Parquet and IPC writers behind it. Concretely,
this gate no longer measures:

- the +572,288 of writer linkage (`marrow::parquet::writer` and the codec
  layer), nor the flag that gated it;
- the ~87 KB second instantiation of the `DynArray`/`DynScalar` dispatch ladder
  that `_write_cli_output`'s own `print(result)` call site cost;
- `--describe`, and the output-format dispatch generally.

What survived, and is measured here: argv parsing and `--help`, now through
`marrow/utils/argparse.mojo` — the leaf module `parse_params`/`render_usage`
were extracted into — plus the parameter node itself.

**The path is no longer a `param`.** `param()` is `Param[T: NumericType]`;
there is no `StringParam` and `ParquetScan` takes a plain `String`, so the path
is late-bound through the parser instead of through the plan. `Bindings` is a
`Dict[String, DynScalar]` passed to `execute`, so the plan itself stays
immutable — the old process-global parameter registry is gone with the rest.
"""

from std.sys import argv

from marrow.dtypes import field, int64, string
from marrow.expr import col, param, scan
from marrow.expr import Gt
from marrow.expr import DynValue
from marrow.scalars import Int64Scalar
from marrow.schema import schema
from marrow.utils.argparse import ArgumentParser


def main() raises:
    var parser = ArgumentParser(
        String("query_param"), description=String("Run a compiled plan.")
    )
    parser.option(
        String("min-a"),
        metavar=String("N"),
        help=String("keep rows whose `a` exceeds this"),
    )
    parser.positional(String("src"), help=String("input Parquet file"))

    var raw = argv()
    var tail = List[String](capacity=len(raw))
    for i in range(1, len(raw)):
        tail.append(String(raw[i]))
    var args = parser.parse(tail)

    if args.help_requested:
        print(parser.help_text())
    else:
        var sch = schema(
            [field("a", int64), field("b", int64), field("name", string)]
        )
        var values: List[DynValue] = [col("a", int64)]
        var plan = (
            scan(args.get(String("src")), sch^)
            .filter(Gt(col("a", int64), param("min-a", int64)))
            .project(["a"], values^)
        )
        var bound = Int64Scalar(Int64(args.get_int(String("min-a"))))
        print(plan.execute(bindings={"min-a": bound^.to_dyn()}))
