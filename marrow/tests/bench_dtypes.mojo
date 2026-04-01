"""Benchmark: runtime dispatch overhead — Variant-based ArrowType.

`dtypes_variant.ArrowType` exposes `byte_width()` on a type-erased handle
using a linear `isa[T]()` chain over up to 13 arms (O(n) worst case).

Two conditions:

  sum    — sum byte_width() across all 13 types (dispatch cost only)
  branch — branch on byte_width() result, 1000 loops per iter

Run with: pixi run bench_mojo
"""

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep

import marrow.dtypes_variant as vd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _variant_types() -> List[vd.ArrowType]:
    var l = List[vd.ArrowType](capacity=13)
    l.append(vd.NullType())
    l.append(vd.BoolType())
    l.append(vd.Int8Type())
    l.append(vd.Int16Type())
    l.append(vd.Int32Type())
    l.append(vd.Int64Type())
    l.append(vd.UInt8Type())
    l.append(vd.UInt16Type())
    l.append(vd.UInt32Type())
    l.append(vd.UInt64Type())
    l.append(vd.Float32Type())
    l.append(vd.Float64Type())
    l.append(vd.StringType())
    return l^


# ---------------------------------------------------------------------------
# Benchmark functions — plain sum (dispatch cost only)
# ---------------------------------------------------------------------------

@parameter
def bench_variant_byte_width(mut b: Bencher, types: List[vd.ArrowType]) raises:
    @always_inline
    @parameter
    def call_fn() raises:
        var total = 0
        for i in range(len(types)):
            total += types[i].byte_width()
        keep(total)

    b.iter[call_fn]()


# ---------------------------------------------------------------------------
# Benchmark functions — branch on byte_width result (1000 loops per iter)
# ---------------------------------------------------------------------------

@parameter
def bench_variant_branch(mut b: Bencher, types: List[vd.ArrowType]) raises:
    @always_inline
    @parameter
    def call_fn() raises:
        var wide = 0
        var narrow = 0
        for _ in range(1000):
            for i in range(len(types)):
                var w = types[i].byte_width()
                if w >= 4:
                    wide += w
                else:
                    narrow += w
        keep(wide + narrow)

    b.iter[call_fn]()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() raises:
    var m = Bench(BenchConfig(num_repetitions=2000))

    var variant_all = _variant_types()

    m.bench_with_input[List[vd.ArrowType], bench_variant_byte_width](
        BenchId("variant/sum", "all-13-types"),
        variant_all,
    )
    m.bench_with_input[List[vd.ArrowType], bench_variant_branch](
        BenchId("variant/branch", "all-13-types x1000"),
        variant_all,
    )

    m.dump_report()
