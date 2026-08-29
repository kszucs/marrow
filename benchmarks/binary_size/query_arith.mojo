"""Binary-size gate: fused **arithmetic**, which nothing else measures.

`SELECT (a + b) * a - b FROM t WHERE a > b` — the same relational shape as
`query_streaming.mojo`, so the delta between the two is exactly what the
arithmetic nodes (`Add`/`Sub`/`Mul` over `NumericBinary`) cost on top of a
column reference and a comparison.

**This gate exists because that cost was invisible.** Q0.4 rewrote all twelve of
the interpreter's binary arms and the fused gates came back byte-identical —
true, and worthless: neither of them contains a single arithmetic expression.
A change to the fused numeric algebra could regress arbitrarily without any gate
noticing.

    pixi run binary_size query_arith

**Ported from the old expression package on 2026-08-29** — the arithmetic
itself carries over unchanged (`marrow/expr/comptime/numeric.mojo` keeps
`Add`/`Sub`/`Mul` and the `+ - *` sugar on `NumericValue`), so this gate
measures exactly what it did before. The only difference is that `project`
derives its output schema instead of taking one; see `query_streaming.mojo`.
The recorded baseline predates the port and is stale.

**What the new package does not have**, and this gate therefore no longer
touches even in principle: `Div`, `Floordiv`, `Mod`, `Pow`, `Neg`, `Abs`,
`Sign`, `Floor`, `Ceil`, `Round`, `Sqrt`, `Exp` and `Ln`. The old package had
them and this gate never named them either, so nothing is lost *here* — but the
fused numeric algebra is smaller than it was, and part of any drop against the
old number is that, not a win.
"""

from marrow.builders import array
from marrow.dtypes import int64
from marrow.expr.builders import col, table
from marrow.expr.`comptime`.numeric import Gt
from marrow.expr.logical import DynValue
from marrow.tabular import record_batch


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["a", "b"])

    var values: List[DynValue] = [
        (col("a", int64) + col("b", int64)) * col("a", int64) - col("b", int64)
    ]
    print(
        table(batch^)
        .filter(Gt(col("a", int64), col("b", int64)))
        .project(["z"], values^)
        .execute()
    )
