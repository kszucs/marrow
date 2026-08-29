"""Binary-size gate: the fused **expression families** outside numeric.

One projection touching string (`Like`), conditional (`Coalesce`), cast
(`NumericCast`) and temporal (`TemporalGt`) nodes, over the same relational
shape as `query_streaming.mojo`. The delta against that gate is what those
families cost together.

**Deliberately one gate for several families, not one gate each.** Each gate is
a full `-O3` build, and a suite nobody runs measures nothing (see Q0.8).
Combining them gives up the ability to attribute a regression to one family,
but it does catch a regression in any of them — and until this landed, none was
linked by *any* gate. If one family later needs its own attribution, split it
out then.

    pixi run binary_size query_exprs

**Ported from the old expression package on 2026-08-29, and it covers less
than it did.** The recorded baseline predates the port and is stale — and it
is *not* comparable to the new number, because two of the five families this
gate was written for do not exist in `marrow.expr`:

- **Membership is gone entirely.** There is no `IsIn` node in either lane, and
  no set-membership value anywhere in `marrow/expr/`. Nothing in this suite
  links `marrow.kernels.membership` any more.
- **Temporal is represented by a comparison, not by field extraction.**
  The old package had `Year`, which pulled in the calendar arithmetic in
  `marrow/kernels/temporal.mojo`; `marrow.expr` has no `Year`, `Month`, `Day`
  or `DateTrunc`. `TemporalGt` keeps a temporal *leaf* in the gate — the
  `TemporalColumn` lane and the temporal comparison arm — but it is a
  different, and much smaller, thing. **`kernels/temporal.mojo` is now linked
  by no gate in this directory.**

Restore both when the nodes land, and re-record rather than comparing across
the gap.
"""

from marrow.builders import TimestampBuilder, array
from marrow.dtypes import Float64Type, int64, second, string, timestamp
from marrow.expr import col, table
from marrow.expr import NumericCast
from marrow.expr import Coalesce, TemporalGt
from marrow.expr import Like
from marrow.expr import DynValue
from marrow.tabular import record_batch


def main() raises:
    var a = array([1, 5, 3], int64)
    var b = array([4, 4, 4], int64)
    var s = array([String("ab"), "bc", "cd"])
    var pat = array([String("a%"), "b%", "c%"])
    var tb = TimestampBuilder(timestamp(second), 3)
    tb.append(Int64(0))
    tb.append(Int64(1))
    tb.append(Int64(2))
    var ts = tb.finish()
    var batch = record_batch(
        [a.copy(), b.copy(), s.copy(), pat.copy(), ts.copy()],
        names=["a", "b", "s", "pat", "ts"],
    )

    var values: List[DynValue] = [
        Like(col("s", string), col("pat", string)),
        Coalesce(col("a", int64), col("b", int64)),
        NumericCast[Float64Type](col("a", int64)),
        TemporalGt(col("ts", timestamp(second)), col("ts", timestamp(second))),
    ]
    print(
        table(batch^).project(["lk", "co", "cast", "later"], values^).execute()
    )
