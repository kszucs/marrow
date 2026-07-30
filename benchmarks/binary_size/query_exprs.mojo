"""Binary-size gate: the fused **expression families** outside numeric.

One projection touching string (`Like`), conditional (`Coalesce`), membership
(`IsIn`), cast (`NumericCast`) and temporal (`Year`) nodes, over the same
relational shape as `query_streaming.mojo`. The delta against that gate is what
those five families cost together.

**Deliberately one gate for five families, not five gates.** Each gate is a full
`-O3` build, and a suite nobody runs measures nothing (see Q0.8). Combining them
gives up the ability to attribute a regression to one family, but it does catch
a regression in any of them — and until this landed, none of the five was linked
by *any* gate. If one family later needs its own attribution, split it out then.

    pixi run binary_size query_exprs
"""

from marrow.builders import array, TimestampBuilder
from marrow.dtypes import (
    Float64Type,
    int64,
    float64,
    string,
    second,
    timestamp,
    int32,
    field,
)
from marrow.schema import schema
from marrow.tabular import record_batch
from marrow.expr.values import (
    col,
    DynValue,
    NumericCast,
    Like,
    IsIn,
    Coalesce,
    Year,
)
from marrow.expr.relations import InMemoryTable, Project, DynRelation


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

    var values = List[DynValue]()
    values.append(DynValue(Like(col("s", string), col("pat", string))))
    values.append(DynValue(Coalesce(col("a", int64), col("b", int64))))
    values.append(DynValue(IsIn(col("a", int64), array([3, 7], int64))))
    values.append(DynValue(NumericCast[Float64Type](col("a", int64))))
    values.append(DynValue(Year(col("ts", timestamp(second)))))

    var proj = Project(
        input=DynRelation(InMemoryTable(batch=batch)),
        names=["lk", "co", "isin", "cast", "yr"],
        values=values^,
        schema=schema(
            [
                field("lk", string),
                field("co", int64),
                field("isin", string),
                field("cast", float64),
                field("yr", int32),
            ]
        ),
    )
    print(DynRelation(proj^).execute())
