from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT d, CAST(count(*) AS BIGINT) AS n FROM events GROUP BY d ORDER BY d NULLS FIRST

    A date32 group key, including the NULL group. The key column has to come
    back as date32 — the grouper materializes keys through `DynBuilder(dtype)`,
    so a key that lost its type would show up here as an int32 day count.

    **This case is the record of a disagreement that `marrow/expr` settled.**
    `Aggregate._output_schema` names a key after its source column when the
    key is a bare column and `key<i>` by position when it is computed
    (`marrow/expr/logical.mojo`). The lane the key was written in no longer
    changes the answer. The predecessor package resolved the name through a
    `bound_column` method that `TemporalColumn` never overrode, so a date or
    timestamp key came back as `key0` in the AOT lane and as `d` in the
    runtime one — one query with two output schemas.

    The expectation is DuckDB's, and DuckDB is right: `GROUP BY d` produces a
    column called `d`. Both lanes now say so.

    -- expected
    d:date32	n:int64
    NULL	1
    '2020-02-29'	1
    '2021-01-01'	1
    '2021-06-15'	2
    '2021-12-31'	1
    """
    var t = table("events")
    var agg = t.aggregate(
        keys=[col("d", date32())],
        aggs=[count_star().alias("n")],
    )
    return agg.sort_by([col("d", date32())], [True])
