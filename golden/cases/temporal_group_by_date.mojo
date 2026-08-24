from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT d, CAST(count(*) AS BIGINT) AS n FROM events GROUP BY d ORDER BY d NULLS FIRST

    A date32 group key, including the NULL group. The key column has to come
    back as date32 — the grouper materializes keys through `DynBuilder(dtype)`,
    so a key that lost its type would show up here as an int32 day count.

    **The two lanes disagree here, and this case is the record of it.**
    `Relation.aggregate` names a key after its source column when
    `BoxedValue.bound_column(schema)` finds one, and `key<i>` when it does not
    (`marrow/exprold/relations.mojo`). `NumericColumn`, `BoolColumn` and
    `StringColumn` each override `bound_column`; `TemporalColumn` and
    `ListColumn` do not, so they inherit the `Value` default that always
    answers -1. A date or timestamp group key therefore comes back as `key0`
    in the AOT lane, and the Mojo lane fails with `Column 'd' not found.` when
    the following sort looks for it. The runtime lane is unaffected —
    `DynValue.bound_column` resolves the name — which is why the Python lane
    answers `d` and passes.

    The expectation is DuckDB's, and DuckDB is right: `GROUP BY d` produces a
    column called `d`. Renaming the case's expectation to `key0` would make
    both lanes green by writing marrow's bug down as the specification.

    `-- skip python` is here only because the mark below is strict and applies
    to both lanes: the runtime lane already answers correctly, so without the
    skip it would report an xpass. The bug is the Mojo lane's alone. Groups
    on a temporal key is still covered in *both* lanes by
    `temporal_group_by_date_trunc_month`, whose key is computed and so is
    named `key0` by both.

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
    var q = agg.sort([col("d", date32())], [True])
    return q
