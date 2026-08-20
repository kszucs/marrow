from golden.helpers import table
from marrow.dtypes import int64
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation


def plan() raises -> DynRelation:
    """
    SELECT CAST(count(v) AS BIGINT) AS n FROM basic

    The contrast with `count(*)`: `v` has one null, so this is 6 where the
    row count is 7.

    -- expected
    n:int64
    6
    """
    var t = table("basic")
    var q = t.aggregate(
        aggs=[col("v", int64).count().alias("n")],
    )
    return q
