from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(cardinality(m) AS BIGINT) AS n FROM nested ORDER BY id NULLS FIRST

    How many entries a map has — the map counterpart of `array_length`, and the
    one list verb marrow does have, unavailable here only because `ListColumn`
    cannot be spelled over a map through `col`.

    Empty is 0 and null is null, the same distinction `nested_array_length`
    asserts for lists.

    -- skip mojo
    -- skip python

    -- expected
    n:int64
    2
    0
    NULL
    """
    var t = table("nested")
    var sized = t.project(
        ["n"], [array_length(col("m", map_of(string, int64)))]
    )
    return sized.sort_by([col("id", int64)], [True])
