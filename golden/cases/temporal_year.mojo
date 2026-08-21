from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(year(ts) AS INTEGER) AS y FROM events

    The field extractions return **int32**, not the int64 DuckDB gives, so
    every twin in this family casts. That is marrow answering the same question
    in a narrower type, not a different answer: `TemporalExtract.OutType` is
    `Int32Type` for all nine kernels (`marrow/expr/values.mojo`), which is what
    lets an extracted field feed the fused numeric lane.

    -- expected
    y:int32
    2021
    2021
    2021
    NULL
    2020
    2021
    """
    var t = table("events")
    var q = t.project(["y"], [col("ts", timestamp(microsecond)).year()])
    return q
