from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT map_extract(m, 'k')[1] AS v, map_extract(m, 'zz')[1] AS w FROM nested ORDER BY id NULLS FIRST

    Looking a key up in a map. DuckDB's `map_extract` answers with a **list** —
    because a map may hold duplicate keys — so the twin takes its first
    element; an engine that returns the value directly has silently decided
    duplicates cannot happen.

    A missing key and a null map both answer null, and the empty map is a third
    input with the same answer for a different reason.

    -- skip mojo

    -- expected
    v:int64	w:int64
    1	NULL
    NULL	NULL
    NULL	NULL
    """
    var t = table("nested")
    var looked = t.project(
        ["v", "w"],
        [
            col("m", map_of(string, int64)).get(lit("k", string)),
            col("m", map_of(string, int64)).get(lit("zz", string)),
        ],
    )
    return looked.sort_by([col("id", int64)], [True])
