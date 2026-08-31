from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT regexp_matches(t, '^[a-z],') AS b FROM text

    A regular-expression predicate. DuckDB's `regexp_matches` is a **partial**
    match and `regexp_full_match` an anchored one — the distinction most
    engines do not draw, and the reason the pattern here anchors explicitly.

    `marrow/kernels/string.mojo` has no such kernel. `LikeKernel` implements
    SQL `LIKE`, which is a different language.

    -- skip mojo
    -- skip python

    -- expected
    b:bool
    True
    False
    False
    NULL
    False
    False
    """
    var t = table("text")
    return t.project(["b"], [col("t", string).regexp_matches("^[a-z],")])
