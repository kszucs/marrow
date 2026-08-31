from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, v, w FROM basic WHERE v NOT IN (1, NULL)

    SQL's most notorious result: `NOT IN` with a NULL in the list matches
    **nothing**, because `v <> NULL` is unknown for every row and `AND` over an
    unknown can never be true. Plain `IN` with a NULL is harmless — it simply
    cannot add rows — so the trap is one-sided, and an `is_in` kernel that
    treats the set as a hash lookup will get this wrong in the natural way.

    -- skip mojo
    -- skip python

    -- expected
    k:string	v:int64	w:int64
    """
    var t = table("basic")
    return t.filter(~col("v", int64).is_in([1, None]))
