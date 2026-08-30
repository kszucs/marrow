from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, v, w FROM basic WHERE k IS NULL

    `IS NULL` over a *string* column. `nulls_is_null` asks it of an int64 one,
    and the two are different instantiations: `NullPredicate` reads its
    operand's validity, and a `StringColumn`'s bound is a `BinaryLikeArray`
    where a `Column[T]`'s is a `PrimitiveArray[T]`.

    -- expected
    k:string	v:int64	w:int64
    NULL	7	70
    """
    var t = table("basic")
    return t.filter(IsNull(col("k", string)))
