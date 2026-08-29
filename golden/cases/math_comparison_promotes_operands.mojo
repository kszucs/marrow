from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT i, f, s, b FROM nums WHERE f > i

    A comparison between two different numeric types. `1.7 > 1` is the row that
    discriminates: an engine that narrowed the double to the integer's type
    before comparing would answer false and return nothing.

    -- expected
    i:int64	f:double	s:string	b:bool
    1	1.7	'1'	True
    """
    var t = table("nums")
    return t.filter(col("f", float64) > col("i", int64))
