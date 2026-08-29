from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT t || '!' AS s FROM text

    Half of SQL's most-tripped-over pair: `||` is null-propagating, so a null
    operand makes the whole result null. `string_concat_function_skips_null` is
    the other half, and the two must not be implemented as one function.

    `marrow/kernels/string.mojo` has no such kernel.

    -- skip mojo

    -- expected
    s:string
    'a,b,c!'
    'xyz!'
    '!'
    NULL
    '  Ab  !'
    'héllo wörld!'
    """
    var t = table("text")
    return t.project(["s"], [col("t", string) + lit("!", string)])
