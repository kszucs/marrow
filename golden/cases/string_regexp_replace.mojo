from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT regexp_replace(t, '[aeiou]', '_') AS s FROM text

    `regexp_replace` replaces the **first** match by default, where the literal
    `replace` replaces all of them — the inverse of what most people assume,
    and the reason this is its own case rather than a variation of
    `string_replace`.

    `marrow/kernels/string.mojo` has no such kernel.

    -- skip mojo

    -- expected
    s:string
    '_,b,c'
    'xyz'
    ''
    NULL
    '  Ab  '
    'héll_ wörld'
    """
    var t = table("text")
    return t.project(["s"], [col("t", string).regexp_replace("[aeiou]", "_")])
