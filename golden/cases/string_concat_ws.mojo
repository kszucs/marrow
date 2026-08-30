from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT concat_ws('-', t, 'x') AS s FROM text

    A third null rule in the same family: `concat_ws` skips null *arguments*
    but returns null if the *separator* is null, and it does not emit a
    separator around the values it skipped.

    `marrow/kernels/string.mojo` has no such kernel.

    -- skip mojo

    -- expected
    s:string
    'a,b,c-x'
    'xyz-x'
    '-x'
    'x'
    '  Ab  -x'
    'héllo wörld-x'
    """
    var t = table("text")
    return t.project(
        ["s"], [concat_ws("-", [col("t", string), lit("x", string)])]
    )
