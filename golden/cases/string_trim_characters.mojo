from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT trim(t, ' Ab') AS s FROM text

    `trim` with an explicit character **set** — every leading and trailing
    character that is a member, not the literal substring. marrow's
    `StripKernel` family trims whitespace only, so the argument-taking form is
    the gap.

    `marrow/kernels/string.mojo` has no such kernel.

    -- skip mojo

    -- expected
    s:string
    'a,b,c'
    'xyz'
    ''
    NULL
    ''
    'héllo wörld'
    """
    var t = table("text")
    return t.project(["s"], [col("t", string).strip(" Ab")])
