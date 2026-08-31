from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT concat(t, '!') AS s FROM text

    The other half: DuckDB's `concat` **ignores** null arguments, so the null
    row answers `!` where `||` answers null. Same inputs, same intent,
    different answer — which is why the corpus asks both.

    `marrow/kernels/string.mojo` has no such kernel.

    -- skip mojo
    -- skip python

    -- expected
    s:string
    'a,b,c!'
    'xyz!'
    '!'
    '!'
    '  Ab  !'
    'héllo wörld!'
    """
    var t = table("text")
    return t.project(["s"], [concat([col("t", string), lit("!", string)])])
