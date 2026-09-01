from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT left(t, 2) AS l, right(t, 2) AS r FROM text

    The two ends, counting characters. `héllo wörld` is the discriminating row
    again: `left(t, 2)` is `hé`, three bytes.

    `marrow/kernels/string.mojo` has no such kernel.

    -- skip python

    -- expected
    l:string	r:string
    'a,'	',c'
    'xy'	'yz'
    ''	''
    NULL	NULL
    '  '	'  '
    'hé'	'ld'
    """
    var t = table("text")
    return t.project(
        ["l", "r"], [col("t", string).left(2), col("t", string).right(2)]
    )
