from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT substr(t, 2, 3) AS s FROM text

    SQL's `substr` is **1-based** and counts characters, where a dataframe
    `slice` is 0-based and several engines count bytes. `héllo wörld` is the
    row that separates the two: characters 2 to 4 are `éll`, bytes 2 to 4 are
    half of `é` followed by `ll`.

    The empty string answers with the empty string rather than an error.

    `marrow/kernels/string.mojo` has no such kernel.

    -- skip mojo

    -- expected
    s:string
    ',b,'
    'yz'
    ''
    NULL
    ' Ab'
    'éll'
    """
    var t = table("text")
    return t.project(["s"], [col("t", string).substr(2, 3)])
