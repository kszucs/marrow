from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT list_contains(l, 5) AS b FROM lists

    Membership within a list. Both the kernel
    (`marrow/kernels/nested.mojo`) and the expression node
    (`marrow/expr/comptime/nested.mojo::ArrayContains`) now exist; what is
    missing is the *method* this case calls. `.contains` lives only on
    `StringValue`, so a `ListValue` needs its own — the free verb
    `array_contains(list, elem)` is the only spelling available today.

    The null list answers null and the empty list answers false, which are
    different answers to "is 5 in there".

    -- skip mojo
    -- skip python

    -- expected
    b:bool
    False
    False
    NULL
    True
    False
    """
    var t = table("lists")
    return t.project(["b"], [col("l", list_(int64)).contains(lit(5, int64))])
