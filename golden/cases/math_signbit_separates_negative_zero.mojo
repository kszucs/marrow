from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT signbit(x) AS b FROM floats

    The only function that distinguishes `-0.0` from `0.0`: they compare equal,
    hash equal and print differently, so `signbit` is the sole observable
    difference — and the reason the `floats` fixture carries both.

    `sign` answers 0 for each of them, which is why it is not a substitute.

    `marrow/kernels/numeric.mojo` has no such kernel.

    -- skip mojo

    -- expected
    b:bool
    False
    True
    False
    True
    False
    False
    True
    NULL
    """
    var t = table("floats")
    return t.project(["b"], [col("x", float64).signbit()])
