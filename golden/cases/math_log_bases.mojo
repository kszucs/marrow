from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT log10(y) AS a, log2(y) AS b FROM floats

    `ln` is the only logarithm marrow has. These two are not `ln(x)/ln(base)`
    in practice — a correctly-rounded `log2` of a power of two is exact where
    the quotient form is not, and `y` holds 1, 2, 4 and 8 to say so.

    `marrow/kernels/numeric.mojo` has no such kernel.

    -- skip mojo

    -- expected
    a:double	b:double
    0.3010299956639812	1.0
    0.6020599913279624	2.0
    0.9030899869919435	3.0
    0.0	0.0
    0.0	0.0
    0.3010299956639812	1.0
    0.3010299956639812	1.0
    NULL	NULL
    """
    var t = table("floats")
    return t.project(
        ["a", "b"], [col("y", float64).log10(), col("y", float64).log2()]
    )
