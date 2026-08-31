from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT sin(y) AS s, cos(y) AS c, atan2(y, n) AS a FROM floats WHERE n IS NOT NULL

    The trigonometric family, including the two-argument `atan2` — which is the
    one that needs both operands' signs and therefore cannot be composed from a
    unary `atan`. `y` is finite everywhere, and the predicate drops the row
    where `n` is null so the second operand is defined.

    `marrow/kernels/numeric.mojo` has no such kernel.

    -- skip mojo
    -- skip python

    -- expected
    s:double	c:double	a:double
    0.9092974268256817	-0.4161468365471424	0.4636476090008061
    -0.7568024953079283	-0.6536436208636119	2.723368324010564
    0.9893582466233818	-0.14550003380861354	1.5707963267948966
    0.8414709848078965	0.5403023058681398	0.7853981633974483
    0.8414709848078965	0.5403023058681398	0.4636476090008061
    0.9092974268256817	-0.4161468365471424	0.5880026035475675
    0.9092974268256817	-0.4161468365471424	2.0344439357957027
    """
    var t = table("floats")
    var defined = t.filter(NotNull(col("n", int64)))
    return defined.project(
        ["s", "c", "a"],
        [
            col("y", float64).sin(),
            col("y", float64).cos(),
            atan2(col("y", float64), col("n", int64)),
        ],
    )
