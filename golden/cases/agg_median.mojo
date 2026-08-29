from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT median(b) AS m FROM nulls

    The median is *holistic*: unlike `sum` or `min` it cannot be folded from a
    constant-size accumulator, so it needs either a sort or a selection
    algorithm over the whole group. `b` has an even number of values, so the
    answer is the midpoint of the two middle ones and not a value from the
    column.

    `marrow/kernels/aggregate.mojo` has no order-statistic kernel.

    -- skip mojo

    -- expected
    m:double
    5.0
    """
    var t = table("nulls")
    return t.aggregate(aggs=[col("b", int64).median().alias("m")])
