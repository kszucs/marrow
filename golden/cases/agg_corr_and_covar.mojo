from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT corr(qty, price) AS c, covar_pop(qty, price) AS cv FROM sales

    Two-operand aggregates. Like `arg_min`, these do not fit `Aggregate[Agg,
    A]`, which binds exactly one operand — and unlike `arg_min` they also need
    a five-component accumulator rather than Welford's three.

    Only rows where *both* columns are valid contribute, which is a third rule
    the one-operand aggregates never have to state.

    -- skip mojo

    -- expected
    c:double	cv:double
    0.8754607533314501	28.90625
    """
    var t = table("sales")
    return t.aggregate(
        aggs=[
            col("qty", int32).corr(col("price", float64)).alias("c"),
            col("qty", int32).covar(col("price", float64)).alias("cv"),
        ]
    )
