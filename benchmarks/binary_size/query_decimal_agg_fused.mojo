# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Binary-size gate for a fused aggregation over **decimal** columns.

`SELECT name, sum(a), min(b) FROM orders GROUP BY name` with `a` and `b` as
`decimal128(10, 2)` — `query_streaming_agg_fused.mojo` with the int64 columns
swapped for decimals and nothing else changed, so the `__text` delta between
the two is what an AOT binary pays to aggregate decimals: the
`DecimalFold` sum, a decimal `min`, and printing decimal values.

The other gates never name a decimal, so they only show that decimal code is
eliminated where it is unused; this one measures it where it is used.

    pixi run bench-size
"""

from marrow.arrays import Decimal128Array
from marrow.builders import Decimal128Builder, array
from marrow.dtypes import Decimal128Type, decimal128, string
from marrow.expr import col, table
from marrow.expr import DynValue
from marrow.tabular import record_batch


def _decimals(values: List[Int]) raises -> Decimal128Array:
    var b = Decimal128Builder(decimal128(10, 2), len(values))
    for v in values:
        b.append(Scalar[Decimal128Type.native](v))
    return b.finish()


def main() raises:
    var a = _decimals([150, 525, 300, 899, 201])
    var b = _decimals([400, 400, 400, 400, 400])
    var nm = array(["p", "q", "p", "q", "p"])
    var batch = record_batch(
        [a^.to_dyn(), b^.to_dyn(), nm^.to_dyn()], names=["a", "b", "name"]
    )

    var dec = decimal128(10, 2)
    var keys: List[DynValue] = [col("name", string)]
    var aggs: List[DynValue] = [
        col("a", dec).sum().alias("a"),
        col("b", dec).min().alias("b"),
    ]
    print(table(batch^).aggregate(aggs^, keys^).execute())
