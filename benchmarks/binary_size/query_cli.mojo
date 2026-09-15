"""The compile guide's example, and the `query_cli` binary-size gate."""

from marrow.dtypes import field, int64, string
from marrow.expr import DynRelation, QueryCli, col, param, scan
from marrow.schema import schema


def query() raises -> DynRelation:
    var orders = scan(
        param("src", string),
        schema(
            [field("id", int64), field("amount", int64), field("name", string)]
        ),
    )
    return orders.filter(col("amount", int64) >= param("min-amount", int64))


def main() raises:
    QueryCli(query()).run()
