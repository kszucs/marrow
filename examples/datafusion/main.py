"""DataFusion + Marrow example.

Demonstrates registering Mojo compute functions as DataFusion UDFs and
running SQL queries over Arrow arrays.

Run with:
    pixi run run
"""

import pyarrow as pa
import marrow as ma
from datafusion import SessionContext, udf


def make_session() -> SessionContext:
    ctx = SessionContext()

    def mojo_add(a: pa.Array, b: pa.Array) -> pa.Array:
        return pa.array(ma.add(a, b))

    ctx.register_udf(
        udf(mojo_add, [pa.int64(), pa.int64()], pa.int64(), "immutable", name="mojo_add")
    )

    return ctx


def main() -> None:
    ctx = make_session()

    batch = pa.record_batch(
        {
            "price": pa.array([100, 200, 300, 400, 500], type=pa.int64()),
            "quantity": pa.array([3, 1, 4, 1, 5], type=pa.int64()),
        }
    )
    ctx.register_record_batches("orders", [[batch]])

    result = pa.Table.from_batches(
        ctx.sql(
            "SELECT price, quantity, mojo_add(price, quantity) AS total FROM orders"
        ).collect()
    )
    for i in range(result.num_rows):
        price = result.column("price")[i].as_py()
        quantity = result.column("quantity")[i].as_py()
        total = result.column("total")[i].as_py()
        print(f"  price={price}, quantity={quantity}, total={total}")


if __name__ == "__main__":
    main()
