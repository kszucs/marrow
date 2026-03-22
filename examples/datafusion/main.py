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
        return pa.array(ma.add(ma.array(a), ma.array(b)))

    ctx.register_udf(
        udf(ma.add, [pa.int64(), pa.int64()], pa.int64(), "immutable", name="mojo_add")
    )

    def mojo_add_gpu(a: pa.Array, b: pa.Array) -> pa.Array:
        return pa.array(ma.add_gpu(ma.array(a), ma.array(b)))

    ctx.register_udf(
        udf(
            mojo_add_gpu,
            [pa.int64(), pa.int64()],
            pa.int64(),
            "immutable",
            name="mojo_add_gpu",
        )
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

    print("=== CPU: mojo_add ===")
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

    print("\n=== GPU: mojo_add_gpu ===")
    try:
        result = pa.Table.from_batches(
            ctx.sql(
                "SELECT price, quantity, mojo_add_gpu(price, quantity) AS total FROM orders"
            ).collect()
        )
        for i in range(result.num_rows):
            price = result.column("price")[i].as_py()
            quantity = result.column("quantity")[i].as_py()
            total = result.column("total")[i].as_py()
            print(f"  price={price}, quantity={quantity}, total={total}")
    except Exception as e:
        print(f"  GPU kernel not available: {e}")


if __name__ == "__main__":
    main()
