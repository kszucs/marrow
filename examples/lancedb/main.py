"""Integration with LanceDB.

Demonstrates reading two LanceDB tables as Arrow data, adding their numeric
columns element-wise with marrow, and writing the result back to LanceDB.
"""

import tempfile

import lancedb
import pyarrow as pa

import marrow as ma


def add_tables(table_a: lancedb.table.Table, table_b: lancedb.table.Table) -> lancedb.table.Table:
    """Add two LanceDB tables element-wise using marrow and return a new table.

    Both tables must share the same schema and row count.  Each numeric column
    is added element-wise via marrow's ``add`` kernel.  The result is written
    to a new table named ``"result"`` inside ``table_a``'s database.

    Args:
        table_a: First LanceDB table.
        table_b: Second LanceDB table with the same schema and row count.

    Returns:
        A new LanceDB table whose numeric columns are the element-wise sum of
        the two input tables.
    """
    arrow_a = table_a.to_arrow()
    arrow_b = table_b.to_arrow()

    result_columns = {}
    for name in arrow_a.schema.names:
        col_a = arrow_a.column(name)
        col_b = arrow_b.column(name)
        if pa.types.is_integer(col_a.type) or pa.types.is_floating(col_a.type):
            # Flatten ChunkedArrays to a single chunk so marrow sees one array.
            flat_a = col_a.combine_chunks() if isinstance(col_a, pa.ChunkedArray) else col_a
            flat_b = col_b.combine_chunks() if isinstance(col_b, pa.ChunkedArray) else col_b
            summed = ma.add(flat_a, flat_b)
            result_columns[name] = summed
        else:
            # Non-numeric columns: carry over from table_a unchanged.
            result_columns[name] = col_a

    result_arrow = pa.table(result_columns, schema=arrow_a.schema)
    db = table_a._conn
    return db.create_table("result", result_arrow, mode="overwrite")


def main():
    """Create two LanceDB tables, add them with marrow, and print the result."""
    with tempfile.TemporaryDirectory() as tmp:
        db = lancedb.connect(tmp)

        schema = pa.schema([
            pa.field("x", pa.float32()),
            pa.field("y", pa.int64()),
            pa.field("label", pa.utf8()),
        ])
        data_a = pa.table({
            "x": pa.array([1.0, 2.0, 3.0], type=pa.float32()),
            "y": pa.array([10, 20, 30], type=pa.int64()),
            "label": pa.array(["a", "b", "c"]),
        }, schema=schema)
        data_b = pa.table({
            "x": pa.array([0.5, 1.5, 2.5], type=pa.float32()),
            "y": pa.array([1, 2, 3], type=pa.int64()),
            "label": pa.array(["a", "b", "c"]),
        }, schema=schema)

        tbl_a = db.create_table("table_a", data_a)
        tbl_b = db.create_table("table_b", data_b)

        result = add_tables(tbl_a, tbl_b)
        print(result.to_arrow())


if __name__ == "__main__":
    main()
