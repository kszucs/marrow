"""Native Parquet I/O — mirrors the ``pyarrow.parquet`` API.

    import marrow.parquet as pq
    table = pq.read_table("data.parquet")
    pq.write_table(table, "out.parquet", compression="lz4")

Reads and writes Arrow only; no PyArrow at runtime.
"""

from . import Table
from . import libmarrow as _ma


def read_table(source):
    """Read a Parquet file into a marrow :class:`Table`.

    Parameters
    ----------
    source : str or path-like
        Path to the Parquet file.
    """
    return Table.wrap(_ma.parquet_read_table(str(source)))


def write_table(table, where, compression="snappy"):
    """Write a table to a Parquet file.

    Parameters
    ----------
    table : marrow.Table or Arrow C stream compatible object (e.g. a PyArrow
        table).
    where : str or path-like
        Output path.
    compression : {"snappy", "zstd", "lz4", "none"}, default "snappy"
        Page compression codec.
    """
    binding = table.unwrap() if isinstance(table, Table) else table
    _ma.parquet_write_table(binding, str(where), compression)
