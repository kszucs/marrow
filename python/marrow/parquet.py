"""Native Parquet I/O — mirrors the ``pyarrow.parquet`` API.

    import marrow.parquet as pq
    table = pq.read_table("data.parquet")
    pq.write_table(table, "out.parquet", compression="lz4")

Reads and writes Arrow only; no PyArrow at runtime.
"""

from . import Table
from . import libmarrow as _ma


def read_table(source, columns=None):
    """Read a Parquet file into a marrow :class:`Table`.

    Parameters
    ----------
    source : str or path-like
        Path to the Parquet file.
    columns : list of str, optional
        Only read these top-level columns, in the given order. Reads all
        columns when ``None``.
    """
    cols = list(columns) if columns is not None else None
    return Table.wrap(_ma.parquet_read_table(str(source), cols))


_PAGE_VERSIONS = {"1.0": 1, "2.0": 2, 1: 1, 2: 2}


def write_table(table, where, compression="snappy", data_page_version="1.0"):
    """Write a table to a Parquet file.

    Parameters
    ----------
    table : marrow.Table or Arrow C stream compatible object (e.g. a PyArrow
        table).
    where : str or path-like
        Output path.
    compression : {"snappy", "zstd", "lz4", "none"}, default "snappy"
        Page compression codec.
    data_page_version : {"1.0", "2.0"}, default "1.0"
        Parquet data page format version.
    """
    try:
        version = _PAGE_VERSIONS[data_page_version]
    except KeyError:
        raise ValueError(f"unsupported data_page_version: {data_page_version!r}")
    binding = table.unwrap() if isinstance(table, Table) else table
    _ma.parquet_write_table(binding, str(where), compression, version)
