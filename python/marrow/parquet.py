"""Native Parquet I/O — mirrors the ``pyarrow.parquet`` API.

    import marrow.parquet as pq
    table = pq.read_table("data.parquet")
    pq.write_table(table, "out.parquet", compression="lz4")
    pq.write_table(table, "out.parquet", use_content_defined_chunking=True)

Reads and writes Arrow only; no PyArrow at runtime.
"""

from . import libmarrow as _ma
from ._wrapper import unwrap
from .tabular import Table


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

_CDC_DEFAULTS = {
    "min_chunk_size": 256 * 1024,
    "max_chunk_size": 1024 * 1024,
    "norm_level": 0,
}


def write_table(
    table,
    where,
    compression="snappy",
    data_page_version="1.0",
    use_content_defined_chunking=False,
):
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
    use_content_defined_chunking : bool or dict, default False
        Derive data page boundaries from a rolling hash over each column's
        ``(def_level, rep_level, value)`` stream instead of a row count, so
        unchanged pages stay byte-identical across an edit and a
        content-addressable store can deduplicate them. ``True`` enables it
        with ``min_chunk_size=262144`` and ``max_chunk_size=1048576``. A dict
        overrides the defaults and must supply both ``min_chunk_size`` and
        ``max_chunk_size`` (bytes); it may also supply ``norm_level``
        (default 0), which widens the chunk-size distribution around the
        average when positive and narrows it when negative -- see
        :func:`pyarrow.parquet.write_table`.

        marrow applies the 20 000-row page cap inside each chunk, matching
        Arrow C++'s default, but has no matching encoded-byte cap: Arrow
        C++'s comes from an encoder-internal running total, and a writer
        that picks page boundaries before encoding has no counterpart for
        it. With a large ``max_chunk_size`` and wide values, marrow will
        therefore write larger pages than Arrow C++ would for the same data.
    """
    try:
        version = _PAGE_VERSIONS[data_page_version]
    except KeyError:
        raise ValueError(f"unsupported data_page_version: {data_page_version!r}")

    if isinstance(use_content_defined_chunking, bool):
        cdc_enabled = use_content_defined_chunking
        cdc_opts = _CDC_DEFAULTS
    elif isinstance(use_content_defined_chunking, dict):
        allowed_keys = set(_CDC_DEFAULTS)
        mandatory_keys = {"min_chunk_size", "max_chunk_size"}
        given_keys = set(use_content_defined_chunking)
        unknown_keys = given_keys - allowed_keys
        if unknown_keys:
            raise ValueError(
                f"Unknown options in 'use_content_defined_chunking': {unknown_keys}"
            )
        missing_keys = mandatory_keys - given_keys
        if missing_keys:
            raise ValueError(
                f"Missing options in 'use_content_defined_chunking': {missing_keys}"
            )
        cdc_enabled = True
        cdc_opts = {**_CDC_DEFAULTS, **use_content_defined_chunking}
    else:
        raise TypeError(
            "'use_content_defined_chunking' should be either boolean or a dictionary"
        )
    cdc_min_chunk_size = cdc_opts["min_chunk_size"]
    cdc_max_chunk_size = cdc_opts["max_chunk_size"]
    cdc_norm_level = cdc_opts["norm_level"]

    binding = table.unwrap() if isinstance(table, Table) else table
    _ma.parquet_write_table(
        binding,
        str(where),
        compression,
        version,
        cdc_enabled,
        cdc_min_chunk_size,
        cdc_max_chunk_size,
        cdc_norm_level,
    )
