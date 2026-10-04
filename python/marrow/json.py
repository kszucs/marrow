# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Newline-delimited JSON — mirrors the ``pyarrow.json`` API.

    import marrow.json as mj
    table = mj.read_json("rows.jsonl")
    table = mj.read_json(
        "rows.jsonl",
        parse_options=mj.ParseOptions(explicit_schema=schema),
    )
    for batch in mj.open_json("rows.jsonl"):
        ...
    mj.write_json(table, "out.jsonl")

Types are inferred as Arrow C++ infers them — ``int64`` widening to
``double``, ISO-8601 strings to ``timestamp[s]``, objects to structs, arrays to
lists — and the errors carry Arrow's wording. Where marrow differs from
pyarrow: the reader is single-threaded (``use_threads`` is accepted and
ignored), a row longer than ``block_size`` grows the read instead of failing,
``NaN`` is rejected as the non-JSON it is, and a file of empty objects reads
as zero rows.

For a lazy scan that composes with the query verbs, see
:func:`marrow.read_json`.
"""

from . import libmarrow as _ma
from ._wrapper import _Wrapper, unwrap
from .tabular import RecordBatch, Table
from .types import Schema

__all__ = ["ReadOptions", "ParseOptions", "read_json", "open_json", "write_json"]

_BEHAVIORS = ("ignore", "error", "infer")


class ReadOptions:
    """How the input is read.

    Parameters
    ----------
    use_threads : bool, optional
        Accepted for pyarrow compatibility; marrow reads on one thread.
    block_size : int, optional
        Bytes per block, default 1 MiB. Each block becomes one chunk of the
        result.
    """

    def __init__(self, use_threads=None, block_size=None):
        self.use_threads = use_threads
        self.block_size = 1 << 20 if block_size is None else int(block_size)
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")


class ParseOptions:
    """How values become columns.

    Parameters
    ----------
    explicit_schema : Schema, optional
        Columns and types to read — a marrow or any Arrow-compatible schema
        (e.g. ``pyarrow.schema``). Inferred when ``None``.
    newlines_in_values : bool, optional
        Not supported: a block is always cut at a newline.
    unexpected_field_behavior : {"infer", "ignore", "error"}, default "infer"
        What a key outside ``explicit_schema`` does.
    """

    def __init__(
        self,
        explicit_schema=None,
        newlines_in_values=None,
        unexpected_field_behavior="infer",
    ):
        if newlines_in_values:
            raise NotImplementedError("newlines_in_values=True is not supported")
        if unexpected_field_behavior not in _BEHAVIORS:
            raise ValueError(
                "unexpected_field_behavior must be one of "
                f"{_BEHAVIORS}, got {unexpected_field_behavior!r}"
            )
        self.explicit_schema = explicit_schema
        self.newlines_in_values = newlines_in_values
        self.unexpected_field_behavior = unexpected_field_behavior


def _args(input_file, read_options, parse_options):
    read_options = read_options or ReadOptions()
    parse_options = parse_options or ParseOptions()
    return (
        str(input_file),
        unwrap(parse_options.explicit_schema),
        parse_options.unexpected_field_behavior,
        read_options.block_size,
    )


def read_json(input_file, read_options=None, parse_options=None):
    """Read a newline-delimited JSON file into a :class:`marrow.Table`.

    ``input_file`` is a path or a URL (``s3://…``, ``gs://…``), opened as
    :func:`marrow.parquet.read_table` opens one. The schema is inferred over
    the whole file, one chunk per block.
    """
    return Table.wrap(
        _ma.json_read_table(*_args(input_file, read_options, parse_options))
    )


class JSONStreamingReader(_Wrapper):
    """Batches of a newline-delimited JSON file, one per block.

    The schema is inferred from the first block and fixed after it, as in
    pyarrow: a later key the schema lacks, or a value its type cannot hold,
    raises.
    """

    __slots__ = ()

    @property
    def schema(self):
        return Schema.wrap(self._binding.schema())

    def read_next_batch(self):
        """The next batch; raises ``StopIteration`` at the end."""
        batch = self._binding.read_next_batch()
        if batch is None:
            raise StopIteration
        return RecordBatch.wrap(batch)

    __next__ = read_next_batch

    def __iter__(self):
        return self

    def read_all(self):
        """Every remaining batch, as a :class:`marrow.Table`."""
        return Table.from_batches(list(self), schema=self.schema)


def open_json(input_file, read_options=None, parse_options=None):
    """A streaming reader over a newline-delimited JSON file."""
    return JSONStreamingReader.wrap(
        _ma.json_open(*_args(input_file, read_options, parse_options))
    )


def write_json(table, where):
    """Write a table as newline-delimited JSON: one object per row, keys in
    schema order, nulls written as ``null``.

    ``table`` is a :class:`marrow.Table` or any Arrow C stream compatible
    object (e.g. a PyArrow table); ``where`` a path or a URL, opened as
    :func:`marrow.parquet.write_table` opens one. Timestamps and dates are
    ISO-8601 strings, and a NaN or an infinity is written as ``null``.
    """
    _ma.json_write_table(unwrap(table), str(where))
