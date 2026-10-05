# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Native Avro object container file I/O.

    import marrow.avro as mavro
    table = mavro.read_table("data.avro")
    mavro.write_table(table, "out.avro", codec="snappy")

Reads with the writer's schema; there is no reader schema to resolve against.
A record field's Avro ``field-id`` (and an array's ``element-id``, a map's
``key-id`` / ``value-id``) is the ``field_id`` metadata of the Arrow field it
becomes, and header metadata outside ``avro.*`` is the table schema's metadata.
"""

from . import libmarrow as _ma
from .tabular import Table


def read_table(source, columns=None):
    """Read an Avro object container file into a marrow :class:`Table`.

    Parameters
    ----------
    source : str or path-like
        Path or URI of the file.
    columns : list of str, optional
        Only read these top-level fields, in the given order. Reads all of them
        when ``None``.
    """
    cols = list(columns) if columns is not None else None
    return Table.wrap(_ma.avro_read_table(str(source), cols))


def write_table(table, where, codec="deflate"):
    """Write a table to an Avro object container file.

    Parameters
    ----------
    table : marrow.Table or Arrow C stream compatible object (e.g. a PyArrow
        table).
    where : str or path-like
        Output path or URI.
    codec : {"deflate", "snappy", "zstandard", "null"}, default "deflate"
        Block compression codec.
    """
    binding = table.unwrap() if isinstance(table, Table) else table
    _ma.avro_write_table(binding, str(where), codec)
