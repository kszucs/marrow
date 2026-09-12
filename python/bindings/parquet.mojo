"""Python bindings for the native Parquet reader/writer.

Thin marshaling over `marrow.parquet.read_table` / `write_table`: the friendly
API (default compression, the `marrow.parquet` module surface) lives in pure
Python; these entry points stay strict.
"""

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from marrow.parquet import (
    read_table as _read_table,
    write_table as _write_table,
)
from marrow.parquet.chunker import ContentDefinedChunking
from marrow.parquet.codecs import Compression
from marrow.tabular import Table


def _codec(name: String) raises -> Compression:
    if name == "none" or name == "uncompressed":
        return Compression.UNCOMPRESSED
    elif name == "snappy":
        return Compression.SNAPPY
    elif name == "zstd":
        return Compression.ZSTD
    elif name == "lz4":
        return Compression.LZ4_RAW
    else:
        raise Error("parquet: unsupported compression '" + name + "'")


def parquet_read_table(
    path: PythonObject, columns: PythonObject
) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    if columns.__is__(builtins.None):
        return _read_table(String(py=path)).to_python_object()
    var cols = List[String]()
    for i in range(Int(py=columns.__len__())):
        cols.append(String(py=columns[i]))
    return _read_table(String(py=path), columns=cols^).to_python_object()


def parquet_write_table(
    table: PythonObject,
    path: PythonObject,
    compression: PythonObject,
    version: PythonObject,
    cdc: PythonObject,
    cdc_min: PythonObject,
    cdc_max: PythonObject,
    cdc_norm: PythonObject,
) raises -> PythonObject:
    var t = Table(py=table)
    var chunking: Optional[ContentDefinedChunking] = None
    if Bool(py=cdc):
        chunking = ContentDefinedChunking(
            Int(py=cdc_min), Int(py=cdc_max), Int(py=cdc_norm)
        )
    _write_table(
        t,
        String(py=path),
        _codec(String(py=compression)),
        Int(py=version),
        content_defined_chunking=chunking^,
    )
    return Python.evaluate("None")


def add_to_module(mut mb: PythonModuleBuilder) raises -> None:
    """Add Parquet reader/writer functions to the Python module."""
    mb.def_function[parquet_read_table]("parquet_read_table")
    mb.def_function[parquet_write_table]("parquet_write_table")
