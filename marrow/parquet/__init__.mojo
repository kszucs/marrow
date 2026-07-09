"""Native Parquet I/O for Marrow.

A from-scratch Parquet reader/writer that speaks Arrow only — no PyArrow at
runtime. The Thrift Compact Protocol metadata codec, page/encoding decoders, and
compression FFI all live in this package; PyArrow is used solely as the test
oracle.

    from marrow.parquet import read_table, write_table
"""

from .reader import read_table
from .writer import write_table
from .compression import Compression
