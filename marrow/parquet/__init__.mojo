"""Native Parquet I/O for Marrow.

A from-scratch Parquet reader/writer that speaks Arrow only — no PyArrow at
runtime. The Thrift Compact Protocol metadata codec, page/encoding decoders, and
compression FFI all live in this package; PyArrow is used solely as the test
oracle.

    from marrow.parquet import read_table, write_table
"""

from .reader import (
    LeafSet,
    leaf_of,
    LEAF_INT8,
    LEAF_INT16,
    LEAF_INT32,
    LEAF_INT64,
    LEAF_UINT8,
    LEAF_UINT16,
    LEAF_UINT32,
    LEAF_UINT64,
    LEAF_FLOAT16,
    LEAF_FLOAT32,
    LEAF_FLOAT64,
    LEAF_BOOL,
    LEAF_STRING,
    LEAF_LARGE_STRING,
    LEAF_BINARY,
    LEAF_LARGE_BINARY,
    LEAF_TEMPORAL32,
    LEAF_TEMPORAL64,
    LEAF_DECIMAL32,
    LEAF_DECIMAL64,
    LEAF_DECIMAL128,
    LEAF_DECIMAL256,
    LEAF_FIXED_SIZE_BINARY,
    LEAF_INT96,
    ParquetFile,
    read_table,
    read_metadata,
    ColumnStatistics,
    read_page_index,
    PageIndex,
    PageBounds,
    RowSelection,
)
from .writer import write_table
from .codecs import Compression
from ..utils import XxHash64
from .bloom import SplitBlockBloomFilter
