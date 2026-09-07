"""marrow — Apache Arrow in Mojo, with a Python face.

Two surfaces, told apart by the entry point:

* **Eager**, PyArrow-shaped — `array`, `record_batch`, `table`, `compute`.
* **Lazy**, relational — `memtable`, `read_parquet`, and the verbs on
  :class:`~marrow.lazy.LazyTable`. Nothing runs until `collect()`.

This module is re-exports. The implementations live next door: `types`,
`arrays`, `tabular`, `expr`, `lazy`, `compute`, `parquet`, `ipc`.
"""

from . import libmarrow
from .types import (
    DataType,
    Field,
    Schema,
    binary,
    bool_,
    date32,
    date64,
    day_time_interval,
    duration,
    field,
    fixed_size_binary,
    fixed_size_list_,
    float16,
    float32,
    float64,
    infer_type,
    int8,
    int16,
    int32,
    int64,
    list_,
    month_day_nano_interval,
    null,
    schema,
    string,
    struct,
    time32,
    time64,
    timestamp,
    uint8,
    uint16,
    uint32,
    uint64,
    year_month_interval,
)
from .arrays import Array, ChunkedArray, Scalar, array, chunked_array
from .tabular import (
    RecordBatch,
    Table,
    concat_arrays,
    concat_tables,
    record_batch,
    table,
)
from .ipc import (
    read_ipc_file,
    read_ipc_file_schema,
    read_ipc_stream,
    read_ipc_stream_schema,
    write_ipc_file,
    write_ipc_stream,
)
from .expr import (
    Aggregate,
    Column,
    Window,
    array_contains,
    array_length,
    case_when,
    coalesce,
    col,
    count_star,
    if_else,
    cume_dist,
    dense_rank,
    is_in,
    lit,
    maximum,
    minimum,
    ntile,
    percent_rank,
    rank,
    row_number,
)
from .lazy import LazyTable, memtable, read_parquet, sql
from . import compute, expr, lazy

ExecContext = libmarrow.ExecContext


# ── Join kinds ─────────────────────────────────────────────────────────────
#
# `marrow.kernels.join` names these as constants and a Mojo golden case writes
# them by name; the binding takes the string. Both spellings resolve to the
# same value so one case text runs in either lane.

JOIN_INNER = "inner"
JOIN_LEFT = "left"
JOIN_RIGHT = "right"
JOIN_FULL = "full"
JOIN_SEMI = "semi"
JOIN_ANTI = "anti"
JOIN_ALL = "all"

__all__ = [
    "Aggregate",
    "Array",
    "ChunkedArray",
    "Column",
    "DataType",
    "ExecContext",
    "Field",
    "JOIN_ALL",
    "JOIN_ANTI",
    "JOIN_FULL",
    "JOIN_INNER",
    "JOIN_LEFT",
    "JOIN_RIGHT",
    "JOIN_SEMI",
    "LazyTable",
    "RecordBatch",
    "Scalar",
    "Schema",
    "Table",
    "Window",
    "array",
    "array_contains",
    "array_length",
    "binary",
    "bool_",
    "case_when",
    "chunked_array",
    "coalesce",
    "col",
    "compute",
    "concat_arrays",
    "concat_tables",
    "count_star",
    "cume_dist",
    "date32",
    "date64",
    "day_time_interval",
    "dense_rank",
    "duration",
    "expr",
    "field",
    "fixed_size_binary",
    "fixed_size_list_",
    "float16",
    "float32",
    "float64",
    "if_else",
    "infer_type",
    "int8",
    "int16",
    "int32",
    "int64",
    "is_in",
    "lazy",
    "list_",
    "lit",
    "maximum",
    "memtable",
    "minimum",
    "month_day_nano_interval",
    "ntile",
    "null",
    "percent_rank",
    "read_ipc_file",
    "read_ipc_file_schema",
    "read_ipc_stream",
    "read_ipc_stream_schema",
    "rank",
    "read_parquet",
    "record_batch",
    "row_number",
    "schema",
    "sql",
    "string",
    "struct",
    "table",
    "time32",
    "time64",
    "timestamp",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "write_ipc_file",
    "write_ipc_stream",
    "year_month_interval",
]
