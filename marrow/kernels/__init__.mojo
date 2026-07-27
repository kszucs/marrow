"""Compute kernels for marrow.

Re-exports kernel structs from the submodules so callers can
``import marrow.kernels as mk`` and use e.g. ``mk.AddKernel.dispatch``,
``mk.SumKernel.dispatch``, ``mk.filter``, ``mk.sort`` directly.

Submodules:
  - `arithmetic.mojo` — binary arithmetic, unary math, GPU dispatch via ``elementwise``
  - `compare.mojo` — comparison kernels producing bit-packed bool output
  - `aggregate.mojo` — reductions using ``std.algorithm`` (sum, min, max, etc.)
  - `filter.mojo` — selection/filter kernels
  - `sort.mojo` — sort kernels
  - `groupby.mojo` — `GroupBy` grouped aggregation (sum, min, max, count, mean)
  - `hashing.mojo` — hash_ for PrimitiveArray, StringArray, StructArray, AnyArray
"""

from marrow.dtypes import (
    PrimitiveType,
    Int8Type,
    Int16Type,
    Int32Type,
    Int64Type,
    UInt8Type,
    UInt16Type,
    UInt32Type,
    UInt64Type,
    Float16Type,
    Float32Type,
    Float64Type,
    bool_ as bool_dt,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
)
from .execution import ExecutionContext
from .aggregate import (
    SumKernel,
    ProductKernel,
    MinKernel,
    MaxKernel,
    MeanKernel,
    AnyKernel,
    AllKernel,
)
from .arithmetic import AddKernel, SubKernel, MulKernel, DivKernel
from .compare import (
    EqKernel,
    NeKernel,
    LtKernel,
    LeKernel,
    GtKernel,
    GeKernel,
)
from .cast import cast
from .distinct import count_distinct, approx_count_distinct
from .filter import filter, drop_null, take
from .sort import sort_indices, sort
