"""Compute kernels for marrow.

Re-exports kernel structs from the submodules so callers can
``import marrow.kernels as mk`` and use e.g. ``mk.AddKernel.dispatch``,
``mk.SumKernel.dispatch``, ``mk.filter``, ``mk.sort`` directly.

Submodules — element-wise first, then the ones that reshape or combine rows:
  - `numeric.mojo` — binary arithmetic and comparison (one family since Q0.7)
  - `boolean.mojo` — logical ops and validity predicates, Kleene semantics
  - `string.mojo` — string predicates and transforms
  - `temporal.mojo` — date/time field extraction and truncation
  - `cast.mojo` — type conversion
  - `conditional.mojo` — coalesce / nullif / case_when
  - `membership.mojo` — `is_in`
  - `nested.mojo` — list-valued predicates
  - `aggregate.mojo` — reductions (sum, min, max, mean, any, all)
  - `distinct.mojo` — exact and approximate distinct counts
  - `groupby.mojo` — grouped aggregation
  - `join.mojo` / `hashtable.mojo` / `hashing.mojo` / `partition.mojo` — the
    hash machinery group-by, join and `is_in` share
  - `filter.mojo` — selection, take, drop_null
  - `sort.mojo` — sort and sort_indices
  - `concat.mojo` — concatenation
  - `core.mojo` — the `Kernel` root trait; `execution.mojo` — `ExecutionContext`
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
from .numeric import (
    AddKernel,
    SubKernel,
    MulKernel,
    DivKernel,
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
from .membership import IsInKernel, is_in
from .sort import sort_indices, sort
