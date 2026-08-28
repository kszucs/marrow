"""Compute kernels for marrow.

Re-exports the compute surface from the submodules so callers can
``import marrow.kernels as mk`` and use e.g. ``mk.AddKernel.dispatch``,
``mk.SumKernel.dispatch``, ``mk.filter``, ``mk.concat`` directly.

**The boundary, and why there is one.** Everything a caller *computes with* is
re-exported here: the element-wise kernels, the folds, and the free-function
verbs. Two things are deliberately not, and both are reached through their own
submodule rather than through this namespace:

  - **`MinKernel` / `MaxKernel`**, because the name means two different things.
    `numeric.MinKernel` is the element-wise binary minimum of two arrays;
    `aggregate.MinKernel` is `MinMax[MinOp]`, a whole-array fold. A flat
    namespace cannot hold both, so neither is re-exported — spell the one you
    mean (`from marrow.kernels.aggregate import MinKernel`). Their `*Op`
    building blocks (`MinOp`, `MaxOp`, `SumOp`, `ProductOp`) stay with them.
  - **The hash and partition machinery** — `groupby`, `join`, `hashtable`,
    `hashing`, `partition` — and the pruning algebra in `interval`. These are
    implementation modules that `expr` composes, not kernels a caller applies
    to an array.

This used to be neither: the docstring promised direct use of every submodule
while seven of nineteen were actually re-exported, so `mk.cast` worked and
`mk.concat` did not, with nothing saying why.

Submodules — element-wise first, then the ones that reshape or combine rows:
  - `numeric.mojo` — binary arithmetic and comparison (one family since Q0.7)
  - `boolean.mojo` — logical ops and validity predicates, Kleene semantics
  - `string.mojo` — string predicates and transforms, incl. LIKE / ILIKE
  - `temporal.mojo` — date/time field extraction and truncation
  - `cast.mojo` — type conversion
  - `conditional.mojo` — coalesce / nullif / case_when / fill_null
  - `membership.mojo` — `is_in`
  - `nested.mojo` — list-valued predicates
  - `aggregate.mojo` — reductions (sum, min, max, mean, any, all)
  - `distinct.mojo` — exact and approximate distinct counts
  - `filter.mojo` — selection, take, drop_null
  - `sort.mojo` — sort and sort_indices
  - `concat.mojo` — concatenation
  - `core.mojo` — the `Kernel` root trait and `Groups`
  - *not re-exported* — `groupby.mojo` (grouped aggregation), `join.mojo` /
    `hashtable.mojo` / `hashing.mojo` / `partition.mojo` (the hash machinery
    group-by, join and `is_in` share), `interval.mojo` (the interval algebra
    `expr/pruning.mojo` evaluates predicates in)

`ExecContext` is re-exported here for convenience but lives in
`marrow/execution.mojo`: it is a thread-count/device policy object that imports
nothing from marrow, and `views.mojo` and `tabular.mojo` need it too.
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
from ..execution import ExecContext
from .aggregate import (
    SumKernel,
    ProductKernel,
    MeanKernel,
    CountKernel,
    AnyKernel,
    AllKernel,
)
from .numeric import (
    AddKernel,
    SubKernel,
    MulKernel,
    DivKernel,
    FloordivKernel,
    ModKernel,
    PowKernel,
    NegKernel,
    AbsKernel,
    SignKernel,
    FloorKernel,
    CeilKernel,
    TruncKernel,
    RoundKernel,
    SqrtKernel,
    ExpKernel,
    Exp2Kernel,
    LogKernel,
    Log2Kernel,
    Log10Kernel,
    Log1pKernel,
    SinKernel,
    CosKernel,
    EqKernel,
    NeKernel,
    LtKernel,
    LeKernel,
    GtKernel,
    GeKernel,
    equal,
)
from .boolean import (
    AndKernel,
    OrKernel,
    NotKernel,
    XorKernel,
    IsNullKernel,
    NotNullKernel,
    IsNanKernel,
    IsInfKernel,
)
from .string import (
    LengthKernel,
    UpperKernel,
    LowerKernel,
    StripKernel,
    LStripKernel,
    RStripKernel,
    ReverseKernel,
    CapitalizeKernel,
    ConcatKernel,
    StartsWithKernel,
    EndsWithKernel,
    ContainsKernel,
    StringEqKernel,
    StringNeKernel,
    StringLtKernel,
    StringLeKernel,
    StringGtKernel,
    StringGeKernel,
    LikePattern,
    LikeKernel,
    ILikeKernel,
)
from .temporal import (
    YearKernel,
    MonthKernel,
    DayKernel,
    QuarterKernel,
    DayOfYearKernel,
    DayOfWeekKernel,
    HourKernel,
    MinuteKernel,
    SecondKernel,
    CalendarUnit,
    DateTruncKernel,
)
from .conditional import (
    Selection,
    CaseWhenKernel,
    case_when,
    CoalesceKernel,
    coalesce,
    NullifKernel,
    nullif,
    FillNullKernel,
    fill_null,
)
from .nested import ArrayLengthKernel, ArrayContainsKernel
from .cast import CastKernel, cast
from .concat import concat
from .core import Kernel, Groups
from .distinct import (
    count_distinct,
    approx_count_distinct,
)
from .filter import Filter, Take, filter, drop_null, take
from .membership import IsInKernel, is_in
from .sort import SortIndices, sort_indices, sort
