# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Compute kernels for marrow.

Re-exports the compute surface so callers can ``import marrow.kernels as mk``
and use e.g. ``mk.AddKernel.dispatch``, ``mk.filter`` or ``mk.cast``.
Concatenation is not a kernel: it is ``DynBuilder.concat`` and each typed
builder's ``concat``.

A name ending ``Kernel`` is something you *apply* to arrays. A ``*Fold`` (such
as ``SumFold``, ``Widening[SumOp]``) is an algebra you *drive*: ``Fold[K, V]``,
an ``AggKernel``, applies one to a column through ``reserve`` / ``update`` /
``finish``, or in one call through ``AggKernel.grouped``.

Two things are reached through their own submodule rather than this namespace:

  - **The `*Op` building blocks** — `MinOp`, `MaxOp`, `SumOp`, `ProductOp`,
    the algebras the folds are assembled from.
  - **The hash and partition machinery** — `dictionary`, `join`,
    `hashtable`, `hashing`, `partition`, which `expr` composes.

Submodules — element-wise first, then the ones that reshape or combine rows:
  - `numeric.mojo` — binary arithmetic and comparison
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
  - `core.mojo` — the `Kernel` root trait
  - `groupby.mojo` — `Groups`, a batch's rows assigned to group ids
  - *not re-exported* — `dictionary.mojo` / `join.mojo` / `hashtable.mojo` /
    `hashing.mojo` / `partition.mojo` (the hash machinery group-by, join and
    `is_in` share)
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
    AggKernel,
    Fold,
    FoldKernel,
    Foldable,
    SumFold,
    ProductFold,
    MeanFold,
    CountFold,
    MinFold,
    MaxFold,
    AnyKernel,
    AllKernel,
)
from .numeric import (
    AddKernel,
    MinKernel,
    MaxKernel,
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
    EqualKernel,
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
    MatchLikeKernel,
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
from .core import Kernel
from .groupby import Groups
from .distinct import ApproxCountDistinctKernel, CountDistinctKernel
from .filter import FilterKernel, TakeKernel
from .membership import IsInKernel
from .sort import SortIndices, SortKernel
