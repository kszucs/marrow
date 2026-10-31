# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Compute kernels for marrow.

Re-exports the compute surface so callers can ``import marrow.kernels as mk``
and use e.g. ``mk.AddKernel.dispatch``, ``mk.filter`` or ``mk.cast``.
Concatenation is a builder's: ``DynBuilder.concat`` and each typed builder's
``concat``. ``concat.mojo`` holds only what a builder cannot do, joining large
batches on several threads.

A name ending ``Kernel`` is something you *apply* to arrays. A ``*Fold`` (such
as ``SumFold``, ``Widening[AddKernel, SUM]``) is an algebra you *drive*:
``Fold[K, V]``, an ``AggKernel``, applies one to a column through ``reserve`` /
``update`` / ``finish``, or in one call through ``AggKernel.grouped``. The
``sum``, ``product``, ``min``, ``max`` and bitwise folds are built from the
kernels that claim ``Associative``, ``Commutative`` and ``Identity``:
``SumFold`` from ``AddKernel``, ``ProductFold`` from ``MulKernel``, ``MinFold``
and ``MaxFold`` from ``MinKernel`` and ``MaxKernel``, and ``BitAndFold``,
``BitOrFold`` and ``BitXorFold`` from ``BitAndKernel``, ``BitOrKernel`` and
``BitXorKernel``.

The hash and partition machinery -- `dictionary`, `join`, `hashtable`,
`hashing`, `partition`, which `expr` composes -- is reached through its own
submodule rather than this namespace; only the three public kernels on
`dictionary`'s encoder, `DictionaryEncodeKernel`, `ValueCountsKernel` and
`ModeKernel`, are re-exported.

Submodules — element-wise first, then the ones that reshape or combine rows:
  - `numeric.mojo` — binary arithmetic and comparison
  - `bitwise.mojo` — bitwise and, or, xor, not, and shifts over integers
  - `boolean.mojo` — logical ops and validity predicates, Kleene semantics
  - `string.mojo` — string predicates and transforms, incl. LIKE / ILIKE,
    and hex / base64 encoding
  - `temporal.mojo` — date/time field extraction and truncation
  - `cast.mojo` — type conversion
  - `conditional.mojo` — coalesce / nullif / case_when / fill_null
  - `membership.mojo` — `is_in`
  - `nested.mojo` — list-valued predicates, and folds over each list
  - `aggregate.mojo` — reductions (sum, min, max, mean, any, all)
  - `scan.mojo` — order-dependent kernels: running folds (cumulative
    sums, products, minimums and maximums), shift, pairwise_diff and
    rolling windows
  - `distinct.mojo` — exact and approximate distinct counts
  - `filter.mojo` — selection, take, drop_null, mask_validity, scatter,
    indices_nonzero / indices_to_mask, replace_with_mask
  - `sort.mojo` — sort and sort_indices
  - `core.mojo` — the `Kernel` root trait and the `Partial` and `Prunable`
    properties
  - `groupby.mojo` — `Groups`, a batch's rows assigned to group ids
  - `dictionary.mojo` — `dictionary_encode`, `value_counts` and `mode`
  - `runs.mojo` — `run_end_encode`, `run_end_decode` and `run_index`
  - *not re-exported* — the encoder and the dictionary packers in
    `dictionary.mojo`, the codec stages in `bitwise.mojo` and `cast.mojo`
    (`ZigzagEncode`, `OrderedEncode`, `ByteSwap`, `NumericCast`, ...),
    `join.mojo` / `hashtable.mojo` / `hashing.mojo` / `partition.mojo` (the
    hash machinery group-by, join and `is_in` share), `concat.mojo` (batches
    joined in parallel), `segments.mojo` (a column cut into consecutive runs
    of rows, and folds over each run), `RunsKernel` and the runs packers in
    `runs.mojo` (where runs of equal rows start), `statistics.mojo` (column
    bounds), `compose.mojo` (stream stages composed, and codecs built from
    stages and packers) and `packing.mojo` (values and byte strings packed
    into bytes and back)
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
    BitAndFold,
    BitOrFold,
    BitXorFold,
    AnyKernel,
    AllKernel,
)
from .bitwise import (
    BitAndKernel,
    BitOrKernel,
    BitXorKernel,
    BitNotKernel,
    ShiftLeftKernel,
    ShiftRightKernel,
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
    ToHexKernel,
    FromHexKernel,
    ToBase64Kernel,
    FromBase64Kernel,
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
    DateTruncKernel,
)
from .conditional import (
    Selection,
    CaseWhenKernel,
    CoalesceKernel,
    NullifKernel,
    FillNullKernel,
)
from .nested import ArrayLengthKernel, ArrayContainsKernel, ListReduceKernel
from .scan import (
    CumulativeKernel,
    PairwiseDiffKernel,
    RollingKernel,
    RollingMeanKernel,
    Scan,
    ShiftKernel,
    SumScan,
)
from .cast import CastArm, CastKernel
from .core import Kernel, Partial, Prunable
from .groupby import Groups
from .dictionary import DictionaryEncodeKernel, ModeKernel, ValueCountsKernel
from .runs import RunIndexKernel, RunEndDecodeKernel, RunEndEncodeKernel
from .distinct import ApproxCountDistinctKernel, CountDistinctKernel
from .filter import (
    FilterKernel,
    IndicesNonzeroKernel,
    IndicesToMaskKernel,
    MaskValidityKernel,
    ReplaceWithMaskKernel,
    ScatterKernel,
    TakeKernel,
)
from .membership import IsInKernel
from .sort import SortIndices, SortKernel
