# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The expression and relational layer: build a plan, run it.

Two lanes that share no node types, meeting at `DynValue`:

- the **comptime** lane, where a subtree's structure lives in its *type* and
  fuses into one SIMD loop — `col("a", int64)` gives a `NumericColumn[Int64Type]`;
- the **runtime** lane, where structure lives in fields and each node
  materialises a column — `col("a")` gives a `RuntimeValue`.

`builders.mojo` is the one surface spanning both, and which lane you get is
decided by what you knew when you wrote the call.

**This module is the boundary, and it exists so `comptime` is escaped once.**
`comptime` is a reserved word, so naming that subpackage directly costs
backticks — ``from marrow.expr.`comptime`.numeric import Gt``. Re-exporting
here means a consumer writes `from marrow.expr import Gt` and never spells it.
The subpackage's own docstring claimed this was already true while this file
was empty (0 bytes) and 33 call sites escaped for themselves.

Re-exports are comptime aliases and generic structs, so a name nothing uses
costs nothing: the closed-erasure property that keeps `kernels::sort` out of a
binary that never sorts applies here too.
"""

from .bindings import Bindings, ParamSpec
from .cli import QueryCli, render_csv, render_table
from .builders import (
    array_contains,
    array_length,
    col,
    count_star,
    dense_rank,
    if_else,
    is_in,
    lit,
    maximum,
    minimum,
    param,
    rank,
    row_number,
    scan,
    scan_iceberg,
    scan_ipc,
    scan_json,
    table,
)
from .logical import (
    Aggregate,
    DynRelation,
    Difference,
    DynValue,
    Except,
    FileScan,
    Filter,
    IcebergScan,
    InMemoryTable,
    IpcScan,
    JsonScan,
    Intersect,
    Intersection,
    JoinChain,
    JoinFilter,
    JoinLink,
    JoinRef,
    JoinPricing,
    JoinRules,
    Limit,
    Multiplicity,
    Multiset,
    Over,
    ParquetScan,
    Project,
    References,
    Relation,
    ScanPath,
    Shape,
    Sort,
    Union,
    Value,
    Window,
    WindowCall,
    WindowFrame,
    WindowFunction,
    WindowSpec,
    WindowValue,
)
from .physical import (
    Datum,
    DynOperator,
    JoinOrder,
    Morsel,
    Operator,
    Pipeline,
    PlannedJoin,
)
from .index import ColumnZones, Index, keep_every
from .estimates import (
    Approx,
    ColumnEstimate,
    Cost,
    DEFAULT_SELECTIVITY,
    Estimate,
)

# -- the comptime lane, spelled here so nothing else has to --------------------
from .`comptime`.aggregates import (
    ApproxCountDistinct,
    Count,
    CountDistinct,
    Max,
    Mean,
    Min,
    Product,
    StdDev,
    Sum,
    Variance,
)
from .`comptime`.boolean import BoolBinary, IsIn, IsNull, Not, NotNull
from .`comptime`.casts import (
    BoolToNum,
    NumToBool,
    NumToString,
    NumericCast,
    StringToNum,
)
from .`comptime`.decimal import (
    DecimalBinary,
    DecimalCompare,
    DecimalDiv,
    DecimalRescale,
    DecimalToNum,
    DecimalToString,
    NumToDecimal,
    StringToDecimal,
)
from .`comptime`.core import (
    BinaryValue,
    BoolValue,
    ComptimeValue,
    DecimalValue,
    DictionaryValue,
    FixedSizeBinaryValue,
    FixedSizeListValue,
    IntervalValue,
    ListValue,
    NullValue,
    NumericValue,
    PrimitiveValue,
    StringValue,
    StructValue,
    TemporalValue,
)
from .`comptime`.nested import ArrayContains, ListLength
from .`comptime`.leaves import (
    BinaryColumn,
    BinaryLiteral,
    BinaryParam,
    BoolColumn,
    BoolLiteral,
    BoolParam,
    DecimalColumn,
    DecimalLiteral,
    DecimalParam,
    DictionaryColumn,
    DictionaryLiteral,
    DictionaryParam,
    FixedSizeBinaryColumn,
    FixedSizeBinaryLiteral,
    FixedSizeBinaryParam,
    FixedSizeListColumn,
    FixedSizeListLiteral,
    FixedSizeListParam,
    IntervalColumn,
    IntervalLiteral,
    IntervalParam,
    ListColumn,
    ListLiteral,
    ListParam,
    NullColumn,
    NullLiteral,
    NullParam,
    NumericColumn,
    NumericLiteral,
    NumericParam,
    StringColumn,
    StringViewColumn,
    StringLiteral,
    StringParam,
    StructColumn,
    StructLiteral,
    StructParam,
    TemporalColumn,
    TemporalLiteral,
    TemporalParam,
)
from .`comptime`.numeric import (
    Add,
    CaseWhen,
    Coalesce,
    Div,
    Eq,
    FillNull,
    Ge,
    Gt,
    Le,
    Lt,
    Mul,
    Ne,
    Nullif,
    NumericBinary,
    NumericCompare,
    Sub,
    TemporalGt,
)
from .`comptime`.rules import promote, widest_shape
from .`comptime`.strings import (
    EndsWith,
    ILike,
    Like,
    Lower,
    StartsWith,
    StringLength,
    Strip,
    Upper,
)

# -- the runtime lane ----------------------------------------------------------
from .runtime.aggregates import RuntimeAggregate
from .runtime.values import RuntimeValue

from .analyze import analyze, summarize
from .optimizer import (
    JoinOrdering,
    AllRules,
    ColumnPruning,
    EliminateFilter,
    MergeLimits,
    MergeProjects,
    PropagateEmpty,
    PushFilterBelowAggregate,
    MergeJoinChains,
    MergeProjectIntoJoin,
    PushFilterIntoJoin,
    Optimizer,
    RemoveEmptyLimit,
    NoRules,
    PushFilterBelowProject,
    PushFilterBelowSort,
    PushFilterIntoScan,
    PushLimitBelowProject,
    RemoveNoOpProject,
    RemoveRedundantSort,
    ScanPruning,
    Rule,
    RuleSet,
    TopN,
    optimize,
)
