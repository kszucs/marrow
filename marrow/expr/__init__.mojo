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

from .bindings import Bindings
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
    table,
)
from .logical import (
    Aggregate,
    DynRelation,
    DynValue,
    Filter,
    InMemoryTable,
    Join,
    Limit,
    ParquetScan,
    Project,
    Relation,
    Shape,
    Sort,
    Value,
    Window,
    WindowExpr,
    WindowFrame,
)
from .physical import Datum, DynOperator, Morsel, Operator, Pipeline
from .index import ColumnZones, Index, ZoneMaps, keep_every

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

from .optimizer import (
    AllRules,
    ColumnPruning,
    EliminateFilter,
    MergeLimits,
    MergeProjects,
    PropagateEmpty,
    PushFilterBelowAggregate,
    PushFilterBelowJoin,
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
