"""Expression and logical plan system for Marrow.

Scalar expressions
------------------
``Expr``        — unified n-ary term expression node
``Value``       — trait every scalar expression node must implement

Factory functions: ``col()``, ``lit()``, ``if_else()``
Operator overloads: ``+``, ``-``, ``*``, ``/``, ``>``, ``<``, ``>=``,
``<=``, ``==``, ``!=``, ``&``, ``|``, ``~``, unary ``-``

Comptime-fused expressions
--------------------------
``NumericValue`` — base trait for numeric fused nodes with SIMD vectorize execution
``BoolValue``    — base trait for boolean output expression nodes

Expression nodes:
``Column[T]``     — typed column reference (positional)
``ColumnRef[name, T]`` — named column placeholder (resolved from RecordBatch)
``Literal[T]``    — scalar constant broadcast to all SIMD lanes
``Negate[T]``     — fused unary negate
``Add[L, R]``     — fused binary add
``Sub[L, R]``     — fused binary subtract
``Mul[L, R]``     — fused binary multiply
``Equal[L, R]``   — fused equality comparison
``NotEqual[L, R]`` — fused inequality comparison
``Less[L, R]``    — fused less-than comparison
``LessEq[L, R]``  — fused less-than-or-equal comparison
``Greater[L, R]`` — fused greater-than comparison
``GreaterEq[L, R]`` — fused greater-than-or-equal comparison
``And[L, R]``     — fused logical AND
``Or[L, R]``      — fused logical OR
``Not[E]``        — fused logical NOT

Fused expressions can be boxed into ``Expr`` via ``to_expr()`` or implicit
conversion, enabling use in plan-building APIs (``filter()``, ``select()``).

Relational plans
----------------
``AnyRelation`` — type-erased relational plan node
``Relation``    — trait every relational plan node must implement

Concrete plan nodes: ``Scan``, ``Filter``, ``Project``, ``InMemoryTable``,
``ParquetScan``, ``Aggregate``, ``Join``
Plan-building: ``AnyRelation.select()``, ``AnyRelation.filter()``
Factory: ``in_memory_table()``, ``parquet_scan()``

Rewriting
---------
``Rewrite``    — trait for non-destructive rewrite rules
``AnyRewrite`` — type-erased rule container
``Rewriter``   — bottom-up fixed-point rewrite driver
"""

from marrow.expr.values import (
    # Traits
    Value,
    # Unified expression node
    Expr,
    # Free-standing factory functions (return Expr)
    col,
    lit,
    if_else,
    # Node kinds
    LOAD,
    LITERAL,
    ADD,
    SUB,
    MUL,
    DIV,
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
    AND,
    OR,
    NEG,
    ABS,
    NOT,
    IS_NULL,
    IF_ELSE,
    CAST,
    FUSED,
)
from marrow.expr.relations import (
    Relation,
    AnyRelation,
    Scan,
    Filter,
    Project,
    InMemoryTable,
    ParquetScan,
    Aggregate,
    Join,
    in_memory_table,
    parquet_scan,
    # Plan node kind constants
    SCAN_NODE,
    FILTER_NODE,
    PROJECT_NODE,
    IN_MEMORY_TABLE_NODE,
    PARQUET_SCAN_NODE,
    AGGREGATE_NODE,
    JOIN_NODE,
    # Join kind constants
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_FULL,
    JOIN_SEMI,
    JOIN_ANTI,
    JOIN_CROSS,
    JOIN_MARK,
    JOIN_SINGLE,
    # Join strictness constants
    JOIN_ALL,
    JOIN_ANY,
    JOIN_ASOF,
    # Join algorithm hints
    JOIN_ALGO_AUTO,
    JOIN_ALGO_HASH,
    JOIN_ALGO_SORT_MERGE,
    JOIN_ALGO_PIECEWISE,
    JOIN_ALGO_GRACE_HASH,
)
from marrow.expr.rewrite import (
    Rewrite,
    AnyRewrite,
    Rewriter,
)
from marrow.expr.executor import (
    ExecutionContext,
    # Value processors
    ValueProcessor,
    AnyValueProcessor,
    ColumnProcessor,
    LiteralProcessor,
    BinaryProcessor,
    UnaryProcessor,
    IsNullProcessor,
    IfElseProcessor,
    FusedProcessor,
    # Relation processors
    RelationProcessor,
    AnyRelationProcessor,
    ScanProcessor,
    ParquetScanProcessor,
    FilterProcessor,
    ProjectProcessor,
    AggregateProcessor,
    JoinProcessor,
    Planner,
    execute,
)
from marrow.expr.fused import (
    # Traits
    TypedValue,
    NumericTypedValue,
    # Expression nodes (old API - aliases for compatibility)
    FusedColumn,
    FusedAdd,
    FusedSub,
    # Expression nodes (new API - aliases)
    FusedColumn as Column,
    FusedAdd as Add,
    FusedSub as Sub,
    # Vectorize dispatch
    _vectorize_dispatch,
)
