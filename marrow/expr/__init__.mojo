"""Expression and logical plan system for Marrow.

Scalar expressions
------------------
``AnyValue``    — type-erased scalar expression node (ArcPointer-backed)
``Value``       — trait every scalar expression node must implement

Concrete scalar nodes: ``Column``, ``Literal``, ``Binary``, ``Unary``,
``IsNull``, ``IfElse``, ``Cast``

Factory functions: ``col()``, ``lit()``, ``if_else()``
Utilities: ``rebuild()``, ``resolve_columns()``
Operator overloads: ``+``, ``-``, ``*``, ``/``, ``>``, ``<``, ``>=``,
``<=``, ``==``, ``!=``, ``&``, ``|``, ``~``, unary ``-``

Relational plans
----------------
``AnyRelation`` — type-erased relational plan node
``Relation``    — trait every relational plan node must implement

Concrete plan nodes: ``Scan``, ``Filter``, ``Project``, ``InMemoryTable``,
``ParquetScan``
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
from marrow.expr.fused_values import (
    # Traits
    TypedValue,
    NumericTypedValue,
    # Expression nodes
    FusedColumn,
    FusedAdd,
    FusedSub,
    # Vectorize dispatch
    _vectorize_dispatch,
)
