"""Expression and logical plan system for Marrow.

Two expression layers
----------------------
``values.mojo`` — the **default** comptime-typed layer.  Nodes are generic
structs (``Column[T]``, ``Add[L, R]``, ``Sub[L, R]``) whose type parameters
encode the whole expression tree, so the compiler inlines evaluation into a
single fused SIMD loop with zero intermediate arrays.

``runtime.mojo`` — the type-erased runtime layer (``Expr``).  It exists so
query plans can be built and executed without knowing concrete comptime
types — this is what the Python bindings drive.  ``Expr`` carries a tag plus
child args and dispatches its own execution by tag in ``eval()``.

A comptime-typed node can be boxed into an ``Expr`` via the
``Expr(value)`` constructor (tag ``FUSED``); the boxed node's ``eval()``,
``dtype()``, and ``write_to()`` all delegate back to the concrete comptime
node, so a fused subtree keeps its single-pass execution even when driven
through the type-erased path.

Scalar expressions (runtime layer)
-----------------------------------
``Expr``   — unified n-ary term expression node
``Value``  — trait every expression node must implement (shared by both layers)

Factory functions: ``col()``, ``lit()``, ``if_else()``
Operator overloads: ``+``, ``-``, ``*``, ``/``, ``>``, ``<``, ``>=``,
``<=``, ``==``, ``!=``, ``&``, ``|``, ``~``, unary ``-``

Comptime-typed expressions
---------------------------
``NumericValue`` — base trait for numeric comptime nodes with SIMD vectorize execution

Expression nodes: ``Column[T]``, ``Add[L, R]``, ``Sub[L, R]``

Relational plans
----------------
``AnyRelation`` — type-erased relational plan node
``Relation``    — trait every relational plan node must implement

Concrete plan nodes: ``Scan``, ``Filter``, ``Project``, ``InMemoryTable``,
``ParquetScan``, ``Aggregate``, ``Join``
Plan-building: ``AnyRelation.select()``, ``AnyRelation.filter()``
Factory: ``in_memory_table()``, ``parquet_scan()``
"""

from marrow.expr.runtime import (
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
from marrow.expr.executor import (
    ExecutionContext,
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
from marrow.expr.values import (
    # Traits
    Value,
    NumericValue,
    # Expression nodes
    Column,
    Add,
    Sub,
    # Vectorize dispatch
    _vectorize_dispatch,
)
