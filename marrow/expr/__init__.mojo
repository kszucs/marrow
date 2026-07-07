"""Type-erased, runtime-dispatched expression and relational-plan system.

Pick this package when the query isn't known until the program runs — this
is what the Python bindings drive, and what any dynamic-SQL or user-supplied-
schema caller needs. ``Expr`` and ``AnyRelation`` build and execute plans
without knowing concrete comptime types, at the cost of tag/vtable dispatch
and a much larger compiled surface (~33x, stripped, in the measured
``benchmarks/binary_size/`` case) than the comptime counterpart in
``marrow.expr``, which pick that package instead when the query is fixed
at compile time.

A comptime-typed node from ``marrow.expr.values`` can be boxed into an
``Expr`` via the ``Expr(value)`` constructor (tag ``FUSED``); ``eval()``,
``dtype()``, and ``write_to()`` on a boxed node all delegate back to the
concrete comptime node, so a fused subtree keeps its single-pass execution
even when driven through this type-erased path.

``expr.mojo`` — ``Expr`` (unified n-ary expression node), factory functions
(``col()``, ``lit()``, ``if_else()``), and the expression tag constants.

``relations.mojo`` — ``AnyRelation`` (type-erased relational plan node) and
its concrete node kinds (``Scan``, ``Filter``, ``Project``, ``InMemoryTable``,
``ParquetScan``, ``Aggregate``, ``Join``), plus the join kind/strictness/
algorithm constants.

``executor.mojo`` — ``Planner``/``*Processor`` (the pull-based streaming
execution engine) and the ``execute(plan)`` convenience wrapper.

See ``docs/aot-relations-design.md`` for the full design.

Usage::

    var plan = in_memory_table(batch).filter(col("x") > lit[Int64Type](0)).select("x")
    var result = execute(plan)
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
    LENGTH,
)
from marrow.expr.plan import (
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
