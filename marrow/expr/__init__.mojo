"""Type-erased, runtime-dispatched expression and relational-plan system.

Pick this package when the query isn't known until the program runs — this
is what the Python bindings drive, and what any dynamic-SQL or user-supplied-
schema caller needs. ``DynValue`` and ``AnyRelation`` build and execute plans
without knowing concrete comptime types, at the cost of tag/vtable dispatch
and a much larger compiled surface (~33x, stripped, in the measured
``benchmarks/binary_size/`` case) than the comptime counterpart in
``marrow.expr``, which pick that package instead when the query is fixed
at compile time.

A comptime-typed node from ``marrow.expr.values`` can be boxed into an
``DynValue`` via the ``DynValue(value)`` constructor (tag ``FUSED``); ``eval()``,
``dtype()``, and ``write_to()`` on a boxed node all delegate back to the
concrete comptime node, so a fused subtree keeps its single-pass execution
even when driven through this type-erased path.

``expr.mojo`` — ``DynValue`` (unified n-ary expression node), factory functions
(``col()``, ``lit()``, ``if_else()``), and the expression tag constants.

``plan.mojo`` — ``AnyRelation`` and its **self-executing** node kinds
(``Scan``, ``Filter``, ``Project``, ``InMemoryTable``, ``ParquetScan``,
``Aggregate``, ``Join``): each node is both the plan node and its own
pull-based executor (``pull()``/``collect()``), so there is no separate
``Planner``/``*Processor`` hierarchy. ``execute(plan)`` drains to one batch.

See ``docs/aot-relations-design.md`` for the full design.

Usage::

    var plan = in_memory_table(batch).filter(col("x") > lit[Int64Type](0)).select("x")
    var result = execute(plan)
"""

from marrow.expr.runtime import (
    # Unified expression node
    DynValue,
    # Free-standing factory functions (return DynValue)
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
    # Streaming execution (fat nodes; formerly executor.mojo)
    Relation,
    Exhausted,
    ExecutionContext,
    execute,
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
