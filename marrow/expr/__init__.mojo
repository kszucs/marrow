"""Unified expression + relational-plan system.

One value box, one interpreter, one relational layer. Which node you box
decides the binary size: box the fused comptime nodes and the interpreter is
dead-code-eliminated (~250 KB); box a ``DynValue`` and the runtime interpreter
links (parsed SQL / Python-driven plans). See ``docs/expr-unification-plan.md``.

``values.mojo`` — the fused comptime value nodes (``NumericColumn``/``Add``/
``Gt``/``Length``…), the ``Table[Tbl]()`` / ``col(name, dtype)`` name-resolved
column handles, and ``AnyValue`` — the universal value box the relational layer
holds (wraps a fused node *or* a ``DynValue``, exposing only ``to_array``).

``runtime.mojo`` — ``DynValue``, the runtime tag-interpreter node the Python
bindings build, with factory functions (``col()``, ``lit()``, ``if_else()``)
and operator overloads.

``relations.mojo`` — ``AnyRelation`` and its **self-executing** fat nodes
(``InMemoryTable``, ``Filter``, ``Project``, ``Aggregate``, ``Join``,
``ParquetScan``, ``Scan``): each node is both the plan node and its own
pull-based executor (``pull()``/``collect()``) over ``List[AnyValue]`` — no
separate ``Planner``/``*Processor``. ``execute(plan)`` drains to one batch.

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
