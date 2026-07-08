"""Unified expression + relational-plan system.

One value box, one interpreter, one relational layer. Which node you box
decides the binary size: box the fused comptime nodes and the interpreter is
dead-code-eliminated (~250 KB); box a ``DynValue`` and the runtime interpreter
links (parsed SQL / Python-driven plans). See ``docs/expr-unification-plan.md``.

``values.mojo`` — the fused comptime algebra (``Add``/``Gt``/``Length``…), the
``col(name, dtype)`` factory, and ``AnyValue`` — the universal value box the
relational layer holds (wraps a fused node *or* a ``DynValue``, exposing only
``to_array``). The named column leaves and ``Table[Tbl]()`` live in
``relations.mojo``.

``dynamic.mojo`` — ``DynValue``, the runtime tag-interpreter node the Python
bindings build, with factory functions (``col()``, ``lit()``, ``if_else()``)
and operator overloads.

``relations.mojo`` — a two-layer relational engine: **descriptive IR nodes**
(``Relation``: ``InMemoryTable``/``Filter``/``Project``/``Aggregate``/``Join``/
``ParquetScan``/``Scan``) that are pure, immutable, and cheaply copied, and the
**operators** (``Operator``) that ``Relation.open(ctx)`` builds to actually run
(pull-based, owning all execution state). ``execute(plan)`` opens the plan into a
fresh operator tree and drains it, so a plan is a reusable template.

Usage::

    var plan = in_memory_table(batch).filter(col("x") > lit[Int64Type](0)).select("x")
    var result = execute(plan)
"""

from marrow.expr.dynamic import (
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
    LENGTH,
)
from marrow.expr.relations import (
    # Descriptive IR nodes
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
    # Execution layer (operators built by Relation.open)
    Operator,
    AnyOperator,
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
    # Join kind constants (hash join: inner/left/right/full/semi/anti)
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_FULL,
    JOIN_SEMI,
    JOIN_ANTI,
    # Join strictness constants
    JOIN_ALL,
    JOIN_ANY,
)
