"""Unified expression + relational-plan system.

One value box, one interpreter, one relational layer. Which node you box
decides the binary size: box the fused comptime nodes and the interpreter is
dead-code-eliminated (~250 KB); box a ``TagValue`` and the runtime interpreter
links (parsed SQL / Python-driven plans). See ``docs/expr-unification-plan.md``.

``values.mojo`` — the fused comptime algebra (``Add``/``Greater``/``Length``…), the
named column leaves (``NumericColumn``/``StringColumn``) with the ``Table[Tbl]()``
and ``col(name, dtype)`` builders, and ``DynValue`` — the universal value box the
relational layer holds (wraps a fused node *or* a ``TagValue``, exposing only
``DynRelation.execute``).

``dynamic.mojo`` — ``TagValue``, the runtime tag-interpreter node the Python
bindings build, with factory functions (``col()``, ``lit()``, ``if_else()``)
and operator overloads.

``relations.mojo`` — the **descriptive IR**: ``Relation`` nodes
(``InMemoryTable``/``Filter``/``Project``/``Aggregate``/``Join``/``ParquetScan``)
that are pure, immutable, and cheaply copied, plus the plan-building API and
``plan.execute()``.

``execution.mojo`` — the **execution layer**: the ``Processor`` each
``Relation.to_processor(ctx)`` builds (pull-based, owning all mutable state — offset,
hash index, grouper, child processors), erased behind ``DynProcessor`` which
drives ``collect()``. ``plan.execute()`` opens a plan into a fresh processor tree and
drains it, so a plan is a reusable template. Depends only on the value box and
kernels (one-way: ``relations`` → ``execution``).

Usage::

    var plan = in_memory_table(batch).filter(col("x") > lit[Int64Type](0)).select("x")
    var result = plan.execute()
"""

from .dynamic import (
    # Unified expression node
    TagValue,
    # Free-standing factory functions (return TagValue)
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
    NOT_NULL,
    IF_ELSE,
    LENGTH,
    CAST,
    MOD,
    FLOORDIV,
    XOR,
)
from ..kernels.execution import ExecutionContext
from .aggregates import AggFunc, FoldedAggregates
from .execution import (
    # Execution layer (processors built by Relation.to_processor)
    Processor,
    DynProcessor,
    Exhausted,
)
from .relations import (
    # Descriptive IR nodes
    Relation,
    DynRelation,
    Filter,
    Project,
    InMemoryTable,
    ParquetScan,
    Aggregate,
    Join,
    in_memory_table,
    parquet_scan,
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
