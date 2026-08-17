"""Expression + relational-plan system, in two lanes.

Expressions come in two forms and they share no node types. Which one you build
decides the binary size: build fused comptime nodes and the runtime interpreter
is dead-code-eliminated (~250 KB); build ``DynValue`` and it links (parsed SQL /
Python-driven plans). Both erase into ``BoxedValue``, which is what the
relational layer holds.

``values.mojo`` — the **AOT lane**: the fused comptime algebra
(``Add``/``Greater``/``Length``…) and the named column leaves
(``NumericColumn``/``StringColumn``) with the ``col(name, dtype)`` builder. Every
operand is bound on its family trait, so a tree of these fuses into one SIMD
loop and nothing in it is erased.

``dynamic.mojo`` — the **runtime lane**: ``DynValue``, one struct holding a tag,
its children and an optional payload, which evaluates by dispatching on the tag
and the operands' runtime dtypes. Built by the untyped factories (``col(name)``,
``lit()``, ``if_else()``) for expressions whose types are not known where they
are written.

``core.mojo`` — the vocabulary the two lanes share: ``Datum`` and
``into_array``. A leaf, so ``dynamic`` no longer imports ``values`` for them.

``builders.mojo`` — the whole ``col``/``lit``/``if_else``/``coalesce``/
``case_when`` overload set, both lanes together. It sits above both lanes
because an overload set cannot span modules.

``relations.mojo`` — the **descriptive IR**: ``Relation`` nodes
(``InMemoryTable``/``Filter``/``Project``/``Aggregate``/``Join``/``ParquetScan``)
that are pure, immutable, and cheaply copied, plus the plan-building API and
``plan.execute()``.

``execution.mojo`` — the **execution layer**: the ``Processor`` each
``Relation.to_processor(ctx)`` builds (pull-based, owning all mutable state — offset,
hash index, grouper, child processors), erased behind ``DynProcessor`` which
drives ``collect()``. ``plan.execute()`` opens a plan into a fresh processor tree and
drains it, so a plan is a reusable template. ``execution`` no longer imports
``relations``: the ``BoxedValue`` both needed moved down beside ``Value``, which
is what leaves ``relations -> execution`` a one-way edge.

One cycle survives, ``values <-> dynamic``, and it is structural rather than a
placement accident: ``DynValue`` conforms to ``Value``, ``Value`` defaults
``count_distinct`` to an ``AggExpr``, and ``AggExpr`` carries an unresolved
``DynValue``. Those three cannot be split across modules by moving code.

Usage::

    var plan = in_memory_table(batch).filter(col("x") > lit[Int64Type](0)).select("x")
    var result = plan.execute()
"""

from .dynamic import DynAgg, DynValue
from .builders import col, lit, if_else
from .values import BoxedValue
from ..execution import ExecContext
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
    Sort,
    Limit,
    in_memory_table,
    parquet_scan,
    # Join kind constants (hash join: inner/left/right/full/semi/anti)
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_FULL,
    JOIN_SEMI,
    JOIN_ANTI,
    JoinKind,
    # Join strictness constants
    JOIN_ALL,
    JOIN_ANY,
)
