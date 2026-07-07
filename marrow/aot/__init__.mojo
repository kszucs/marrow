"""Comptime-typed, fully-monomorphized expression system.

Pick this package when the query is known at compile time and you want a
tiny, self-contained binary — every node's type parameters encode the whole
expression tree, so ``.execute()`` compiles straight to fused SIMD loops with
no tag dispatch and no vtables. See ``marrow.dyn`` for the runtime-typed
counterpart (used by the Python bindings and anywhere the plan isn't known
until the program runs) and ``docs/aot-relations-design.md`` for the full
design, including a measured ~31x binary-size difference between the two
(``benchmarks/binary_size/``).

``values.mojo`` — traits (``Value``, ``NumericValue``, ``StringValue``,
``BoolValue``), positional expression nodes (``NumericColumn[T]``, ``Add[L, R]``,
``Sub[L, R]``, ``Lt``, ``Gt``, ``Eq``, ``StringColumn``, ``Length[S]``), and the
polars-style ``col(name, dtype)`` factory (which builds the *named* leaves from
``relations.mojo``) — all re-exported here as the default surface.

``relations.mojo`` — the named relational layer: the ``Table[T]()``
column-access handle over a plain schema struct, the ``Column`` base trait and
its ``NumericColumn[T]`` / ``StringColumn`` leaves (only ``name`` is a runtime
field; position is resolved by name), and ``Project`` / ``Filter``. **Not**
re-exported here — its ``NumericColumn``/``StringColumn`` would collide with
``values.mojo``'s positional ones of the same name. Import explicitly:
``from marrow.aot.relations import Table, Project, Filter`` (``col`` lives in
``values.mojo``).

Usage (positional layer)::

    var col_a = NumericColumn[Int64Type](0)
    var col_b = NumericColumn[Int64Type](1)
    var expr = Add(col_a, col_b)
    var result = expr.execute(batch)

Usage (relational layer, see ``marrow.aot.relations`` for the full example)::

    struct Orders(Table):
        var a: Int32Type
        var b: StringType

    var t = table[Orders]()
    var plan = Project(Tuple(t.a, t.a)).filter(Gt(t.a, t.a))
    var result = execute(plan, batch)
"""

from marrow.aot.values import (
    # Traits
    Value,
    NumericValue,
    StringValue,
    BoolValue,
    # Expression nodes
    NumericColumn,
    Add,
    Sub,
    Lt,
    Gt,
    Eq,
    StringColumn,
    Length,
    # Column factory (resolved by name)
    col,
    # Vectorize dispatch
    _vectorize_dispatch,
)
