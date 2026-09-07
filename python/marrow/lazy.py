"""The lazy relational frontend — an ibis-flavoured ``LazyTable`` over ``Plan``.

Nothing here executes until :meth:`LazyTable.collect`. Every verb returns a new
``LazyTable`` wrapping a new plan, because the underlying ``DynRelation`` is an
immutable description: copying a plan is a refcount bump, so chaining is cheap
and a plan can be executed more than once.

    import marrow

    t = marrow.read_parquet("hits.parquet")
    top = (
        t.filter(t["price"] > 100)
        .aggregate(by=["region"], total=("sum", "price"))
        .order_by(("total", "descending"))
        .head(10)
    )
    print(top)            # the plan — a free EXPLAIN
    batch = top.collect()  # now it runs

"ibis-flavoured" is a naming guideline only: there is no ``ibis`` dependency
and this is not an ibis backend.

**The verbs are the Mojo lane's verbs.** ``filter``, ``select``, ``project``,
``with_columns``, ``drop``, ``rename``, ``limit``, ``aggregate`` and ``join``
are `DynRelation`'s own methods and take the same arguments in the same order,
so a query reads the same in either language. Only three things are added here,
each because Mojo cannot express it: ``**kwargs`` for named projections and
aggregates, the ``on=``/dict/``"descending"`` shorthands, and ``order_by`` as
an alias for ``sort_by`` (which is what `DynRelation` calls it).
"""

from . import libmarrow as _ma
from ._wrapper import _Wrapper, unwrap
from .expr import Aggregate, Column, Window, col
from .tabular import RecordBatch
from .types import Schema

__all__ = ["LazyTable", "memtable", "read_parquet", "sql"]


def _sort_key(entry):
    """One ``order_by`` entry -> ``(key, ascending)``.

    Accepts ``"col"``, ``("col", "ascending"|"descending")``, ``("col", bool)``
    and a ``Column``. The string spelling matches ``RecordBatch.sort_by`` and
    PyArrow, so the eager and lazy surfaces order rows the same way.
    """
    if isinstance(entry, tuple):
        key, direction = entry
        if isinstance(direction, bool):
            ascending = direction
        else:
            ascending = direction != "descending"
        return unwrap(key), ascending
    return unwrap(entry), True


def _aggregate_spec(name, value):
    """One named aggregate -> what ``Plan.aggregate`` accepts.

    ``("sum", "amount")`` becomes the ``(func, column, out_name)`` triple the
    binding marshals; an ``Aggregate`` is aliased to the keyword it was given.
    """
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(
                f"aggregate: {name}=... expects (func, column), got {value!r}"
            )
        func, column = value
        return (func, column, name)
    if isinstance(value, str):
        # `t.aggregate(by=["k"], n="count")` — count over what?
        raise ValueError(
            f"aggregate: {name}={value!r} is ambiguous; use "
            f'{name}=("{value}", "<column>")'
        )
    if isinstance(value, Column):
        raise ValueError(
            f"aggregate: {name}={value.render()!r} is a column expression, "
            f"not an aggregate; call a reduction on it, e.g. "
            f"{name}=marrow.col(...).sum()"
        )
    return value.unwrap().alias(name)


def _projection(positional, named, verb):
    """`(names, values)` from either keywords or two parallel lists.

    `values` are the caller's objects, not unwrapped: `with_columns` has to see
    whether they are window functions before it can pick a plan node, and
    re-deriving which branch was taken in order to find that out is how the two
    got out of step."""
    if positional and named:
        raise TypeError(f"{verb}: pass keywords or two lists, not both")
    if positional:
        if len(positional) != 2:
            raise TypeError(
                f"{verb}: positional form takes exactly two lists "
                f"(names, values), got {len(positional)}"
            )
        names, values = positional
        names = [str(n) for n in names]
        values = list(values)
        if len(names) != len(values):
            raise ValueError(f"{verb}: {len(names)} names but {len(values)} values")
        return names, values
    names = list(named)
    return names, [named[n] for n in names]


class LazyTable(_Wrapper):
    """A lazy relational table: a query plan you can keep composing.

    Wraps the bound ``Plan`` type. Named ``LazyTable`` rather than ``Table``
    because ``marrow.Table`` is the eager, PyArrow-shaped table.
    """

    # -- introspection ----------------------------------------------------

    @property
    def schema(self):
        return Schema.wrap(self._binding.schema())

    @property
    def column_names(self):
        return self._binding.column_names()

    def __getitem__(self, name):
        """A column reference — ``t["price"]`` — for building predicates."""
        return col(name)

    def __repr__(self):
        return f"LazyTable\n{self._plan_text()}"

    def __str__(self):
        return self._plan_text()

    def _plan_text(self):
        """The bound plan's own ``__str__``, called explicitly.

        ``str(binding)`` does *not* reach it: ``def_method`` fills the type's
        ``tp_dict``, and ``PythonTypeBuilder.bind`` never installs a ``tp_str``
        slot, so ``str()`` falls back to ``tp_repr`` and returns
        ``"<marrow.Plan: Sort(...)>"``. Verified on this build.
        """
        return self._binding.__str__()

    # -- relational verbs -------------------------------------------------

    def select(self, *names):
        """Project columns by name. ``t.select("a", "b")``.

        A real ``select`` lowering, not a ``project`` of column reads:
        ``project`` probes each expression's dtype and builds a fresh
        ``Field``, so a non-nullable column would come out nullable and its
        metadata would be dropped. ``select`` copies the input field whole."""
        if len(names) == 1 and isinstance(names[0], (list, tuple)):
            names = tuple(names[0])
        return LazyTable.wrap(self._binding.select([str(n) for n in names]))

    def drop(self, *names):
        """Every column except these — ``t.drop("a", "b")``.

        The survivors keep their **input order**, and an unknown name raises
        rather than being ignored: a typo in a ``drop`` list is otherwise
        silent, and the column it meant to remove survives."""
        if len(names) == 1 and isinstance(names[0], (list, tuple)):
            names = tuple(names[0])
        return LazyTable.wrap(self._binding.drop([str(n) for n in names]))

    def rename(self, names, new_names=None):
        """Rename columns, leaving the rest untouched and in place.

        Two spellings, and both are somebody's native one::

            t.rename({"v": "value"})           # polars', and this frontend's
            t.rename(["v"], ["value"])         # `DynRelation.rename`'s

        The second is what the plan node takes and what a Mojo golden case
        writes, so it is accepted here verbatim rather than adapted by a shim
        in the corpus."""
        if new_names is None:
            mapping = names
            old_names = [str(k) for k in mapping]
            new_names = [str(v) for v in mapping.values()]
        else:
            old_names = [str(n) for n in names]
            new_names = [str(n) for n in new_names]
        if len(old_names) != len(new_names):
            raise ValueError(
                f"rename: {len(old_names)} names but {len(new_names)} new names"
            )
        return LazyTable.wrap(self._binding.rename(old_names, new_names))

    def project(self, *positional, **named):
        """Computed columns — ``t.project(total=t["a"] + t["b"])``.

        Keywords name the output columns, so this replaces the projection
        entirely (it is ``SELECT <these>``, not ``with_columns``).

        Two parallel lists — ``t.project(["total"], [t["a"] + t["b"]])`` — are
        also accepted, which is how the plan node and the Mojo lane spell it.
        Mojo has no ``**kwargs``, so the positional form is the only one both
        lanes can share.
        """
        names, values = _projection(positional, named, "project")
        return LazyTable.wrap(
            self._binding.project(names, [unwrap(v) for v in values])
        )

    def with_columns(self, *positional, **named):
        """Add or replace computed columns, keeping every other one.

            t.with_columns(total=t["qty"] * t["price"])
            t.with_columns(rn=marrow.row_number().over(order_by=["v"]))

        ``project``'s usable half, and the verb polars and ibis lean on
        hardest: a new name is appended, an existing one is replaced **at its
        original position**, and every expression sees the *input* columns
        rather than a partially-updated output. Chain two calls for sequential
        semantics.

        Takes the same two shapes as :meth:`project` — keywords, or two
        parallel lists. The output name is always written, never derived from
        the expression.

        **Window functions go to a different plan node.** `DynRelation` has
        two `with_columns` overloads and they are deliberately disjoint: a
        `List[WindowExpr]` cannot convert to a `List[DynValue]`, so the plan
        layer cannot confuse a windowed projection with an ordinary one. The
        values are all-or-nothing here for the same reason — mixing the two in
        one call would have to split into two nodes, and which one ran first
        would change the answer.
        """
        names, values = _projection(positional, named, "with_columns")
        windows = [isinstance(v, Window) for v in values]
        bindings = [unwrap(v) for v in values]
        if any(windows):
            if not all(windows):
                raise TypeError(
                    "with_columns: pass window functions or ordinary "
                    "expressions, not both in one call — chain two calls"
                )
            return LazyTable.wrap(
                self._binding.with_window_columns(names, bindings)
            )
        return LazyTable.wrap(self._binding.with_columns(names, bindings))

    # ibis spells `with_columns` as `mutate`. Both work.
    mutate = with_columns

    def filter(self, predicate):
        """Keep rows where ``predicate`` is true."""
        return LazyTable.wrap(self._binding.filter(unwrap(predicate)))

    def aggregate(self, *args, by=None, aggs=None, keys=None, **named_aggs):
        """Grouped aggregation.

            t.aggregate(by=["region"], total=("sum", "price"), n=("count", "id"))

        ``by`` is a list of column names or expressions; an empty ``by`` is one
        implicit group (``SELECT sum(x)`` with no ``GROUP BY``). Keyword
        aggregates name their output column; positional ones must carry their
        own ``.alias(...)``.

        Keys with no aggregates is ``SELECT DISTINCT``, which the plan layer
        executes. Neither keys nor aggregates is meaningless and raises.

        **The plan node's own spelling is also accepted**, as keywords::

            t.aggregate(aggs=[col("v").sum()], keys=["region"])

        `DynRelation.aggregate` takes ``(aggs, keys)`` in that order, where
        this frontend leads with ``by`` -- so the two cannot share a positional
        form, and a golden case writes the keyword one to run as a single text
        in both lanes. Passing ``aggs=`` or ``keys=`` selects it, which is why
        neither name can also be an output column here.
        """
        if aggs is None and keys is None and _is_aggregate_list(args):
            # `aggregate([col("v").sum()], ["k"])` -- the plan node's order.
            # Unambiguous: `by` names grouping keys, and an aggregate can never
            # be one (`reject_aggregate` refuses it at the node).
            aggs = args[0]
            keys = args[1] if len(args) > 1 else ()
            args = ()
        if aggs is not None or keys is not None:
            if args or by is not None or named_aggs:
                raise TypeError(
                    "aggregate: the plan form takes only `aggs=` and `keys=`"
                )
            specs = [unwrap(a) for a in (aggs or ())]
            key_list = [unwrap(k) for k in (keys or ())]
        else:
            positional = args
            if by is None and positional:
                by, positional = positional[0], positional[1:]
            if by is None:
                by = ()
            if isinstance(by, (str, bytes)) or not hasattr(by, "__iter__"):
                by = [by]
            key_list = [unwrap(k) for k in by]
            specs = [unwrap(a) for a in positional]
            specs += [_aggregate_spec(n, v) for n, v in named_aggs.items()]
        if not specs and not key_list:
            raise ValueError("aggregate: needs at least one key or aggregate")
        return LazyTable.wrap(self._binding.aggregate(key_list, specs))

    def order_by(self, *keys, nulls_first=True):
        """Sort. ``t.order_by("a", ("b", "descending"))``."""
        if len(keys) == 1 and isinstance(keys[0], list):
            keys = tuple(keys[0])
        if not keys:
            raise ValueError("order_by: needs at least one key")
        resolved = [_sort_key(k) for k in keys]
        return LazyTable.wrap(
            self._binding.sort(
                [k for k, _ in resolved],
                [asc for _, asc in resolved],
                nulls_first,
            )
        )

    def sort_by(self, *args, keys=None, ascending=None, nulls_first=True):
        """Sort — PyArrow's name for :meth:`order_by`, and the plan node's.

        Takes either spelling::

            t.sort_by("a", ("b", "descending"))               # this frontend's
            t.sort_by(keys=[col("a")], ascending=[False])     # the plan node's

        `DynRelation.sort_by` takes parallel key and direction lists, which is
        what a Mojo golden case writes; passing ``keys=`` selects that form so
        the one text runs in both lanes."""
        if keys is None and _is_direction_list(args):
            # `sort_by([col("a")], [True])` -- the plan node's positional form.
            # Unambiguous: the friendly form's arguments are names, tuples or
            # expressions, and never a list of bools.
            keys, ascending = args[0], args[1]
            args = ()
        if keys is not None or ascending is not None:
            if args:
                raise TypeError(
                    "sort_by: the plan form takes only `keys=` and `ascending=`"
                )
            key_list = list(keys or ())
            if not key_list:
                raise ValueError("sort_by: needs at least one key")
            directions = (
                list(ascending)
                if ascending is not None
                else [True] * len(key_list)
            )
            if len(directions) != len(key_list):
                raise ValueError(
                    f"sort_by: {len(key_list)} keys but "
                    f"{len(directions)} directions"
                )
            # Normalised into `order_by`'s `(key, bool)` pairs rather than
            # reaching the binding separately: one sorting path, so the two
            # spellings cannot disagree about defaults or null placement.
            return self.order_by(
                *zip(key_list, [bool(d) for d in directions]),
                nulls_first=nulls_first,
            )
        return self.order_by(*args, nulls_first=nulls_first)

    def limit(self, n, offset=0):
        """At most ``n`` rows, after skipping ``offset``."""
        return LazyTable.wrap(self._binding.limit(n, offset))

    def head(self, n=5):
        return self.limit(n, 0)

    def join(
        self,
        other,
        *positional,
        on=None,
        left_on=None,
        right_on=None,
        how="inner",
        left_keys=None,
        right_keys=None,
        kind=None,
    ):
        """Equijoin. ``on`` is shorthand for equal key names on both sides.

        Keys may be given by **name**, resolved against each side's schema, or
        by **position**::

            t.join(other, on="region")                                  # names
            t.join(other, left_keys=[0], right_keys=[0], kind="inner")  # the
                                                                        # plan's

        `Plan.join` takes column indices, because the join operator hashes
        whole columns of the input: a key is a position in the schema rather
        than an expression to evaluate. The Mojo lane has no schema in hand at
        plan-build time and so names its keys positionally, which is what a
        golden case writes; ``left_keys=``/``right_keys=``/``kind=`` are those
        parameter names verbatim.

        A key that names no column raises here, where the schema is in hand and
        the message can say which side it looked in.
        """
        if positional:
            # `join(other, [0], [0], JOIN_INNER)` -- the plan node's form.
            # Unambiguous: the name-based spelling is keyword-only.
            if left_keys is not None or right_keys is not None:
                raise TypeError("join: keys given both positionally and by name")
            left_keys = positional[0]
            right_keys = positional[1] if len(positional) > 1 else positional[0]
            if len(positional) > 2 and kind is None:
                kind = positional[2]
        if left_keys is not None or right_keys is not None:
            if on is not None or left_on is not None or right_on is not None:
                raise TypeError(
                    "join: pass either names (`on`/`left_on`/`right_on`) or "
                    "positions (`left_keys`/`right_keys`), not both"
                )
            left, right = list(left_keys or ()), list(right_keys or ())
        else:
            if on is not None:
                left_on = right_on = on
            if left_on is None or right_on is None:
                raise ValueError(
                    "join: pass `on`, or both `left_on` and `right_on`"
                )
            if isinstance(left_on, (str, bytes)):
                left_on = [left_on]
            if isinstance(right_on, (str, bytes)):
                right_on = [right_on]
            left = _key_indices(self.column_names, left_on, "left")
            right = _key_indices(other.column_names, right_on, "right")
        if len(left) != len(right):
            raise ValueError(
                f"join: {len(left)} left keys but {len(right)} right keys"
            )
        return LazyTable.wrap(
            self._binding.join(
                other.unwrap(), left, right, kind if kind is not None else how
            )
        )

    # -- execution --------------------------------------------------------

    def collect(self, num_threads=0):
        """Run the plan and return one eager :class:`marrow.RecordBatch`.

        The whole plan is drained, so a multi-row-group Parquet scan comes back
        complete rather than one row group at a time.

        ``num_threads`` is the CPU worker budget, spelled and defaulted exactly
        as on the eager surface (``RecordBatch.group_by(..., num_threads=0)``):

        * ``0`` — **auto** (the default): each kernel picks serial vs all-cores
          from its own row-count threshold, so a small query pays no worker
          setup and a large one uses the machine.
        * ``1`` — serial, forced.
        * ``N >= 2`` — exactly ``N`` workers, forced, threshold bypassed.

        It lives here rather than on the constructor because ``collect`` is the
        only place a plan actually *runs*: a ``LazyTable`` is an immutable plan
        that every verb returns a fresh copy of, and a stored worker count
        would have to survive ``join``, where two tables with different
        settings have no defensible winner.
        """
        return RecordBatch.wrap(self._binding.execute(num_threads))

    def batches(self, num_threads=0):
        """Run the plan and return the engine's own batches, unconcatenated.

        `collect()` calls `Pipeline.collect`, which drains the chain and
        concatenates into one batch; this keeps the boundaries the engine
        already produced, which is the shape `Table.from_batches` wants."""
        return [
            RecordBatch.wrap(b) for b in self._binding.batches(num_threads)
        ]

    def to_table(self, num_threads=0):
        """Run the plan and return an eager :class:`~marrow.Table`."""
        from .tabular import Table

        return Table.from_batches(self.batches(num_threads))

    def to_pyarrow(self, num_threads=0):
        """Run the plan and hand the result to PyArrow (zero-copy, C Data)."""
        import pyarrow as pa

        return pa.record_batch(self.collect(num_threads))

    def optimize(self):
        """The plan the rewriter would run — fifteen rules plus column pruning.

        `collect()` alone optimizes nothing, so this is opt-in. The result is
        an ordinary `LazyTable`: print it, diff it against this one, keep
        composing it, or run it."""
        return LazyTable.wrap(self._binding.optimize())

    def explain(self):
        """The plan as text, without running it.

        Renders recursively — a ``Filter`` names the node it filters — so this
        is the whole tree, not just the root."""
        return self._plan_text()


def _is_direction_list(args):
    """`sort_by(keys, ascending)` — a second argument that is a list of bools.

    The friendly spelling passes names, `("col", "descending")` tuples or
    expressions, so a list of bools in that position can only be the plan
    node's parallel direction list."""
    return (
        len(args) == 2
        and isinstance(args[0], (list, tuple))
        and isinstance(args[1], (list, tuple))
        and len(args[1]) > 0
        and all(isinstance(d, bool) for d in args[1])
    )


def _is_aggregate_list(args):
    """`aggregate(aggs, keys)` — a first argument that is a list of aggregates.

    `by` names grouping keys, and an aggregate can never be one: `Aggregate`
    in the key position is what `reject_aggregate` refuses at the plan node.
    So a non-empty list of `Aggregate` unambiguously means the plan's order."""
    return (
        len(args) in (1, 2)
        and isinstance(args[0], (list, tuple))
        and len(args[0]) > 0
        and all(isinstance(a, Aggregate) for a in args[0])
    )


def _key_indices(column_names, keys, side):
    """Join keys by name -> positions in `column_names`."""
    out = []
    for key in keys:
        if isinstance(key, int):
            out.append(key)
            continue
        name = key.name() if isinstance(key, Column) else str(key)
        if name not in column_names:
            raise ValueError(
                f"join: {side} key '{name}' is not a column of that table "
                f"(have {list(column_names)})"
            )
        out.append(column_names.index(name))
    return out


# ── Entry points ───────────────────────────────────────────────────────────


def read_parquet(path, schema=None):
    """A lazy table over a Parquet file.

    The schema doubles as the projection — only its columns are read — and is
    inferred from the file's footer when omitted (metadata only, no column
    data).
    """
    return LazyTable.wrap(_ma.parquet_scan(str(path), unwrap(schema)))


def memtable(batch):
    """A lazy table over an in-memory :class:`marrow.RecordBatch`.

    ``memtable`` is ibis's name for exactly this, and the pairing is the point:
    the *verb* says which world you are in. Lazy is ``memtable`` /
    ``read_parquet``, eager is ``table`` / ``record_batch`` — each namespace
    spelled consistently with the library it is modelled on.
    """
    return LazyTable.wrap(_ma.in_memory_table(unwrap(batch)))


def sql(query, tables=None, **named):
    """A lazy table from a SQL query over named in-memory tables.

    ``tables`` is a mapping of name to :class:`marrow.RecordBatch`; the same
    thing can be passed as keywords, which reads better for the one-table case
    that most queries are::

        marrow.sql("SELECT k, SUM(v) AS total FROM basic GROUP BY k",
                   basic=batch)

    The result is an ordinary :class:`LazyTable`, so a parsed query composes
    with the built verbs and with ``collect()`` exactly as any other plan does.
    Nothing is executed here — this parses and plans only.
    """
    sources = dict(tables or {})
    sources.update(named)
    if not sources:
        raise ValueError("sql() needs at least one table")
    names = list(sources)
    batches = [unwrap(sources[name]) for name in names]
    return LazyTable.wrap(_ma.sql_plan(str(query), names, batches))
