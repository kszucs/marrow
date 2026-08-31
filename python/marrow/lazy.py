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
from . import RecordBatch, _Wrapper
from ._expr import Column, col

__all__ = ["LazyTable", "memtable", "read_parquet"]


def _unwrap_expr(value):
    """A plan-layer expression argument.

    ``str`` passes straight through: the binding turns a name into ``col(name)``
    itself, so the common case needs no expression object at all.
    """
    if isinstance(value, str):
        return value
    if hasattr(value, "unwrap"):
        return value.unwrap()
    return value


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
        return _unwrap_expr(key), ascending
    return _unwrap_expr(entry), True


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
    """`(names, values)` from either keywords or two parallel lists."""
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
        values = [_unwrap_expr(v) for v in values]
        if len(names) != len(values):
            raise ValueError(f"{verb}: {len(names)} names but {len(values)} values")
        return names, values
    names = list(named)
    return names, [_unwrap_expr(named[n]) for n in names]


class LazyTable(_Wrapper):
    """A lazy relational table: a query plan you can keep composing.

    Wraps the bound ``Plan`` type. Named ``LazyTable`` rather than ``Table``
    because ``marrow.Table`` is the eager, PyArrow-shaped table.
    """

    # -- introspection ----------------------------------------------------

    @property
    def schema(self):
        return self._binding.schema()

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

    def rename(self, mapping):
        """Rename columns, leaving the rest untouched and in place.

        ``t.rename({"v": "value"})`` — polars' spelling. The plan takes two
        parallel lists (old, new) and mentions only the columns that change, so
        the dict is unzipped here rather than expanded to full width."""
        return LazyTable.wrap(
            self._binding.rename(
                [str(k) for k in mapping], [str(v) for v in mapping.values()]
            )
        )

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
        return LazyTable.wrap(self._binding.project(names, values))

    def with_columns(self, *positional, **named):
        """Add or replace computed columns, keeping every other one.

            t.with_columns(total=t["qty"] * t["price"])

        ``project``'s usable half, and the verb polars and ibis lean on
        hardest: a new name is appended, an existing one is replaced **at its
        original position**, and every expression sees the *input* columns
        rather than a partially-updated output. Chain two calls for sequential
        semantics.

        Takes the same two shapes as :meth:`project` — keywords, or two
        parallel lists. The output name is always written, never derived from
        the expression.
        """
        names, values = _projection(positional, named, "with_columns")
        return LazyTable.wrap(self._binding.with_columns(names, values))

    # ibis spells `with_columns` as `mutate`. Both work.
    mutate = with_columns

    def filter(self, predicate):
        """Keep rows where ``predicate`` is true."""
        return LazyTable.wrap(self._binding.filter(_unwrap_expr(predicate)))

    def aggregate(self, by=(), *aggs, **named_aggs):
        """Grouped aggregation.

            t.aggregate(by=["region"], total=("sum", "price"), n=("count", "id"))

        ``by`` is a list of column names or expressions; an empty ``by`` is one
        implicit group (``SELECT sum(x)`` with no ``GROUP BY``). Keyword
        aggregates name their output column; positional ones must carry their
        own ``.alias(...)``.

        Keys with no aggregates is ``SELECT DISTINCT``, which the plan layer
        executes. Neither keys nor aggregates is meaningless and raises.
        """
        if isinstance(by, (str, bytes)) or not hasattr(by, "__iter__"):
            by = [by]
        keys = [_unwrap_expr(k) for k in by]
        specs = [_unwrap_expr(a) for a in aggs]
        specs += [_aggregate_spec(n, v) for n, v in named_aggs.items()]
        if not specs and not keys:
            raise ValueError("aggregate: needs at least one key or aggregate")
        return LazyTable.wrap(self._binding.aggregate(keys, specs))

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

    # PyArrow spells it `sort_by`; ibis spells it `order_by`. Both work, and
    # `sort_by` is also what `DynRelation` calls it.
    sort_by = order_by

    def limit(self, n, offset=0):
        """At most ``n`` rows, after skipping ``offset``."""
        return LazyTable.wrap(self._binding.limit(n, offset))

    def head(self, n=5):
        return self.limit(n, 0)

    def join(self, other, on=None, left_on=None, right_on=None, how="inner"):
        """Equijoin. ``on`` is shorthand for equal key names on both sides.

        Keys are given by **name** here and resolved to column indices against
        each side's schema, because that is what ``Plan.join`` takes: the join
        operator hashes whole columns of the input, so a key is a position in
        the schema rather than an expression to evaluate. A key that names no
        column raises here, where the schema is in hand and the message can say
        which side it looked in.
        """
        if on is not None:
            left_on = right_on = on
        if left_on is None or right_on is None:
            raise ValueError("join: pass `on`, or both `left_on` and `right_on`")
        if isinstance(left_on, (str, bytes)):
            left_on = [left_on]
        if isinstance(right_on, (str, bytes)):
            right_on = [right_on]
        if len(left_on) != len(right_on):
            raise ValueError(
                f"join: {len(left_on)} left keys but {len(right_on)} right keys"
            )
        return LazyTable.wrap(
            self._binding.join(
                other.unwrap(),
                _key_indices(self.column_names, left_on, "left"),
                _key_indices(other.column_names, right_on, "right"),
                how,
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

    def to_pyarrow(self, num_threads=0):
        """Run the plan and hand the result to PyArrow (zero-copy, C Data)."""
        import pyarrow as pa

        return pa.record_batch(self.collect(num_threads))

    def explain(self):
        """The plan as text, without running it.

        Renders recursively — a ``Filter`` names the node it filters — so this
        is the whole tree, not just the root."""
        return self._plan_text()


def _key_indices(column_names, keys, side):
    """Join keys by name -> positions in `column_names`."""
    out = []
    for key in keys:
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
    binding = _ma.parquet_scan(
        str(path),
        schema.unwrap() if hasattr(schema, "unwrap") else schema,
    )
    return LazyTable.wrap(binding)


def memtable(batch):
    """A lazy table over an in-memory :class:`marrow.RecordBatch`.

    ``memtable`` is ibis's name for exactly this, and the pairing is the point:
    the *verb* says which world you are in. Lazy is ``memtable`` /
    ``read_parquet``, eager is ``table`` / ``record_batch`` — each namespace
    spelled consistently with the library it is modelled on.
    """
    return LazyTable.wrap(
        _ma.in_memory_table(batch.unwrap() if hasattr(batch, "unwrap") else batch)
    )
