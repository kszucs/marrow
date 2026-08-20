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
    batch = top.collect() # now it runs

"ibis-flavoured" is a naming guideline only: there is no ``ibis`` dependency and
this is not an ibis backend.

Column expressions come from :mod:`marrow._expr_column`. That module is owned by
the expression bindings; where it is absent this module still works for
everything expressible with column *names* — select, drop, order_by, join keys,
and ``(func, column)`` aggregates — and only ``t[name]`` / ``filter`` need it.
"""

from . import libmarrow as _ma
from . import RecordBatch, _Wrapper

try:  # Owned by the expression bindings; optional until they land.
    from ._expr_column import Aggregate, Column, col

    _HAVE_EXPRESSIONS = True
except ImportError:  # pragma: no cover - exercised only pre-merge
    Aggregate = Column = col = None
    _HAVE_EXPRESSIONS = False


__all__ = ["LazyTable", "memtable", "read_parquet"]


_DEFAULT_MORSEL_SIZE = 8192


def _require_expressions(what):
    if not _HAVE_EXPRESSIONS:
        raise RuntimeError(
            f"{what} needs marrow._expr_column (the expression bindings), "
            "which is not installed in this build. Column names work "
            "everywhere a bare column reference is accepted."
        )


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
        # `t.aggregate(by=["k"], n="count")` — count over the group key.
        raise ValueError(
            f"aggregate: {name}={value!r} is ambiguous; use "
            f'{name}=("{value}", "<column>")'
        )
    _require_expressions("aggregate with expression aggregates")
    unwrapped = value.unwrap()
    return unwrapped.alias(name)


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
        _require_expressions("t[name]")
        return col(name)

    def __repr__(self):
        return f"LazyTable\n{self._plan_text()}"

    def __str__(self):
        return self._plan_text()

    def _plan_text(self):
        """The bound plan's own ``__str__``, called explicitly.

        ``str(binding)`` does *not* reach it: ``def_method`` fills the type's
        ``tp_dict``, not the CPython ``tp_str`` slot, so ``str()`` finds the
        derived ``repr`` and returns ``"<marrow.Plan: Sort(...)>"``. The
        expression bindings documented that hazard; the plan surface was
        printing the bracketed form through it.
        """
        return self._binding.__str__()

    # -- relational verbs -------------------------------------------------

    def select(self, *names):
        """Project columns by name. ``t.select("a", "b")``."""
        if len(names) == 1 and isinstance(names[0], (list, tuple)):
            names = tuple(names[0])
        return LazyTable.wrap(self._binding.select([str(n) for n in names]))

    def drop(self, *names):
        """Every column except these — ``t.drop("a", "b")``.

        A real ``Drop`` lowering, not a ``select`` of the complement computed
        here: the plan node keeps each surviving ``Field`` whole and raises on
        an unknown name itself, so neither the column list nor the error has to
        be reconstructed in Python.
        """
        if len(names) == 1 and isinstance(names[0], (list, tuple)):
            names = tuple(names[0])
        return LazyTable.wrap(self._binding.drop([str(n) for n in names]))

    def rename(self, mapping):
        """Rename columns, leaving the rest untouched and in place.

        ``t.rename({"v": "value"})`` — polars' spelling. The plan takes two
        parallel lists (old, new) and mentions only the columns that change, so
        the dict is unzipped here rather than expanded to full width.
        """
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
        lanes can share, and the golden corpus runs one text through both.
        """
        names, values = _projection(positional, named, "project")
        return LazyTable.wrap(self._binding.project(names, values))

    def with_columns(self, *positional, **named):
        """Add or replace computed columns, keeping every other one.

            t.with_columns(total=t["qty"] * t["price"])

        ``project``'s usable half, and the verb polars and ibis lean on hardest:
        a new name is appended, an existing one is replaced **at its original
        position**, and every expression sees the *input* columns rather than a
        partially-updated output. Chain two calls for sequential semantics.

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
        """
        if isinstance(by, (str, bytes)) or not hasattr(by, "__iter__"):
            by = [by]
        keys = [_unwrap_expr(k) for k in by]
        specs = [_unwrap_expr(a) for a in aggs]
        specs += [_aggregate_spec(n, v) for n, v in named_aggs.items()]
        if not specs:
            raise ValueError("aggregate: needs at least one aggregate")
        return LazyTable.wrap(self._binding.aggregate(keys, specs))

    def order_by(self, *keys, nulls_first=True, stable=True):
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
                stable,
            )
        )

    # PyArrow spells it `sort_by`; ibis spells it `order_by`. Both work.
    sort_by = order_by

    def limit(self, n, offset=0):
        """At most ``n`` rows, after skipping ``offset``.

        Straight after ``order_by`` with ``offset == 0`` this folds into the
        sort's top-K path rather than sorting everything.
        """
        return LazyTable.wrap(self._binding.limit(n, offset))

    def head(self, n=5):
        return self.limit(n, 0)

    def join(
        self,
        other,
        on=None,
        left_on=None,
        right_on=None,
        how="inner",
        strictness="all",
    ):
        """Equijoin. ``on`` is shorthand for equal key names on both sides."""
        if on is not None:
            left_on = right_on = on
        if left_on is None or right_on is None:
            raise ValueError("join: pass `on`, or both `left_on` and `right_on`")
        if isinstance(left_on, (str, bytes)):
            left_on = [left_on]
        if isinstance(right_on, (str, bytes)):
            right_on = [right_on]
        return LazyTable.wrap(
            self._binding.join(
                other.unwrap(),
                [_unwrap_expr(k) for k in left_on],
                [_unwrap_expr(k) for k in right_on],
                how,
                strictness,
            )
        )

    # -- execution --------------------------------------------------------

    def collect(self, num_threads=0):
        """Run the plan and return one eager :class:`marrow.RecordBatch`.

        The whole plan is drained, so a multi-row-group Parquet scan comes back
        complete rather than one row group at a time.

        ``num_threads`` is the CPU worker budget, spelled and defaulted exactly
        as on the eager surface (``RecordBatch.group_by(..., num_threads=0)``,
        ``RecordBatch.join(..., num_threads=0)``):

        * ``0`` — **auto** (the default): each kernel picks serial vs all-cores
          from its own row-count threshold, so a small query pays no worker
          setup and a large one uses the machine.
        * ``1`` — serial, forced.
        * ``N >= 2`` — exactly ``N`` workers, forced, threshold bypassed.

        It lives here rather than on the constructor or on a module-level
        default because ``collect`` is the only place a plan actually *runs*: a
        ``LazyTable`` is an immutable plan that every verb returns a fresh copy
        of, and a stored worker count would have to survive ``join``, where two
        tables with different settings have no defensible winner.
        """
        return RecordBatch.wrap(self._binding.execute(num_threads))

    def to_pyarrow(self, num_threads=0):
        """Run the plan and hand the result to PyArrow (zero-copy, C Data)."""
        import pyarrow as pa

        return pa.record_batch(self.collect(num_threads))

    def explain(self):
        """The plan as text, without running it.

        One line: no `Relation` node renders its children, so this names the
        root operator only. See ``docs/alpha-findings/b2-plan-bindings.md``.
        """
        return self._plan_text()


# ── Entry points ───────────────────────────────────────────────────────────


def read_parquet(path, schema=None, morsel_size=_DEFAULT_MORSEL_SIZE):
    """A lazy table over a Parquet file.

    The schema doubles as the projection — only its columns are read — and is
    inferred from the file's footer when omitted (metadata only, no column
    data).
    """
    binding = _ma.parquet_scan(
        str(path),
        schema.unwrap() if hasattr(schema, "unwrap") else schema,
        morsel_size,
    )
    return LazyTable.wrap(binding)


def memtable(batch, morsel_size=_DEFAULT_MORSEL_SIZE):
    """A lazy table over an in-memory :class:`marrow.RecordBatch`.

    ``memtable`` is ibis's name for exactly this, and the pairing is the point:
    the *verb* says which world you are in. Lazy is ``memtable`` /
    ``read_parquet``, eager is ``table`` / ``record_batch`` -- each namespace
    spelled consistently with the library it is modelled on, rather than
    ``table`` and ``scan`` differing by a name nobody can rank.
    """
    return LazyTable.wrap(
        _ma.in_memory_table(
            batch.unwrap() if hasattr(batch, "unwrap") else batch, morsel_size
        )
    )
