"""The Python expression surface — ``Column`` and ``Aggregate``.

``marrow.libmarrow`` exposes two binding types from the runtime expression
lane: ``Expr`` (Mojo ``RuntimeValue``) and ``Agg`` (Mojo ``RuntimeAggregate``).
Both are deliberately spartan — named methods only, one required argument each,
no coercion. This module is the other half: the composition wrappers that own
the operator dunders, scalar coercion, keyword arguments and ``__repr__``.

**The method names are the Mojo typed lane's, verb for verb.** ``col("a",
int64) * lit(2, int64)`` in Mojo and ``col("a") * lit(2)`` here build the same
query; what differs is which lane resolves the dtype, not what the expression
is called. Where the two could diverge they do not: ``/`` is float64 in both
(``FloatBinary``'s rule), ``is_valid`` is Arrow's spelling in both, and
``count()`` counts non-null values in both while ``count_star()`` counts rows.

Two things the binding layer cannot do, and therefore does not:

- **Operators.** ``PythonTypeBuilder.bind`` installs four CPython slots —
  ``tp_new``, ``tp_init``, ``tp_dealloc``, ``tp_repr`` — and ``def_method``
  fills the type's ``tp_dict``, not a slot. Measured on this build:
  ``e.__str__()`` returns ``"gt(a, 1)"`` while ``str(e)`` returns the
  ``tp_repr`` output ``"<marrow.Expr: gt(a, 1)>"``. So an ``__add__``
  registered there would never fire for ``+`` either. Operators must live in
  Python.
- **``__eq__`` returning a non-bool.** ``col("a") == 1`` is a predicate, not a
  test. ``Column`` therefore has no usable ``__hash__`` (Python drops it when
  ``__eq__`` is defined), which matches ``pyarrow.compute.Expression``.

Usage::

    from marrow import col, lit

    predicate = (col("a") > 10) & col("s").startswith("x")
    total = col("amount").sum().alias("total")
"""

from . import Array, RecordBatch, _Wrapper, array
from . import libmarrow as _ma

__all__ = [
    "Aggregate",
    "Column",
    "case_when",
    "coalesce",
    "col",
    "count_star",
    "if_else",
    "lit",
]


def _expr(value):
    """The ``Expr`` binding for `value`, coercing a Python scalar to a literal.

    Anything that is not already a ``Column`` becomes a one-element literal —
    the same rule ``lit()`` applies, so ``col("a") + 1`` and
    ``col("a") + lit(1)`` build the same tree."""
    if isinstance(value, Column):
        return value._binding
    return lit(value)._binding


class Column(_Wrapper):
    """An expression over a named column — the Python face of ``RuntimeValue``.

    Built with :func:`col` and :func:`lit`, combined with operators, and
    consumed either eagerly through :meth:`execute` or by the relational plan
    layer, which calls :meth:`unwrap` to get the ``Expr`` binding back."""

    # ── representation ──────────────────────────────────────────────────────

    def render(self):
        """This expression as a string — ``"gt(a, 1)"``."""
        return self._binding.render()

    def name(self):
        """The referenced column's name, or ``""`` for anything else."""
        return self._binding.name()

    def referenced_columns(self):
        """Every column name this expression reads, in first-seen order."""
        return self._binding.referenced_columns()

    def __str__(self):
        return self.render()

    def __repr__(self):
        return f"<marrow.Column: {self.render()}>"

    # ── evaluation ──────────────────────────────────────────────────────────

    def execute(self, batch):
        """Evaluate against one :class:`~marrow.RecordBatch`, eagerly."""
        binding = batch._binding if isinstance(batch, RecordBatch) else batch
        return Array.wrap(self._binding.execute(binding))

    # ── arithmetic ──────────────────────────────────────────────────────────

    def __add__(self, other):
        return Column.wrap(self._binding.add(_expr(other)))

    def __radd__(self, other):
        return Column.wrap(_expr(other).add(self._binding))

    def __sub__(self, other):
        return Column.wrap(self._binding.sub(_expr(other)))

    def __rsub__(self, other):
        return Column.wrap(_expr(other).sub(self._binding))

    def __mul__(self, other):
        return Column.wrap(self._binding.mul(_expr(other)))

    def __rmul__(self, other):
        return Column.wrap(_expr(other).mul(self._binding))

    def __truediv__(self, other):
        return Column.wrap(self._binding.truediv(_expr(other)))

    def __rtruediv__(self, other):
        return Column.wrap(_expr(other).truediv(self._binding))

    def __floordiv__(self, other):
        return Column.wrap(self._binding.floordiv(_expr(other)))

    def __rfloordiv__(self, other):
        return Column.wrap(_expr(other).floordiv(self._binding))

    def __mod__(self, other):
        return Column.wrap(self._binding.mod(_expr(other)))

    def __rmod__(self, other):
        return Column.wrap(_expr(other).mod(self._binding))

    def __pow__(self, other):
        return Column.wrap(self._binding.pow(_expr(other)))

    def __rpow__(self, other):
        return Column.wrap(_expr(other).pow(self._binding))

    def __neg__(self):
        return Column.wrap(self._binding.neg())

    # ── comparison ──────────────────────────────────────────────────────────

    def __lt__(self, other):
        return Column.wrap(self._binding.lt(_expr(other)))

    def __le__(self, other):
        return Column.wrap(self._binding.le(_expr(other)))

    def __gt__(self, other):
        return Column.wrap(self._binding.gt(_expr(other)))

    def __ge__(self, other):
        return Column.wrap(self._binding.ge(_expr(other)))

    def __eq__(self, other):
        return Column.wrap(self._binding.eq(_expr(other)))

    def __ne__(self, other):
        return Column.wrap(self._binding.ne(_expr(other)))

    # An expression compares to a predicate, not to a bool, so it cannot also
    # be a dict key. Same trade-off as `pyarrow.compute.Expression`.
    __hash__ = None

    # ── boolean ─────────────────────────────────────────────────────────────

    def __and__(self, other):
        return Column.wrap(self._binding.and_(_expr(other)))

    def __rand__(self, other):
        return Column.wrap(_expr(other).and_(self._binding))

    def __or__(self, other):
        return Column.wrap(self._binding.or_(_expr(other)))

    def __ror__(self, other):
        return Column.wrap(_expr(other).or_(self._binding))

    def __xor__(self, other):
        return Column.wrap(self._binding.xor(_expr(other)))

    def __rxor__(self, other):
        return Column.wrap(_expr(other).xor(self._binding))

    def __invert__(self):
        return Column.wrap(self._binding.invert())

    # ── math ────────────────────────────────────────────────────────────────

    def abs(self):
        return Column.wrap(self._binding.abs())

    def __abs__(self):
        return self.abs()

    def sign(self):
        return Column.wrap(self._binding.sign())

    def floor(self):
        return Column.wrap(self._binding.floor())

    def __floor__(self):
        return self.floor()

    def ceil(self):
        return Column.wrap(self._binding.ceil())

    def __ceil__(self):
        return self.ceil()

    def round(self):
        return Column.wrap(self._binding.round())

    def __round__(self, ndigits=None):
        if ndigits is not None:
            raise NotImplementedError(
                "round(expr, ndigits) is not supported; only round(expr)"
            )
        return self.round()

    def trunc(self):
        return Column.wrap(self._binding.trunc())

    def __trunc__(self):
        return self.trunc()

    def sqrt(self):
        """The square root, as ``float64`` whatever the input type."""
        return Column.wrap(self._binding.sqrt())

    def exp(self):
        return Column.wrap(self._binding.exp())

    def ln(self):
        return Column.wrap(self._binding.ln())

    # ── string ──────────────────────────────────────────────────────────────

    def upper(self):
        return Column.wrap(self._binding.upper())

    def lower(self):
        return Column.wrap(self._binding.lower())

    def strip(self):
        return Column.wrap(self._binding.strip())

    def lstrip(self):
        return Column.wrap(self._binding.lstrip())

    def rstrip(self):
        return Column.wrap(self._binding.rstrip())

    def reverse(self):
        return Column.wrap(self._binding.reverse())

    def capitalize(self):
        return Column.wrap(self._binding.capitalize())

    def length(self):
        """Byte length per element, as ``int32`` — pyarrow's ``utf8_length``."""
        return Column.wrap(self._binding.length())

    def startswith(self, prefix):
        return Column.wrap(self._binding.startswith(_expr(prefix)))

    def endswith(self, suffix):
        return Column.wrap(self._binding.endswith(_expr(suffix)))

    def contains(self, substring):
        return Column.wrap(self._binding.contains(_expr(substring)))

    def like(self, pattern):
        """SQL ``LIKE`` — ``%`` and ``_`` wildcards, case-sensitive."""
        return Column.wrap(self._binding.like(pattern))

    def ilike(self, pattern):
        """SQL ``ILIKE`` — as :meth:`like`, case-insensitive."""
        return Column.wrap(self._binding.ilike(pattern))

    # ── temporal ────────────────────────────────────────────────────────────

    def year(self):
        return Column.wrap(self._binding.year())

    def month(self):
        return Column.wrap(self._binding.month())

    def day(self):
        return Column.wrap(self._binding.day())

    def hour(self):
        return Column.wrap(self._binding.hour())

    def minute(self):
        return Column.wrap(self._binding.minute())

    def second(self):
        return Column.wrap(self._binding.second())

    def day_of_week(self):
        """The day of the week, Monday = 0."""
        return Column.wrap(self._binding.day_of_week())

    def quarter(self):
        return Column.wrap(self._binding.quarter())

    def day_of_year(self):
        return Column.wrap(self._binding.day_of_year())

    def date_trunc(self, unit):
        """Truncate to ``unit`` — ``"second"``, ``"minute"``, ``"hour"``,
        ``"day"``, ``"month"``, ``"quarter"``, ``"year"``.

        The unit is validated when the expression is built, not on the first
        row that evaluates it."""
        return Column.wrap(self._binding.date_trunc(unit))

    # ── conditional, membership, casting, nested ────────────────────────────

    def coalesce(self, *others):
        """This expression where it is valid, the next one where it is null."""
        return coalesce(self, *others)

    def nullif(self, other):
        """Null wherever this equals `other`, otherwise unchanged."""
        return Column.wrap(self._binding.nullif(_expr(other)))

    def fill_null(self, other):
        """`other` wherever this is null, this expression elsewhere.

        The same result as ``coalesce(other)`` for two operands; both are bound
        because the Mojo lane has both and they carry different kernels."""
        return Column.wrap(self._binding.fill_null(_expr(other)))

    def isin(self, values):
        """Membership against a value set — a list or a :class:`~marrow.Array`.

        The set is hashed once per batch rather than per row, which is why it
        is a value set and not an expression: ``col("a").isin(col("b"))`` is a
        different question and is not this one."""
        if isinstance(values, Array):
            value_set = values._binding
        elif isinstance(values, _ma.Array):
            value_set = values
        else:
            value_set = array(list(values))._binding
        return Column.wrap(self._binding.isin(value_set))

    def cast(self, target_type, *, safe=True):
        """Cast to `target_type`, a :class:`~marrow.DataType`.

        With ``safe=True`` (the default) a lossy conversion raises; with
        ``safe=False`` the raw truncating/wrapping conversion is used, except
        for string parsing, which nulls the unparseable value. Same flag,
        default and meaning as :func:`marrow.compute.cast`, which casts an
        array rather than an expression."""
        return Column.wrap(self._binding.cast(target_type, safe))

    def array_length(self):
        """The number of elements in each list, as ``int32``.

        The only verb that reads a list column: a list element is a whole
        sub-array rather than a value an expression can hold, so a list is
        consumed into a numeric column or not read at all."""
        return Column.wrap(self._binding.array_length())

    # ── null handling ───────────────────────────────────────────────────────

    def is_null(self):
        """True where this expression is null. Never null itself."""
        return Column.wrap(self._binding.is_null())

    def is_valid(self):
        """True where this expression is *not* null — ``~is_null()``.

        Spelled as PyArrow spells it. Polars calls it ``is_not_null``; that
        name is not aliased here, because the Arrow spelling is the one the
        rest of this package uses (``Array.is_valid``)."""
        return Column.wrap(self._binding.is_valid())

    def is_nan(self):
        """True where this floating-point expression is NaN.

        Null in, null out — a null is *not* a NaN, which is the difference from
        :meth:`is_null`."""
        return Column.wrap(self._binding.is_nan())

    def is_inf(self):
        """True where this floating-point expression is +/-infinity."""
        return Column.wrap(self._binding.is_inf())

    # ── aggregation ─────────────────────────────────────────────────────────

    def aggregate(self, function, *, alias=None):
        """Aggregate by name — ``"sum"``, ``"mean"``, ``"count"``, …"""
        agg = Aggregate.wrap(self._binding.aggregate(function))
        return agg.alias(alias) if alias is not None else agg

    def sum(self, *, alias=None):
        return self._reduce("sum", alias)

    def mean(self, *, alias=None):
        """``AVG(x)``. Spelled ``mean`` as Arrow spells it."""
        return self._reduce("mean", alias)

    def product(self, *, alias=None):
        return self._reduce("product", alias)

    def min(self, *, alias=None):
        return self._reduce("min", alias)

    def max(self, *, alias=None):
        return self._reduce("max", alias)

    def count(self, *, alias=None):
        """``COUNT(x)`` — the *non-null* values of ``x``.

        Not the same as :func:`count_star`, which counts rows; the two differ
        on any nullable column."""
        return self._reduce("count", alias)

    def count_distinct(self, *, alias=None):
        """``COUNT(DISTINCT x)`` — exact, and skipping nulls like ``count``."""
        return self._reduce("count_distinct", alias)

    def approx_count_distinct(self, *, alias=None):
        """``COUNT(DISTINCT x)`` from a sketch, for when exact is too dear."""
        return self._reduce("approx_count_distinct", alias)

    def variance(self, *, alias=None):
        """``VAR_POP(x)`` — the population variance, Arrow's default."""
        return self._reduce("variance", alias)

    def var_samp(self, *, alias=None):
        """``VAR_SAMP(x)`` — the sample variance, ``ddof=1``."""
        return self._reduce("var_samp", alias)

    def stddev(self, *, alias=None):
        """``STDDEV_POP(x)``."""
        return self._reduce("stddev", alias)

    def stddev_samp(self, *, alias=None):
        """``STDDEV_SAMP(x)`` — the square root of :meth:`var_samp`."""
        return self._reduce("stddev_samp", alias)

    def _reduce(self, verb, alias):
        """One aggregate, by the binding method of the same name.

        The verbs go through the dedicated binding methods rather than through
        ``aggregate(name)`` so an unknown one is a Python ``AttributeError``
        here instead of a Mojo raise on a string, and so the two lanes' verb
        lists are the same list."""
        agg = Aggregate.wrap(getattr(self._binding, verb)())
        return agg.alias(alias) if alias is not None else agg


class Aggregate(_Wrapper):
    """An aggregate over an expression — the Python face of
    ``RuntimeAggregate``.

    Produced by :meth:`Column.sum` and friends, consumed by
    ``LazyTable.aggregate``, which calls :meth:`unwrap`."""

    def alias(self, name):
        """Name this aggregate's output column."""
        return Aggregate.wrap(self._binding.alias(name))

    def name(self):
        """The output column name — the alias if set, else the function."""
        return self._binding.name()

    def referenced_columns(self):
        """Every column name this aggregate reads."""
        return self._binding.referenced_columns()

    def render(self):
        """This aggregate as a string — ``"sum(a) AS total"``."""
        return self._binding.render()

    def __str__(self):
        return self.render()

    def __repr__(self):
        return f"<marrow.Aggregate: {self.render()}>"


# ── constructors ───────────────────────────────────────────────────────────


def col(name, dtype=None):
    """Reference a column by name — ``col("amount")``.

    The dtype is resolved against the batch, not here; that is what makes this
    the *runtime* lane. ``dtype`` is accepted and ignored so that
    ``col("amount", int64)`` — the spelling the Mojo comptime lane *requires*,
    since a fused AOT leaf fixes its type at compile time — is one expression
    both lanes run."""
    return Column.wrap(_ma.expr_column(name))


def lit(value, type=None):
    """A constant — ``lit(10)``, ``lit("x")``, ``lit(2.5, float32)``.

    The value goes through :func:`marrow.array` as a one-element array, so
    type inference and an explicit `type` behave exactly as they do there."""
    if isinstance(value, Column):
        return value
    return Column.wrap(_ma.expr_literal(array([value], type)._binding))


def count_star(*, alias=None):
    """``COUNT(*)`` — how many rows, not how many non-null values.

    A free function rather than a ``Column`` method, because it is the one
    aggregate with no input column::

        t.aggregate(by=["region"], n=marrow.count_star())

    ``col("x").count()`` is the other thing SQL spells ``COUNT(x)``: it counts
    the *valid* values of ``x``, so the two disagree on any nullable column.
    Keyword aggregates rename the result to their keyword, so ``alias`` is only
    needed positionally."""
    agg = Aggregate.wrap(_ma.expr_count_star())
    return agg.alias(alias) if alias is not None else agg


def if_else(condition, if_true, if_false):
    """Element-wise conditional — ``if_else(col("a") > 0, col("a"), lit(0))``.

    A null condition counts as **false** rather than producing a null, which is
    Arrow's ``ExecArrayCaseWhen`` rule and PyArrow's ``pc.case_when``. A
    selected value that is itself null does stay null."""
    return Column.wrap(
        _ma.expr_if_else(_expr(condition), _expr(if_true), _expr(if_false))
    )


def coalesce(*values):
    """First non-null across N expressions — PyArrow's ``pc.coalesce``.

    N-ary rather than a fold of binary nodes, because the kernel is n-ary:
    folding would materialise one intermediate column per extra operand."""
    if not values:
        raise ValueError("coalesce: needs at least one value")
    return Column.wrap(_ma.expr_coalesce([_expr(v) for v in values]))


def case_when(*pairs, else_=None):
    """Multi-branch ``CASE WHEN`` — ``case_when((c1, v1), (c2, v2), else_=d)``.

    The first ``v`` whose ``c`` is **valid and true**; a null condition counts
    as false. With no ``else_``, an unmatched row is null."""
    if not pairs:
        raise ValueError("case_when: needs at least one (condition, value)")
    conditions, values = [], []
    for pair in pairs:
        condition, value = pair
        conditions.append(_expr(condition))
        values.append(_expr(value))
    return Column.wrap(
        _ma.expr_case_when(conditions, values, None if else_ is None else _expr(else_))
    )
