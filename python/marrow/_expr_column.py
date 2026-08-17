"""The Python expression surface — ``Column`` and ``Aggregate``.

``marrow.libmarrow`` exposes two binding types from the runtime expression
lane: ``Expr`` (Mojo ``DynValue``) and ``Agg`` (Mojo ``DynAgg``). Both are
deliberately spartan — named methods only, one required argument each, no
coercion. This module is the other half: the composition wrappers that own the
operator dunders, scalar coercion, keyword arguments and ``__repr__``.

Two things the binding layer cannot do, and therefore does not:

- **Operators.** ``def_method`` puts a name in the type's ``tp_dict``; it does
  not fill the corresponding CPython slot. Measured on this build:
  ``e.__str__()`` returns ``"a"`` while ``str(e)`` returns the derived
  ``"<marrow.Expr: a>"`` — so ``__add__`` / ``__eq__`` registered there would
  never fire for ``+`` / ``==`` either. Operators must live in Python.
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

__all__ = ["Aggregate", "Column", "col", "if_else", "lit"]


def _expr(value):
    """The ``Expr`` binding for `value`, coercing a Python scalar to a literal.

    Anything that is not already a ``Column`` becomes a one-element literal —
    the same rule ``lit()`` applies, so ``col("a") + 1`` and
    ``col("a") + lit(1)`` build the same tree."""
    if isinstance(value, Column):
        return value._binding
    return lit(value)._binding


class Column(_Wrapper):
    """An expression over a named column — the Python face of ``DynValue``.

    Built with :func:`col` and :func:`lit`, combined with operators, and
    consumed either eagerly through :meth:`execute` or by the relational plan
    layer, which calls :meth:`unwrap` to get the ``Expr`` binding back."""

    # ── representation ──────────────────────────────────────────────────────

    def render(self):
        """This expression as a string — ``"add(a, literal)"``."""
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

    def sqrt(self):
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
        return Column.wrap(self._binding.day_of_week())

    def quarter(self):
        return Column.wrap(self._binding.quarter())

    def day_of_year(self):
        return Column.wrap(self._binding.day_of_year())

    def date_trunc(self, unit):
        """Truncate to ``unit`` — ``"year"``, ``"month"``, ``"day"``, …"""
        return Column.wrap(self._binding.date_trunc(unit))

    # ── conditional, membership, casting ────────────────────────────────────

    def coalesce(self, other):
        """This expression where it is valid, `other` where it is null."""
        return Column.wrap(self._binding.coalesce(_expr(other)))

    def nullif(self, other):
        """Null wherever this equals `other`, otherwise unchanged."""
        return Column.wrap(self._binding.nullif(_expr(other)))

    def isin(self, values):
        """Membership against a value set — a list or a :class:`~marrow.Array`."""
        if isinstance(values, Array):
            value_set = values._binding
        elif isinstance(values, _ma.Array):
            value_set = values
        else:
            value_set = array(list(values))._binding
        return Column.wrap(self._binding.isin(value_set))

    def cast(self, target_type):
        """Cast to `target_type`, a :class:`~marrow.DataType`."""
        return Column.wrap(self._binding.cast(target_type))

    # TODO(alpha): is_null / is_valid / is_nan / fill_null land here once
    # `DynValue` grows them; the binding has the matching TODO.

    # ── aggregation ─────────────────────────────────────────────────────────

    def aggregate(self, function, *, alias=None):
        """Aggregate by name — ``"sum"``, ``"mean"``, ``"count"``, …"""
        agg = Aggregate.wrap(self._binding.aggregate(function))
        return agg.alias(alias) if alias is not None else agg

    def sum(self, *, alias=None):
        return self.aggregate("sum", alias=alias)

    def mean(self, *, alias=None):
        return self.aggregate("mean", alias=alias)

    def product(self, *, alias=None):
        return self.aggregate("product", alias=alias)

    def min(self, *, alias=None):
        return self.aggregate("min", alias=alias)

    def max(self, *, alias=None):
        return self.aggregate("max", alias=alias)

    def count(self, *, alias=None):
        return self.aggregate("count", alias=alias)


class Aggregate(_Wrapper):
    """An aggregate over an expression — the Python face of ``DynAgg``.

    Produced by :meth:`Column.sum` and friends, consumed by the plan layer's
    ``group_by(...).aggregate([...])``, which calls :meth:`unwrap`."""

    def alias(self, name):
        """Name this aggregate's output column."""
        return Aggregate.wrap(self._binding.alias(name))

    def function(self):
        """The aggregate function's name."""
        return self._binding.function()

    def name(self):
        """The output column name — the alias if set, else the function."""
        return self._binding.name()

    def input(self):
        """The expression being aggregated."""
        return Column.wrap(self._binding.input())

    def render(self):
        """This aggregate as a string — ``"sum(a) as total"``."""
        return self._binding.render()

    def __str__(self):
        return self.render()

    def __repr__(self):
        return f"<marrow.Aggregate: {self.render()}>"


# ── constructors ───────────────────────────────────────────────────────────


def col(name):
    """Reference a column by name — ``col("amount")``.

    The dtype is resolved against the batch, not here; that is what makes this
    the *runtime* lane. The Mojo-side ``col(name, dtype)`` builds the fused AOT
    node instead, and has no Python equivalent."""
    return Column.wrap(_ma.expr_column(name))


def lit(value, type=None):
    """A constant — ``lit(10)``, ``lit("x")``, ``lit(2.5, float32)``.

    The value goes through :func:`marrow.array` as a one-element array, so
    type inference and an explicit `type` behave exactly as they do there."""
    if isinstance(value, Column):
        return value
    return Column.wrap(_ma.expr_literal(array([value], type)._binding))


def if_else(condition, if_true, if_false):
    """Element-wise conditional — ``if_else(col("a") > 0, col("a"), lit(0))``."""
    return Column.wrap(
        _ma.expr_if_else(_expr(condition), _expr(if_true), _expr(if_false))
    )
