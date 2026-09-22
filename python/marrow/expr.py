"""The Python expression surface — ``Column`` and ``Aggregate``.

``marrow.libmarrow`` exposes two binding types from the runtime expression
lane: ``Expr`` (Mojo ``RuntimeValue``) and ``Agg`` (Mojo ``RuntimeAggregate``).
Both are deliberately spartan. This module is the other half: the composition
wrappers that own the operator dunders, scalar coercion, keyword arguments and
``__repr__``.

**The method names are the Mojo typed lane's, verb for verb.** ``col("a",
int64) * lit(2, int64)`` in Mojo and ``col("a") * lit(2)`` here build the same
query; what differs is which lane resolves the dtype, not what the expression
is called. Where the two could diverge they do not: ``/`` is float64 in both
(``FloatBinary``'s rule), ``is_valid`` is Arrow's spelling in both, and
``count()`` counts non-null values in both while ``count_star()`` counts rows.

**The verbs are generated, not written out.** ``marrow/expr/runtime/values.mojo``
owns the one list of what the interpreter answers to, ``expr_verbs()`` hands it
over as ``{verb: arity}``, and :func:`_install_verbs` turns each into a method
of the same name. Writing them out by hand is what let 27 of them — the whole
SQL string surface, half the temporal one — exist in Mojo and be unreachable
from Python: the list was in two places and only one was maintained. A verb
added to the Mojo lane now arrives here with no edit to this file.

Anything needing real logic is still written by hand below, and those are
exactly the verbs a table cannot describe: the operators (which coerce a bare
Python scalar), ``cast``/``isin``/``like``/``ilike``/``date_trunc`` (typed
payloads), ``coalesce``/``case_when`` (n-ary), and the boolean connectives
(which constant-fold in Mojo).

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

from . import libmarrow as _ma
from ._wrapper import _Wrapper, unwrap
from .arrays import Array, _as_binding_array, array

__all__ = [
    "Aggregate",
    "Column",
    "Window",
    "array_contains",
    "array_length",
    "case_when",
    "coalesce",
    "col",
    "count_star",
    "if_else",
    "is_in",
    "lit",
    "cume_dist",
    "dense_rank",
    "maximum",
    "minimum",
    "ntile",
    "percent_rank",
    "rank",
    "row_number",
]


def _expr(value):
    """The ``Expr`` binding for `value`, coercing a Python scalar to a literal.

    Anything that is not already a ``Column`` becomes a one-element literal —
    the same rule ``lit()`` applies, so ``col("a") + 1`` and
    ``col("a") + lit(1)`` build the same tree."""
    if isinstance(value, Column):
        return value._binding
    return lit(value)._binding


def _pattern(value):
    """A constant pattern: a plain string, or a literal expression.

    A non-literal `Column` is rejected rather than stringified — `col("p")`
    would otherwise silently become the pattern ``"p"``, matching the column's
    *name* instead of reading its values."""
    if isinstance(value, str):
        return value
    if isinstance(value, Column):
        if value._binding.tag() != "literal":
            raise TypeError(
                f"like/ilike: the pattern must be constant, got {value.render()!r}"
            )
        return value.name()
    raise TypeError(f"like/ilike: expected a string or a literal, got {value!r}")


def _call(verb, *operands):
    """One node, by verb name — the single path to ``expr_call``."""
    return Column.wrap(_ma.expr_call(verb, [_expr(o) for o in operands]))


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
        binding = unwrap(batch)
        return Array.wrap(self._binding.execute(binding))

    # ── arithmetic ──────────────────────────────────────────────────────────

    def __add__(self, other):
        return _call("add", self, other)

    def __radd__(self, other):
        return _call("add", other, self)

    def __sub__(self, other):
        return _call("sub", self, other)

    def __rsub__(self, other):
        return _call("sub", other, self)

    def __mul__(self, other):
        return _call("mul", self, other)

    def __rmul__(self, other):
        return _call("mul", other, self)

    def __truediv__(self, other):
        """``l / r`` — Python's true division, not ``marrow.compute.divide``.

        Both operands widen to ``float64`` before the kernel, so ``-1 / 3`` is
        -0.333 and a zero divisor answers an infinity or a NaN. The like-named
        compute function follows ``pyarrow.compute.divide`` instead: integer
        in, integer out, and a zero divisor raises.
        """
        return _call("truediv", self, other)

    def __rtruediv__(self, other):
        return _call("truediv", other, self)

    def __floordiv__(self, other):
        return _call("floordiv", self, other)

    def __rfloordiv__(self, other):
        return _call("floordiv", other, self)

    def __mod__(self, other):
        return _call("mod", self, other)

    def __rmod__(self, other):
        return _call("mod", other, self)

    def __pow__(self, other):
        return _call("pow", self, other)

    def __rpow__(self, other):
        return _call("pow", other, self)

    def __neg__(self):
        return _call("neg", self)

    # ── comparison ──────────────────────────────────────────────────────────

    def __lt__(self, other):
        return _call("lt", self, other)

    def __le__(self, other):
        return _call("le", self, other)

    def __gt__(self, other):
        return _call("gt", self, other)

    def __ge__(self, other):
        return _call("ge", self, other)

    def __eq__(self, other):
        return _call("eq", self, other)

    def __ne__(self, other):
        return _call("ne", self, other)

    # ── boolean ─────────────────────────────────────────────────────────────
    #
    # `and`, `or` and `not` are the three verbs `expr_call` will not build:
    # they constant-fold in Mojo at construction, and `PropagateEmpty` depends
    # on that fold, so they keep their own entry points.

    def __and__(self, other):
        return Column.wrap(_ma.expr_and(self._binding, _expr(other)))

    def __rand__(self, other):
        return Column.wrap(_ma.expr_and(_expr(other), self._binding))

    def __or__(self, other):
        return Column.wrap(_ma.expr_or(self._binding, _expr(other)))

    def __ror__(self, other):
        return Column.wrap(_ma.expr_or(_expr(other), self._binding))

    def __xor__(self, other):
        return _call("xor", self, other)

    def __rxor__(self, other):
        return _call("xor", other, self)

    def __invert__(self):
        return Column.wrap(_ma.expr_not(self._binding))

    # ── math with a Python protocol ─────────────────────────────────────────
    #
    # `abs`, `floor`, `ceil`, `round` and `trunc` are generated like every
    # other verb; these are the dunders that make the builtins reach them.

    def __abs__(self):
        return self.abs()

    def __floor__(self):
        return self.floor()

    def __ceil__(self):
        return self.ceil()

    def __round__(self, ndigits=None):
        if ndigits is not None:
            raise NotImplementedError(
                "round(expr, ndigits) is not supported; only round(expr)"
            )
        return self.round()

    def __trunc__(self):
        return self.trunc()

    # ── verbs carrying a typed payload ──────────────────────────────────────

    def like(self, pattern):
        """SQL ``LIKE`` — ``%`` matches any run, ``_`` any single character.

        The pattern is compiled once per batch rather than read per row, so it
        must be constant: a plain string, or a ``lit(...)`` carrying one. The
        Mojo lane's `like` takes a value node, so accepting a literal is what
        lets `x.like(lit("h%"))` be one expression in both lanes."""
        return Column.wrap(_ma.expr_like(self._binding, _pattern(pattern)))

    def ilike(self, pattern):
        """:meth:`like`, case-insensitively."""
        return Column.wrap(_ma.expr_ilike(self._binding, _pattern(pattern)))

    def date_trunc(self, unit):
        """Truncate to ``unit`` — ``"second"``, ``"minute"``, ``"hour"``,
        ``"day"``, ``"month"``, ``"quarter"``, ``"year"``.

        The unit is validated when the expression is built, not on the first
        row that evaluates it."""
        return Column.wrap(_ma.expr_date_trunc(self._binding, unit))

    def cast(self, target_type, *, safe=True):
        """Cast to `target_type`, a :class:`~marrow.DataType`.

        With ``safe=True`` (the default) a lossy conversion raises; with
        ``safe=False`` the raw truncating/wrapping conversion is used, except
        for string parsing, which nulls the unparseable value. Same flag,
        default and meaning as :func:`marrow.compute.cast`, which casts an
        array rather than an expression."""
        return Column.wrap(_ma.expr_cast(self._binding, unwrap(target_type), safe))

    def isin(self, values):
        """Membership against a value set — a list or a :class:`~marrow.Array`.

        The set is hashed once per batch rather than per row, which is why it
        is a value set and not an expression: ``col("a").isin(col("b"))`` is a
        different question and is not this one."""
        return Column.wrap(_ma.expr_isin(self._binding, _as_binding_array(values)))

    # ── n-ary conditionals ──────────────────────────────────────────────────

    def coalesce(self, *others):
        """This expression where it is valid, the next one where it is null."""
        return coalesce(self, *others)

    # ── window functions over this column ───────────────────────────────────

    def lag(self, offset=1):
        """``LAG(x, offset)`` — this column read `offset` rows earlier."""
        return Window.wrap(self._binding.lag(offset))

    def lead(self, offset=1):
        """``LEAD(x, offset)`` — this column read `offset` rows later."""
        return Window.wrap(self._binding.lead(offset))

    def first_value(self):
        """``FIRST_VALUE(x)`` — this column at the frame's first row."""
        return Window.wrap(self._binding.first_value())

    def last_value(self):
        """``LAST_VALUE(x)`` — this column at the frame's last row.

        Under the default frame that is the *current* row, not the partition's
        last: the frame ends at the current row. Everyone expects otherwise."""
        return Window.wrap(self._binding.last_value())

    def nth_value(self, n):
        """``NTH_VALUE(x, n)`` — this column at the frame's `n`-th row,
        1-based."""
        return Window.wrap(self._binding.nth_value(n))

    def over(self, *args, **kwargs):
        """Always an error — only an aggregate can be windowed.

        Present so the mistake reports what to do instead. `Value.over` raises
        the same way in Mojo: a per-row value has nothing to do with a frame,
        and a silent column of copies is a worse answer than a diagnostic."""
        raise TypeError(
            f"over: {self.render()!r} is a per-row value, not an aggregate; "
            f"aggregate first, then window the result — e.g. "
            f"col(...).sum().over(...)"
        )

    # ── aggregation ─────────────────────────────────────────────────────────

    def aggregate(self, function, *, alias=None):
        """Aggregate by name — ``"sum"``, ``"mean"``, ``"count"``, …

        Every reduction goes through here. ``RuntimeAggregate.__init__``
        validates the name against its own vocabulary, so the check lives in
        one place and ``col("s").count_distnct()`` raises where it was
        written."""
        agg = Aggregate.wrap(_ma.expr_aggregate(self._binding, function))
        return agg.alias(alias) if alias is not None else agg


# ---------------------------------------------------------------------------
# Generated verbs
# ---------------------------------------------------------------------------

# Notes for the verbs whose name does not carry its own meaning. Everything
# else gets a one-line docstring naming the Mojo verb it forwards to; the
# semantics are documented once, in `marrow/expr/runtime/values.mojo`.
_NOTES = {
    "length": "The **byte** count — SQL ``octet_length``. See ``char_length``.",
    "char_length": "The **character** count, as ``int64`` — SQL ``length``.",
    "ascii": "The first character's code point, ``0`` for the empty string.",
    "minimum": (
        "The smaller of the two, per row. Null-in-null-out, where SQL's "
        "``LEAST`` and ``pc.min_element_wise`` both skip nulls."
    ),
    "maximum": "The larger of the two, per row — the mirror of ``minimum``.",
    "position": "1-based index of the first occurrence, ``0`` when absent.",
    "substr": "SQL ``substr(a, start, count)`` — 1-based, counting characters.",
    "lpad": "Pad on the left, **truncating** when already longer than width.",
    "rpad": "``lpad`` from the other end.",
    "left": "The first ``count`` characters; negative drops from the end.",
    "right": "``left`` from the other end.",
    "epoch": "Seconds since the Unix epoch, as ``int64``.",
    "last_day": "The last day of this value's month, as ``date32``.",
    "array_length": (
        "The number of elements in each list, as ``int32``. The only verb "
        "that reads a list column: a list element is a whole sub-array "
        "rather than a value an expression can hold."
    ),
    "array_contains": "True where ``list[i]`` contains the value ``elem[i]``.",
    "is_null": "True where this is null. Never null itself.",
    "is_valid": "True where this is *not* null — Arrow's spelling of ``~is_null``.",
    "is_nan": "True where this floating-point value is NaN.",
    "is_inf": "True where this floating-point value is +/-infinity.",
    "fill_null": "`other` wherever this is null, this expression elsewhere.",
    "nullif": "Null wherever this equals `other`, otherwise unchanged.",
    "sqrt": "The square root, as ``float64`` whatever the input type.",
}


def _make_verb(verb, arity):
    """One ``Column`` method forwarding to ``expr_call(verb, ...)``."""
    extra = arity - 1

    def method(self, *args):
        if len(args) != extra:
            raise TypeError(f"{verb}() takes {extra} argument(s), got {len(args)}")
        return _call(verb, self, *args)

    method.__name__ = verb
    method.__qualname__ = f"Column.{verb}"
    method.__doc__ = _NOTES.get(verb) or (
        f"``{verb}`` — the runtime lane's verb of the same name."
    )
    return method


def _install_verbs():
    """Attach every vocabulary verb that is not already written by hand.

    Hand-written wins, always: ``cast`` and ``like`` carry payloads the table
    cannot describe, and the operators coerce bare Python scalars. Attaching
    only what is missing means adding a hand-written override never needs a
    matching deletion from a blocklist."""
    for verb, arity in _ma.expr_verbs().items():
        if not hasattr(Column, verb):
            setattr(Column, verb, _make_verb(verb, arity))


def _make_reduction(verb):
    def method(self, *, alias=None):
        return self.aggregate(verb, alias=alias)

    method.__name__ = verb
    method.__qualname__ = f"Column.{verb}"
    method.__doc__ = _AGG_NOTES.get(verb) or f"``{verb.upper()}(x)``."
    return method


_AGG_NOTES = {
    "mean": "``AVG(x)``. Spelled ``mean`` as Arrow spells it.",
    "count": (
        "``COUNT(x)`` — the *non-null* values of ``x``. Not the same as "
        ":func:`count_star`, which counts rows; the two differ on any "
        "nullable column."
    ),
    "count_distinct": "``COUNT(DISTINCT x)`` — exact, and skipping nulls.",
    "approx_count_distinct": "``COUNT(DISTINCT x)`` from a sketch.",
    "variance": "``VAR_POP(x)`` — the population variance, Arrow's default.",
    "var_samp": "``VAR_SAMP(x)`` — the sample variance, ``ddof=1``.",
    "stddev": "``STDDEV_POP(x)``.",
    "stddev_samp": "``STDDEV_SAMP(x)`` — the square root of ``var_samp``.",
}


def _install_reductions():
    """Attach the aggregate vocabulary as ``Column`` methods.

    ``Agg``'s twelve reductions were twelve registered binding methods
    restating ``RuntimeAggregate.VOCABULARY``; ``agg_verbs()`` hands over the
    list instead, and ``aggregate(name)`` is the one path they all take."""
    for verb in _ma.agg_verbs():
        if not hasattr(Column, verb):
            setattr(Column, verb, _make_reduction(verb))


_install_verbs()
_install_reductions()


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

    def over(
        self,
        partition_by=(),
        order_by=(),
        ascending=(),
        nulls_first=True,
        rows=None,
    ):
        """``SUM(x) OVER (...)`` — this aggregate evaluated over each frame.

        Only an aggregate may be windowed: a per-row value has nothing to do
        with a frame, and `col("v").over(...)` is a mistake worth a diagnostic
        rather than a silent column of copies."""
        return _over(
            self._binding, partition_by, order_by, ascending, nulls_first, rows
        )

    def __str__(self):
        return self.render()

    def __repr__(self):
        return f"<marrow.Aggregate: {self.render()}>"


class Window(_Wrapper):
    """A window function and the window it runs in — the Python face of
    ``WindowExpr``.

    Built by :func:`row_number` and friends, or by a column verb like
    :meth:`Column.lag`, then placed in a window with :meth:`over`. Consumed by
    ``LazyTable.with_columns``, which routes a `Window` to the plan's windowed
    overload."""

    def over(
        self,
        partition_by=(),
        order_by=(),
        ascending=(),
        nulls_first=True,
        rows=None,
    ):
        """``OVER (PARTITION BY ... ORDER BY ...)`` — the window this runs in.

        `ascending` defaults to all-ascending, sized to `order_by`, so the
        common case names only the keys. `rows` is an explicit ``ROWS`` frame
        as ``(preceding, following)``; without it the default ``RANGE`` frame
        applies, and the two agree only when the order key has no duplicates.
        """
        return _over(
            self._binding, partition_by, order_by, ascending, nulls_first, rows
        )

    def referenced_columns(self):
        """Every column name this window function reads."""
        return self._binding.referenced_columns()

    def render(self):
        """This window function and its window, as text.

        `WindowExpr.spec()` is deliberately *not* exposed: it renders the
        window alone as an identity key for grouping — `with_columns` stacks
        one `Window` node per distinct spec — and reads as
        ``"p=k,|o=va,|n=True"``. That is an internal discriminant, not a
        clause a caller should be shown."""
        return self._binding.render()

    def __str__(self):
        return self.render()

    def __repr__(self):
        return f"<marrow.Window: {self.render()}>"


def _over(binding, partition_by, order_by, ascending, nulls_first, rows):
    """`over` for both `Aggregate` and `Window` — one marshalling rule."""
    return Window.wrap(
        binding.over(
            [_key(k) for k in partition_by],
            [_key(k) for k in order_by],
            [bool(a) for a in ascending],
            bool(nulls_first),
            tuple(rows) if rows is not None else None,
        )
    )


def _key(value):
    """A partition/order key: an expression, or a column name."""
    return value.unwrap() if isinstance(value, Column) else value


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
    return Column.wrap(_ma.expr_literal(unwrap(array([value], type))))


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


# ── free-function spellings ────────────────────────────────────────────────
#
# `marrow.expr.builders` exposes these as free functions in Mojo, because the
# comptime lane's operands are bound on family traits and a method would have
# to live on whichever trait the *first* operand happens to satisfy. Python
# has them as `Column` methods, which reads better; both spellings exist here
# so one golden case text runs in either lane.


def array_length(value):
    """The number of elements in each list, as ``int32``."""
    return value.array_length()


def array_contains(list_value, element):
    """True where ``list_value[i]`` contains ``element[i]``."""
    return list_value.array_contains(element)


def is_in(value, value_set):
    """Membership against a value set — ``Column.isin`` under Mojo's name."""
    return value.isin(value_set)


def minimum(left, right):
    """The smaller of the two, per row. Null-in-null-out."""
    return _expr_or_column(left).minimum(right)


def maximum(left, right):
    """The larger of the two, per row. Null-in-null-out."""
    return _expr_or_column(left).maximum(right)


def _expr_or_column(value):
    """`value` as a `Column`, so a bare scalar may lead a binary verb."""
    return value if isinstance(value, Column) else lit(value)


# ── window functions that read no column ───────────────────────────────────
#
# Free functions in both lanes, because they read position rather than a
# column: there is no receiver for them to be a method of.


def row_number():
    """``ROW_NUMBER()`` — a distinct position per row within the partition."""
    return Window.wrap(_ma.window_row_number())


def rank():
    """``RANK()`` — ties share the first position and the next row skips the
    gap: ``1, 2, 2, 2, 5``."""
    return Window.wrap(_ma.window_rank())


def dense_rank():
    """``DENSE_RANK()`` — ties share a position and nothing is skipped:
    ``1, 2, 2, 2, 3``."""
    return Window.wrap(_ma.window_dense_rank())


def percent_rank():
    """``PERCENT_RANK()`` — ``(rank - 1) / (rows - 1)``, 0 to 1 inclusive."""
    return Window.wrap(_ma.window_percent_rank())


def cume_dist():
    """``CUME_DIST()`` — the fraction of the partition at or before this row's
    peer group."""
    return Window.wrap(_ma.window_cume_dist())


def ntile(buckets):
    """``NTILE(n)`` — the partition split into `buckets` as evenly as it
    divides."""
    return Window.wrap(_ma.window_ntile(buckets))
