"""Type-erased runtime expression nodes for the marrow expression system.

``DynValue`` is the runtime counterpart to the comptime-typed nodes in
``marrow.expr.values``.  It lets query plans be built and executed without
knowing concrete comptime types — this is what the Python bindings (and any
other runtime-typed caller) drive.  A single ``DynValue`` node carries a tag
plus its child args, and dispatches its own execution by tag in ``eval()``.
To mix it with fused values, box it into ``AnyValue`` (``marrow.expr.values``)
— the one box the relational layer holds.

Factory functions
-----------------
``col(index)``  / ``col(name)`` — column reference
``lit[T](value)``               — typed scalar literal
``if_else(cond, then_, else_)`` — conditional

Operator overloads on ``DynValue``: ``+``, ``-``, ``*``, ``/``, ``>``,
``<``, ``>=``, ``<=``, ``==``, ``!=``, ``&``, ``|``, ``~`` (NOT),
unary ``-``.  Instance methods: ``.abs()``, ``.is_null()``, ``.length()``,
``.cast(to)``.

Expression tags
---------------
LOAD    - Column reference
LITERAL - Constant value
ADD/SUB/MUL/DIV/EQ/NE/LT/LE/GT/GE/AND/OR - Binary operations
NEG/ABS/NOT - Unary operations
IS_NULL - Null check
IF_ELSE - Conditional
LENGTH - String byte length (dispatches to kernels.string.LengthKernel)
LIKE/ILIKE - SQL LIKE pattern match (kernels.string.LikeKernel/ILikeKernel)
IS_IN - Set membership (kernels.membership.is_in)
COALESCE/NULLIF/CASE_WHEN - Conditional / null handling (kernels.conditional)
YEAR/MONTH/DAY/HOUR/MINUTE/SECOND/DAY_OF_WEEK/QUARTER/DAY_OF_YEAR - Temporal
  field extraction (kernels.temporal); DATE_TRUNC - floor to a unit boundary
"""

from ..arrays import AnyArray, BoolArray
from ..builders import StringBuilder
from ..dtypes import AnyDataType, NumericType
from ..scalars import AnyScalar, PrimitiveScalar
from ..schema import Schema
from ..tabular import RecordBatch
from .pruning import PruneStats, PruneBound
from ..kernels.numeric import (
    AddKernel,
    SubKernel,
    MulKernel,
    DivKernel,
    ModKernel,
    FloordivKernel,
    NegKernel,
    AbsKernel,
)
from ..kernels.boolean import (
    AndKernel,
    OrKernel,
    NotKernel,
    XorKernel,
    NotNullKernel,
    IsNullKernel,
)
from ..kernels.numeric import (
    NumericCompareKernel,
    EqKernel,
    NeKernel,
    LtKernel,
    LeKernel,
    GtKernel,
    GeKernel,
)
from ..kernels.string import (
    StringPredicateKernel,
    StringEqKernel,
    StringNeKernel,
    StringLtKernel,
    StringLeKernel,
    StringGtKernel,
    StringGeKernel,
    LengthKernel,
    LikeKernel,
    ILikeKernel,
)
from ..kernels.membership import is_in as is_in_kernel
from ..kernels.conditional import (
    coalesce as coalesce_kernel,
    nullif as nullif_kernel,
    case_when as case_when_kernel,
)
from ..kernels.temporal import (
    YearKernel,
    MonthKernel,
    DayKernel,
    HourKernel,
    MinuteKernel,
    SecondKernel,
    DayOfWeekKernel,
    QuarterKernel,
    DayOfYearKernel,
    date_trunc as date_trunc_kernel,
)
from ..kernels.cast import cast as cast_array


# ---------------------------------------------------------------------------
# Node-kind / op constants
# ---------------------------------------------------------------------------

comptime LOAD: UInt8 = 0
comptime LITERAL: UInt8 = 1
comptime ADD: UInt8 = 2
comptime SUB: UInt8 = 3
comptime MUL: UInt8 = 4
comptime DIV: UInt8 = 5
comptime EQ: UInt8 = 6
comptime NE: UInt8 = 7
comptime LT: UInt8 = 8
comptime LE: UInt8 = 9
comptime GT: UInt8 = 10
comptime GE: UInt8 = 11
comptime AND: UInt8 = 12
comptime OR: UInt8 = 13
comptime NEG: UInt8 = 14
comptime ABS: UInt8 = 15
comptime NOT: UInt8 = 16
comptime IS_NULL: UInt8 = 17
comptime IF_ELSE: UInt8 = 18
comptime LENGTH: UInt8 = 20
comptime CAST: UInt8 = 21
comptime MOD: UInt8 = 22
comptime FLOORDIV: UInt8 = 23
comptime XOR: UInt8 = 24
comptime NOT_NULL: UInt8 = 25
# Wave 1 kernels wired into the runtime interpreter (T2.2).
comptime LIKE: UInt8 = 26
comptime ILIKE: UInt8 = 27
comptime IS_IN: UInt8 = 28
comptime COALESCE: UInt8 = 29
comptime NULLIF: UInt8 = 30
comptime CASE_WHEN: UInt8 = 31
comptime YEAR: UInt8 = 32
comptime MONTH: UInt8 = 33
comptime DAY: UInt8 = 34
comptime HOUR: UInt8 = 35
comptime MINUTE: UInt8 = 36
comptime SECOND: UInt8 = 37
comptime DAY_OF_WEEK: UInt8 = 38
comptime QUARTER: UInt8 = 39
comptime DAY_OF_YEAR: UInt8 = 40
comptime DATE_TRUNC: UInt8 = 41


# ---------------------------------------------------------------------------
# DynValue - unified n-ary term expression node
# ---------------------------------------------------------------------------


struct DynValue(
    Copyable,
    ImplicitlyCopyable,
    ImplicitlyDeletable,
    Movable,
    Writable,
):
    """Unified expression node using tag-based dispatch.

    Every operation (``ADD``, ``IS_NULL``, ...) is just a tag plus child
    args — there is no per-op wrapper struct.  ``eval()`` dispatches
    directly on the tag, recursing into ``_args``.
    """

    var _tag: UInt8
    var _args: List[DynValue]
    var _kind_data: UInt8
    var _value: Optional[AnyScalar]
    var _name: String
    var _cast_to: Optional[AnyDataType]
    """Target dtype for a CAST node (None for every other tag)."""
    var _value_set: Optional[AnyArray]
    """Captured value-set array for an IS_IN node (None for every other tag)."""

    def __init__(
        out self,
        tag: UInt8,
        var args: List[DynValue],
        kind_data: UInt8,
        var value: Optional[AnyScalar],
        var name: String,
        var cast_to: Optional[AnyDataType] = None,
        var value_set: Optional[AnyArray] = None,
    ):
        self._tag = tag
        self._args = args^
        self._kind_data = kind_data
        self._value = value.copy()
        self._name = name^
        self._cast_to = cast_to^
        self._value_set = value_set^

    def __init__(out self, *, copy: Self):
        self._tag = copy._tag
        self._args = List[DynValue]()
        for i in range(len(copy._args)):
            self._args.append(copy._args[i].copy())
        self._kind_data = copy._kind_data
        self._value = copy._value.copy()
        self._name = copy._name.copy()
        self._cast_to = copy._cast_to.copy()
        self._value_set = copy._value_set.copy()

    # Explicit (empty) destructor so this self-referential struct
    # (`_args: List[DynValue]`) is ImplicitlyDeletable; fields are still destroyed
    # automatically after the body runs.
    def __del__(deinit self):
        pass

    def kind(self) -> UInt8:
        return self._tag

    def kind_data(self) -> UInt8:
        """Return the kind-specific data (e.g. column index for LOAD nodes)."""
        return self._kind_data

    def name(self) -> String:
        """Return the column name for a named LOAD node (empty otherwise).

        ``_name`` is overloaded: it also carries the LIKE/ILIKE pattern and the
        ``date_trunc`` unit. The tag check is what keeps a ``LIKE`` node from
        reporting ``"%foo%"`` — and a ``DATE_TRUNC`` node ``"day"`` — as its
        output column name."""
        if self._tag == LOAD:
            return self._name.copy()
        else:
            return String()

    def execute(self, batch: RecordBatch) raises -> AnyArray:
        """Evaluate against *batch*. This is the ``AnyValue``-box entry point the
        fused/streaming relations call per morsel; ``eval`` resolves named
        ``col(...)`` references inline (a per-column schema lookup, no tree copy),
        so there is no per-morsel ``resolve_names`` reconstruction."""
        return self.eval(batch)

    def column_index(self, schema: Schema) raises -> Int:
        """Resolve this expression to a column position for use as a join/group
        key. Requires a bare column reference (``col(...)``); raises on a
        computed expression, so callers never mis-read a non-column key."""
        var resolved = self.resolve_names(schema)
        if resolved.kind() != LOAD:
            raise Error(
                "expected a column reference as a key, got a computed"
                " expression"
            )
        return Int(resolved.kind_data())

    def resolve_names(self, schema: Schema) raises -> DynValue:
        """Recursively resolve ``col("name")`` references against *schema*.

        Returns a copy of this expression tree with every named LOAD node
        replaced by a positional column reference.
        """
        if self._tag == LOAD and self._name.byte_length() > 0:
            var idx = schema.get_field_index(self._name)
            if idx == -1:
                raise Error(
                    "resolve_names: column '" + self._name + "' not found"
                )
            return col(idx)
        var result = self.copy()
        for i in range(len(result._args)):
            result._args[i] = self._args[i].resolve_names(schema)
        return result^

    def dtype(self) -> Optional[AnyDataType]:
        if self._tag == LITERAL:
            return self._value.value().type()
        elif self._tag == CAST:
            return self._cast_to.copy()
        return None

    def referenced_columns(self) -> List[String]:
        """Every distinct column this expression reads, in first-seen order.

        Walks the tree collecting each ``LOAD`` leaf: named ``col("x")`` leaves
        contribute their name; positional ``col(i)`` leaves contribute the index
        rendered as a string. Deduped so a column referenced twice
        (``col("a") + col("a")``) appears once. Plan analysis (projection pushdown,
        column pruning) uses this to know which inputs a projection/filter needs.
        """
        var out = List[String]()
        self._collect_columns(out)
        return out^

    def _collect_columns(self, mut out: List[String]):
        if self._tag == LOAD:
            var ref_name = (
                self._name.copy() if self._name.byte_length()
                > 0 else String(Int(self._kind_data))
            )
            var seen = False
            for i in range(len(out)):
                if out[i] == ref_name:
                    seen = True
                    break
            if not seen:
                out.append(ref_name^)
        else:
            for i in range(len(self._args)):
                self._args[i]._collect_columns(out)

    def is_deterministic(self) -> Bool:
        """Whether repeated evaluation on identical input yields identical output.

        Every tag currently supported (columns, literals, arithmetic, compares,
        boolean logic, casts, conditionals, validity predicates) is a pure
        function of its inputs, so this is always ``True``. Non-deterministic tags
        (``random``, ``now``, ...) would return ``False`` here and gate CSE /
        subtree-caching once they exist.
        """
        return True

    def _compare[
        N: NumericCompareKernel, S: StringPredicateKernel
    ](self, left: AnyArray, right: AnyArray) raises -> AnyArray:
        """Apply one comparison operator, choosing the kernel family the
        operands belong to.

        Numeric and string comparison are unrelated implementations — SIMD over
        fixed-width lanes versus an elementwise walk over variable-width data —
        so the operator names a *pair* of kernels and the dtype picks one. This
        used to live inside the numeric kernel as a `comptime StringKernel` plus
        a dtype branch in its `dispatch`, which made every numeric comparison
        carry a string counterpart it never used. Interpreting an operator is
        this layer's job; the kernels stay one family each.
        """
        if left.dtype().is_string() or left.dtype().is_large_string():
            return S.dispatch(left, right)
        else:
            return N.dispatch(left, right)

    def eval(self, batch: RecordBatch) raises -> AnyArray:
        """Evaluate this expression tree against *batch*, dispatching on tag."""
        if self._tag == LOAD:
            # Named LOADs resolve their position by name here (one schema lookup
            # per column reference, no tree copy); positional LOADs (from
            # col(index) / bound keys) use kind_data directly.
            if self._name.byte_length() > 0:
                var idx = batch.schema.get_field_index(self._name)
                if idx == -1:
                    raise Error("eval: column '" + self._name + "' not found")
                return batch.columns[idx].copy()
            return batch.columns[Int(self._kind_data)].copy()
        elif self._tag == LITERAL:
            return self._value.value().repeat(batch.num_rows())
        elif self._tag == ADD:
            return AddKernel.dispatch(
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == SUB:
            return SubKernel.dispatch(
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == MUL:
            return MulKernel.dispatch(
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == DIV:
            return DivKernel.dispatch(
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == MOD:
            return ModKernel.dispatch(
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == FLOORDIV:
            return FloordivKernel.dispatch(
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == EQ:
            return self._compare[EqKernel, StringEqKernel](
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == NE:
            return self._compare[NeKernel, StringNeKernel](
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == LT:
            return self._compare[LtKernel, StringLtKernel](
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == LE:
            return self._compare[LeKernel, StringLeKernel](
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == GT:
            return self._compare[GtKernel, StringGtKernel](
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == GE:
            return self._compare[GeKernel, StringGeKernel](
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == AND:
            return AndKernel.dispatch(
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == OR:
            return OrKernel.dispatch(
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == XOR:
            return XorKernel.dispatch(
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == NEG:
            return NegKernel.dispatch(self._args[0].eval(batch))
        elif self._tag == ABS:
            return AbsKernel.dispatch(self._args[0].eval(batch))
        elif self._tag == NOT:
            return NotKernel.dispatch(self._args[0].eval(batch))
        elif self._tag == IS_NULL:
            return IsNullKernel.dispatch(self._args[0].eval(batch))
        elif self._tag == NOT_NULL:
            return NotNullKernel.dispatch(self._args[0].eval(batch))
        elif self._tag == LENGTH:
            return LengthKernel.dispatch(self._args[0].eval(batch))
        elif self._tag == CAST:
            return cast_array(self._args[0].eval(batch), self._cast_to.value())
        elif self._tag == IF_ELSE:
            # One-branch CASE WHEN: the same null semantics (a null condition
            # counts as false) and the same dtype coverage as `case_when`.
            return case_when_kernel(
                [self._args[0].eval(batch).as_bool().copy()],
                [self._args[1].eval(batch)],
                self._args[2].eval(batch),
            )
        elif self._tag == LIKE:
            var left = self._args[0].eval(batch)
            return LikeKernel.dispatch(left, self._pattern_array(left.length()))
        elif self._tag == ILIKE:
            var left = self._args[0].eval(batch)
            return ILikeKernel.dispatch(
                left, self._pattern_array(left.length())
            )
        elif self._tag == IS_IN:
            return is_in_kernel(
                self._args[0].eval(batch), self._value_set.value()
            ).to_any()
        elif self._tag == COALESCE:
            var arrays = List[AnyArray]()
            for i in range(len(self._args)):
                arrays.append(self._args[i].eval(batch))
            return coalesce_kernel(arrays)
        elif self._tag == NULLIF:
            return nullif_kernel(
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == CASE_WHEN:
            var has_else = Int(self._kind_data)
            var m = (len(self._args) - has_else) // 2
            var conditions = List[BoolArray]()
            var values = List[AnyArray]()
            for k in range(m):
                conditions.append(
                    self._args[2 * k].eval(batch).as_bool().copy()
                )
                values.append(self._args[2 * k + 1].eval(batch))
            var else_ = Optional[AnyArray](None)
            if has_else == 1:
                else_ = self._args[len(self._args) - 1].eval(batch)
            return case_when_kernel(conditions, values, else_^)
        elif self._tag == YEAR:
            return YearKernel.dispatch(self._args[0].eval(batch))
        elif self._tag == MONTH:
            return MonthKernel.dispatch(self._args[0].eval(batch))
        elif self._tag == DAY:
            return DayKernel.dispatch(self._args[0].eval(batch))
        elif self._tag == HOUR:
            return HourKernel.dispatch(self._args[0].eval(batch))
        elif self._tag == MINUTE:
            return MinuteKernel.dispatch(self._args[0].eval(batch))
        elif self._tag == SECOND:
            return SecondKernel.dispatch(self._args[0].eval(batch))
        elif self._tag == DAY_OF_WEEK:
            return DayOfWeekKernel.dispatch(self._args[0].eval(batch))
        elif self._tag == QUARTER:
            return QuarterKernel.dispatch(self._args[0].eval(batch))
        elif self._tag == DAY_OF_YEAR:
            return DayOfYearKernel.dispatch(self._args[0].eval(batch))
        elif self._tag == DATE_TRUNC:
            return date_trunc_kernel(self._args[0].eval(batch), self._name)
        else:
            raise Error("DynValue.eval: unknown expression kind ", self._tag)

    def _pattern_array(self, n: Int) raises -> AnyArray:
        """Broadcast a LIKE/ILIKE pattern (stored in ``_name``) into an ``n``-row
        ``StringArray`` — the string-predicate kernels compare element-wise, so
        the constant pattern is materialised once per morsel here."""
        var b = StringBuilder(capacity=n)
        for _ in range(n):
            b.append(self._name)
        return b.finish()

    def prune(self, stats: PruneStats) raises -> PruneBound:
        """Pruning evaluation by tag (the runtime counterpart of the fused
        nodes' `prune`): column -> its stats interval, literal -> a point
        interval, comparisons -> the min/max rule, AND/OR -> combine. Anything
        not modelled (arithmetic, NOT, IS_NULL, ...) is conservatively unknown /
        maybe-true, so a caller only ever skips data it has proven cannot match.
        """
        if self._tag == LOAD:
            var iv = stats.by_name(
                self._name
            ) if self._name.byte_length() > 0 else stats.by_index(
                Int(self._kind_data)
            )
            return PruneBound.interval(iv[0].copy(), iv[1].copy())
        elif self._tag == LITERAL:
            return PruneBound.interval(
                Optional(self._value.value().copy()),
                Optional(self._value.value().copy()),
            )
        elif self._tag == EQ:
            return PruneBound.boolean(
                self._args[0].prune(stats).maybe_eq(self._args[1].prune(stats))
            )
        elif self._tag == LT:
            return PruneBound.boolean(
                self._args[0].prune(stats).maybe_lt(self._args[1].prune(stats))
            )
        elif self._tag == LE:
            return PruneBound.boolean(
                self._args[0].prune(stats).maybe_le(self._args[1].prune(stats))
            )
        elif self._tag == GT:
            return PruneBound.boolean(
                self._args[0].prune(stats).maybe_gt(self._args[1].prune(stats))
            )
        elif self._tag == GE:
            return PruneBound.boolean(
                self._args[0].prune(stats).maybe_ge(self._args[1].prune(stats))
            )
        elif self._tag == AND:
            return PruneBound.boolean(
                self._args[0].prune(stats).maybe_true
                and self._args[1].prune(stats).maybe_true
            )
        elif self._tag == OR:
            return PruneBound.boolean(
                self._args[0].prune(stats).maybe_true
                or self._args[1].prune(stats).maybe_true
            )
        else:
            return PruneBound.unknown()

    def _op_name(self) -> String:
        """Display name for an operator tag (empty if not an operator)."""
        if self._tag == ADD:
            return "add"
        elif self._tag == SUB:
            return "sub"
        elif self._tag == MUL:
            return "mul"
        elif self._tag == DIV:
            return "div"
        elif self._tag == MOD:
            return "mod"
        elif self._tag == FLOORDIV:
            return "floordiv"
        elif self._tag == EQ:
            return "equal"
        elif self._tag == NE:
            return "not_equal"
        elif self._tag == LT:
            return "less"
        elif self._tag == LE:
            return "less_equal"
        elif self._tag == GT:
            return "greater"
        elif self._tag == GE:
            return "greater_equal"
        elif self._tag == AND:
            return "and"
        elif self._tag == OR:
            return "or"
        elif self._tag == XOR:
            return "xor"
        elif self._tag == NEG:
            return "neg"
        elif self._tag == ABS:
            return "abs"
        elif self._tag == NOT:
            return "not"
        elif self._tag == IS_NULL:
            return "is_null"
        elif self._tag == NOT_NULL:
            return "not_null"
        elif self._tag == LENGTH:
            return "length"
        elif self._tag == CAST:
            return "cast"
        elif self._tag == IF_ELSE:
            return "if_else"
        elif self._tag == LIKE:
            return "match_like"
        elif self._tag == ILIKE:
            return "match_like_ci"
        elif self._tag == IS_IN:
            return "is_in"
        elif self._tag == COALESCE:
            return "coalesce"
        elif self._tag == NULLIF:
            return "nullif"
        elif self._tag == CASE_WHEN:
            return "case_when"
        elif self._tag == YEAR:
            return "year"
        elif self._tag == MONTH:
            return "month"
        elif self._tag == DAY:
            return "day"
        elif self._tag == HOUR:
            return "hour"
        elif self._tag == MINUTE:
            return "minute"
        elif self._tag == SECOND:
            return "second"
        elif self._tag == DAY_OF_WEEK:
            return "day_of_week"
        elif self._tag == QUARTER:
            return "quarter"
        elif self._tag == DAY_OF_YEAR:
            return "day_of_year"
        elif self._tag == DATE_TRUNC:
            return "date_trunc"
        else:
            return String()

    def write_to[W: Writer](self, mut writer: W):
        if self._tag == LOAD:
            # A named reference renders as its name until `resolve_names` binds
            # it to a position; rendering it as `input(0)` beforehand would
            # report an index it does not yet have (every unresolved name
            # carries `_kind_data == 0`, so they would all print alike).
            if self._name:
                writer.write(self._name)
            else:
                writer.write(t"input({self._kind_data})")
        elif self._tag == LITERAL:
            writer.write("literal(...)")
        else:
            var op = self._op_name()
            if op.byte_length() == 0:
                writer.write(t"<unknown>({self._tag})")
            else:
                writer.write(t"{op}(")
                for i in range(len(self._args)):
                    if i > 0:
                        writer.write(t", ")
                    self._args[i].write_to(writer)
                # LIKE/ILIKE carry their pattern and DATE_TRUNC its unit in
                # ``_name`` (not an arg node) — surface it in the rendering.
                if (
                    self._tag == LIKE
                    or self._tag == ILIKE
                    or self._tag == DATE_TRUNC
                ):
                    writer.write(t", {self._name}")
                elif self._tag == IS_IN:
                    writer.write(t", value_set")
                writer.write(t")")

    # Node builders shared by the operator overloads below
    def _binary(self, tag: UInt8, rhs: DynValue) -> DynValue:
        return DynValue(
            tag=tag,
            args=[self.copy(), rhs.copy()],
            kind_data=0,
            value=None,
            name=String(),
        )

    def _unary(self, tag: UInt8) -> DynValue:
        return DynValue(
            tag=tag,
            args=[self.copy()],
            kind_data=0,
            value=None,
            name=String(),
        )

    def cast(self, to: AnyDataType) -> DynValue:
        """Build a runtime cast node — ``col("a").cast(float64)``. Evaluated by
        the ``marrow.kernels.cast`` router (numeric/bool/temporal families)."""
        return DynValue(
            tag=CAST,
            args=[self.copy()],
            kind_data=0,
            value=None,
            name=String(),
            cast_to=to.copy(),
        )

    # Operator overloads (methods on DynValue)
    def __add__(self, rhs: DynValue) -> DynValue:
        return self._binary(ADD, rhs)

    def __sub__(self, rhs: DynValue) -> DynValue:
        return self._binary(SUB, rhs)

    def __mul__(self, rhs: DynValue) -> DynValue:
        return self._binary(MUL, rhs)

    def __truediv__(self, rhs: DynValue) -> DynValue:
        return self._binary(DIV, rhs)

    def __mod__(self, rhs: DynValue) -> DynValue:
        return self._binary(MOD, rhs)

    def __floordiv__(self, rhs: DynValue) -> DynValue:
        return self._binary(FLOORDIV, rhs)

    def __xor__(self, rhs: DynValue) -> DynValue:
        return self._binary(XOR, rhs)

    def __gt__(self, rhs: DynValue) -> DynValue:
        return self._binary(GT, rhs)

    def __lt__(self, rhs: DynValue) -> DynValue:
        return self._binary(LT, rhs)

    def __ge__(self, rhs: DynValue) -> DynValue:
        return self._binary(GE, rhs)

    def __le__(self, rhs: DynValue) -> DynValue:
        return self._binary(LE, rhs)

    def __eq__(self, rhs: DynValue) -> DynValue:
        return self._binary(EQ, rhs)

    def __ne__(self, rhs: DynValue) -> DynValue:
        return self._binary(NE, rhs)

    def __and__(self, rhs: DynValue) -> DynValue:
        return self._binary(AND, rhs)

    def __or__(self, rhs: DynValue) -> DynValue:
        return self._binary(OR, rhs)

    def __neg__(self) -> DynValue:
        return self._unary(NEG)

    def __invert__(self) -> DynValue:
        return self._unary(NOT)

    def is_null(self) -> DynValue:
        return self._unary(IS_NULL)

    def not_null(self) -> DynValue:
        return self._unary(NOT_NULL)

    def abs(self) -> DynValue:
        return self._unary(ABS)

    def length(self) -> DynValue:
        return self._unary(LENGTH)

    # --- string pattern matching (kernels.string) --------------------------
    def like(self, var pattern: String) -> DynValue:
        """SQL ``LIKE`` against a constant *pattern* (``%``/``_`` wildcards,
        case-sensitive) — dispatches to ``kernels.string.LikeKernel``."""
        return DynValue(
            tag=LIKE,
            args=[self.copy()],
            kind_data=0,
            value=None,
            name=pattern^,
        )

    def ilike(self, var pattern: String) -> DynValue:
        """Case-insensitive SQL ``LIKE`` — ``kernels.string.ILikeKernel``."""
        return DynValue(
            tag=ILIKE,
            args=[self.copy()],
            kind_data=0,
            value=None,
            name=pattern^,
        )

    # --- set membership (kernels.membership) -------------------------------
    def isin(self, value_set: AnyArray) -> DynValue:
        """``self IN value_set`` — the value set is captured in the node and
        probed by ``kernels.membership.is_in``."""
        return DynValue(
            tag=IS_IN,
            args=[self.copy()],
            kind_data=0,
            value=None,
            name=String(),
            value_set=value_set.copy(),
        )

    # --- conditional / null handling (kernels.conditional) -----------------
    def nullif(self, other: DynValue) -> DynValue:
        """``NULLIF(self, other)`` — ``kernels.conditional.nullif``."""
        return self._binary(NULLIF, other)

    # --- temporal extraction (kernels.temporal) ----------------------------
    def year(self) -> DynValue:
        return self._unary(YEAR)

    def month(self) -> DynValue:
        return self._unary(MONTH)

    def day(self) -> DynValue:
        return self._unary(DAY)

    def hour(self) -> DynValue:
        return self._unary(HOUR)

    def minute(self) -> DynValue:
        return self._unary(MINUTE)

    def second(self) -> DynValue:
        return self._unary(SECOND)

    def day_of_week(self) -> DynValue:
        return self._unary(DAY_OF_WEEK)

    def quarter(self) -> DynValue:
        return self._unary(QUARTER)

    def day_of_year(self) -> DynValue:
        return self._unary(DAY_OF_YEAR)

    def date_trunc(self, var unit: String) -> DynValue:
        """Floor a temporal column to *unit* (``second``/``minute``/``hour``/
        ``day``) — ``kernels.temporal.date_trunc``."""
        return DynValue(
            tag=DATE_TRUNC,
            args=[self.copy()],
            kind_data=0,
            value=None,
            name=unit^,
        )

    # --- aggregates (marrow.expr.aggregates) --------------------------------
    #
    # An aggregate is not another `DynValue` tag: it does not produce a value
    # per row, it collapses rows within a group. `col("x").sum()` therefore
    # yields a `DynAgg` — this expression plus the aggregate applied to it —
    # which `AnyRelation.aggregate` turns into an output column.

    def aggregate(self, var func: String) -> DynAgg:
        """Apply the aggregate named ``func`` to this expression. The named
        entry point the sugar below is written in terms of, and what a frontend
        holding a runtime function name calls."""
        return DynAgg(func^, self.copy())

    def sum(self) -> DynAgg:
        return self.aggregate("sum")

    def product(self) -> DynAgg:
        return self.aggregate("product")

    def mean(self) -> DynAgg:
        return self.aggregate("mean")

    def min(self) -> DynAgg:
        return self.aggregate("min")

    def max(self) -> DynAgg:
        return self.aggregate("max")

    def count(self) -> DynAgg:
        return self.aggregate("count")

    def count_distinct(self) -> DynAgg:
        return self.aggregate("count_distinct")

    def approx_count_distinct(self) -> DynAgg:
        return self.aggregate("approx_count_distinct")


struct DynAgg(Copyable, Movable, Writable):
    """An aggregate applied to a runtime expression — ``col("x").sum()``.

    The dynamic counterpart of the fused ``AggExpr`` (``marrow.expr.values``):
    it names the aggregate rather than naming its ``Aggregation`` type, so the
    function is resolved once — against the input's dtype — when the plan is
    built. ``alias`` sets the output column name; without one the function's own
    name is used."""

    var func: String
    var input: DynValue
    var out_name: String

    def __init__(
        out self,
        var func: String,
        var input: DynValue,
        var out_name: String = String(),
    ):
        self.func = func^
        self.input = input^
        self.out_name = out_name^

    def alias(self, var name: String) -> DynAgg:
        """Name this aggregate's output column."""
        return DynAgg(self.func, self.input.copy(), name^)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.func, "(")
        self.input.write_to(writer)
        writer.write(")")
        if self.out_name:
            writer.write(" as ", self.out_name)


# ---------------------------------------------------------------------------
# Factory functions (return DynValue)
# ---------------------------------------------------------------------------


def col(index: Int) -> DynValue:
    """Reference to the ``index``-th input column."""
    return DynValue(
        tag=LOAD,
        args=List[DynValue](),
        kind_data=UInt8(index),
        value=None,
        name=String(),
    )


def col(var name: String) -> DynValue:
    """Reference to a named column."""
    return DynValue(
        tag=LOAD, args=List[DynValue](), kind_data=0, value=None, name=name^
    )


def lit[T: NumericType](value: Scalar[T.native]) raises -> DynValue:
    """A scalar constant."""
    return DynValue(
        tag=LITERAL,
        args=List[DynValue](),
        kind_data=0,
        value=PrimitiveScalar[T](value).to_any(),
        name=String(),
    )


def if_else(cond: DynValue, then_: DynValue, else_: DynValue) -> DynValue:
    """Element-wise conditional."""
    return DynValue(
        tag=IF_ELSE,
        args=[cond.copy(), then_.copy(), else_.copy()],
        kind_data=0,
        value=None,
        name=String(),
    )


def coalesce(var args: List[DynValue]) -> DynValue:
    """First non-null value across *args*, elementwise
    (``kernels.conditional.coalesce``)."""
    return DynValue(
        tag=COALESCE,
        args=args^,
        kind_data=0,
        value=None,
        name=String(),
    )


def case_when(
    conditions: List[DynValue],
    values: List[DynValue],
    else_: Optional[DynValue] = None,
) -> DynValue:
    """Multi-branch ``CASE WHEN`` (``kernels.conditional.case_when``).

    ``conditions[k]`` selects ``values[k]`` for the first branch that is
    valid-and-true; ``else_`` (or null) is used when none match. Conditions and
    values are stored interleaved in ``_args`` so ``referenced_columns`` /
    ``resolve_names`` recurse over every branch; ``kind_data`` flags whether an
    ``else_`` is present.
    """
    var args = List[DynValue]()
    for k in range(len(conditions)):
        args.append(conditions[k].copy())
        args.append(values[k].copy())
    var has_else: UInt8 = 0
    if else_:
        args.append(else_.value().copy())
        has_else = 1
    return DynValue(
        tag=CASE_WHEN,
        args=args^,
        kind_data=has_else,
        value=None,
        name=String(),
    )
