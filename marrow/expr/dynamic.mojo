"""The erased lane: an expression whose operations are named, not typed.

Two lanes exist and they no longer share node types.

`marrow.expr.values` is the **AOT lane**: every node's operands are bound on a
family trait (`L: NumericValue`), its output dtype is a comptime `NumericType`,
and it fuses into a SIMD loop. Nothing there is erased.

This module is the **runtime lane**. `DynValue` is one struct holding its
children, an optional payload, and a pointer to the function that evaluates it;
what stays runtime is the *dtype* of the operands, not the operation.

**Why they are separate.** They used to be one: every fused node carried a second
`_erased` body, selected by a hand-propagated `comptime IsErased`, and `DynValue`
claimed `NumericValue`/`BoolValue`/`StringValue`/`TemporalValue` so the fused
nodes would accept it as an operand. That conformance was **unsound** — the box
stubbed both of the promises those traits make:

- `comptime OutType: NumericType`, whose `native` was a placeholder `DType.bool`;
- `lane[W] -> SIMD[OutType.native, W]`, which had no lane and returned zero.

It satisfied the signatures and none of the contract, and the compiler surfaced
that as `attempt to resolve a recursive reference to declaration
'DynValue.__gt__'` — the error that forced the fluent surface to be split across
a `NumericOps` sub-trait for as long as it existed. Splitting the lanes removes
the cause: no *family* node is parameterised on this struct, so nothing needs a
relaxed bound, `IsErased` has nothing to propagate, and `NumericValue` carries
its whole surface again.

`NullPredicate` (`is_null`/`is_valid`) can still take this struct as an operand:
its bound is `A: Value` and it calls only runtime methods on it — an honest
bound, honestly satisfied, and the reason `Value` is the trait this erases into.
Those two arrive as defaults on `Value` and hand back a *fused* node, which is
what a `[V: Value]` caller gets. This struct overrides them so that inside this
lane a predicate stays a tag node: one that left the lane could not be combined
with one that did not, since `BoolValue.__or__` takes a `BoolValue` and a
`DynValue` is not one.

**What is erased here is the type, not the operation.** This lane exists for
expressions built where no *dtype* is available — a column named at run time,
`aggregate("sum")`, a future SQL or wire frontend. The operation is still known
when the node is built, so it stays comptime: `__gt__` names
`_compare[GtKernel, StringGtKernel]` and a program links exactly the kernels its
expressions mention. The `_tag` string that remains drives `render`, `prune` and
`name` only — it never selects a kernel. See `EvalFn` for what it cost to learn
that distinction.
"""

from std.builtin.rebind import rebind
from std.memory import ArcPointer
from std.utils import Variant

from ..arrays import DynArray
from ..dtypes import DynType, Float64Type
from ..scalars import DynScalar, StringScalar
from ..schema import Schema
from ..tabular import RecordBatch
from ..kernels.cast import cast as cast_array
from ..kernels.numeric import (
    BinaryNumericKernel,
    NumericCompareKernel,
    UnaryFloatKernel,
    UnaryNumericKernel,
    AbsKernel,
    AddKernel,
    CeilKernel,
    DivKernel,
    ExpKernel,
    FloordivKernel,
    FloorKernel,
    LogKernel,
    ModKernel,
    MulKernel,
    NegKernel,
    PowKernel,
    RoundKernel,
    SignKernel,
    SqrtKernel,
    SubKernel,
)
from ..kernels.temporal import (
    TemporalExtractKernel,
    CalendarUnit,
    DateTruncKernel,
    DayKernel,
    DayOfWeekKernel,
    DayOfYearKernel,
    HourKernel,
    MinuteKernel,
    MonthKernel,
    QuarterKernel,
    SecondKernel,
    YearKernel,
)
from ..kernels.boolean import (
    AndKernel,
    BoolBinaryKernel,
    BoolUnaryKernel,
    IsInfKernel,
    IsNanKernel,
    IsNullKernel,
    NotKernel,
    NotNullKernel,
    OrKernel,
    UnaryPredicateKernel,
    XorKernel,
)
from ..kernels.conditional import (
    BinaryConditionalKernel,
    CoalesceKernel,
    FillNullKernel,
    NullifKernel,
    case_when as case_when_kernel,
    fill_null as fill_null_kernel,
)
from ..kernels.numeric import (
    EqKernel,
    GeKernel,
    GtKernel,
    LeKernel,
    LtKernel,
    NeKernel,
)
from ..kernels.membership import IsInKernel
from ..kernels.string import (
    StringMapKernel,
    StringPredicateKernel,
    CapitalizeKernel,
    ConcatKernel,
    ContainsKernel,
    EndsWithKernel,
    ILikeKernel,
    LStripKernel,
    LengthKernel,
    LikeKernel,
    LowerKernel,
    ReverseKernel,
    RStripKernel,
    StartsWithKernel,
    StringEqKernel,
    StringGeKernel,
    StringGtKernel,
    StringLeKernel,
    StringLtKernel,
    StringNeKernel,
    StripKernel,
    UpperKernel,
)
from .core import Datum
from ..kernels.interval import (
    Interval,
    AndInterval,
    OrInterval,
    LtInterval,
    LeInterval,
    GtInterval,
    GeInterval,
    EqInterval,
)
from .pruning import PruneStats
from .values import Value


# ---------------------------------------------------------------------------
# Operand promotion — what `a + b` means across numeric types
# ---------------------------------------------------------------------------
def _numeric_rank(t: DynType) -> Int:
    """Runtime twin of the comptime `promote[L, R]` in `values.mojo`.

    `test_numeric_rank_agrees_across_lanes` pins the two together — they must
    order types identically or `a + b` means different things in the two lanes.
    """
    if t.is_floating_point():
        return 1000 + 8 * t.byte_width()
    else:
        return 8 * t.byte_width()


def _promote_operands(mut left: DynArray, mut right: DynArray) raises:
    """Cast both operands to the wider numeric domain, in place."""
    var lt = left.dtype()
    var rt = right.dtype()
    if lt == rt:
        pass
    elif _numeric_rank(lt) >= _numeric_rank(rt):
        right = cast_array(right, lt)
    else:
        left = cast_array(left, rt)


def _to_float(var a: DynArray) raises -> DynArray:
    """Cast one operand to float64.

    `sqrt`/`exp`/`ln` are `FloatUnary` in the fused lane — always float64, so
    `sqrt(int64_col)` works there. Their kernels dispatch over floating dtypes
    *only*, so without this the same expression raises here instead of
    computing.
    """
    var f64 = DynType(Float64Type())
    if a.dtype() == f64:
        return a^
    return cast_array(a^, f64)


def _promote_float(mut left: DynArray, mut right: DynArray) raises:
    """Cast both operands to float64, in place.

    True division and `**` are always float64 in the fused lane
    (`FloatBinary.OutType = Float64Type`, so `5 / 2 == 2.5`). Widening to the
    wider *operand* is not the same rule — for two int64 columns it stays
    integral and `/` would mean floor division here and true division there.
    """
    var f64 = DynType(Float64Type())
    if left.dtype() != f64:
        left = cast_array(left, f64)
    if right.dtype() != f64:
        right = cast_array(right, f64)


# ---------------------------------------------------------------------------
# The payload — the one runtime value some tags carry beyond their children
# ---------------------------------------------------------------------------
comptime DynPayload = Variant[NoneType, String, DynType, DynArray, DynScalar]


# ---------------------------------------------------------------------------
# The evaluator — what makes this lane cost only what it uses
# ---------------------------------------------------------------------------
comptime EvalFn = def(
    List[DynArray], DynPayload, RecordBatch
) thin raises -> DynArray
"""How one node computes its column, given its children's already-computed
columns, its payload and the batch.

**This is the size-critical decision in the module.** A node used to name its
operation with a string alone and one `_eval` switch resolved it, which meant
every kernel arm was reachable from every `DynValue`: building a single one
linked all ~70 of them. Measured, that cost `query_dynvalue` **+1,807,168 bytes
of `__text` (+45.7%)** — the whole win from deleting the 41-tag interpreter,
handed straight back.

The operation is comptime here instead. `__sub__` names `_binary[SubKernel]` and
nothing else, so a program links exactly the kernels its expressions mention;
the tag string survives only for `render`/`prune`/`name`, which reference no
kernel at all. It is a plain fn pointer rather than a node type per operation
because the children, the payload and every tree walk over them are already on
this struct — parameterising the *struct* would have duplicated all of that per
kernel to buy the same property.

The signature deliberately mentions no `Self`: a field whose function type names
the struct it lives in is rejected (`struct has recursive reference to itself`),
which is why children arrive pre-evaluated as `List[DynArray]`.
"""


struct DynValue(Copyable, Movable, Value, Writable):
    """A runtime-built expression: a tag, its children, and an optional payload.

    Conforms to `Value` and to nothing else. `Value` is a trait of runtime
    methods plus one comptime member, `OutShape`, which this struct can state
    truthfully — `execute` returns a `DynArray` unconditionally, so the answer
    is always "columnar". Contrast the family traits, whose comptime
    `OutType: NumericType`, `State` and `lane` it could only stub. That
    distinction is the whole design: **erase into a trait of methods, never into
    one with comptime members you cannot supply.**
    """

    comptime OutShape = 1

    var _tag: String
    """How this node prints and prunes. Not how it evaluates — `_eval_fn` is
    that, and it is comptime. A tag never selects a kernel."""
    var _kids: List[ArcPointer[Self]]
    """Children, behind `ArcPointer` because a bare `List[Self]` is rejected:
    `field '_kids' has non-implicitly deletable type`. The indirection is also
    what makes copying a subtree O(1)."""
    var _payload: DynPayload
    var _eval_fn: EvalFn

    def __init__(
        out self, var tag: String, ev: EvalFn, var payload: DynPayload
    ):
        self._tag = tag^
        self._kids = List[ArcPointer[Self]]()
        self._payload = payload^
        self._eval_fn = ev

    def __init__(out self, var tag: String, ev: EvalFn, a: Self):
        self._tag = tag^
        self._kids = [ArcPointer(a.copy())]
        self._payload = DynPayload(NoneType())
        self._eval_fn = ev

    def __init__(out self, var tag: String, ev: EvalFn, a: Self, b: Self):
        self._tag = tag^
        self._kids = [ArcPointer(a.copy()), ArcPointer(b.copy())]
        self._payload = DynPayload(NoneType())
        self._eval_fn = ev

    def __init__(
        out self, var tag: String, ev: EvalFn, a: Self, b: Self, c: Self
    ):
        self._tag = tag^
        self._kids = [
            ArcPointer(a.copy()),
            ArcPointer(b.copy()),
            ArcPointer(c.copy()),
        ]
        self._payload = DynPayload(NoneType())
        self._eval_fn = ev

    def __init__(out self, *, copy: Self):
        self._tag = copy._tag.copy()
        self._kids = copy._kids.copy()
        self._payload = copy._payload.copy()
        self._eval_fn = copy._eval_fn

    # --- payload accessors --------------------------------------------------
    def _text(self) raises -> String:
        if self._payload.isa[String]():
            return self._payload[String].copy()
        else:
            raise Error("DynValue: tag '", self._tag, "' carries no text")

    # --- evaluation ---------------------------------------------------------
    #
    # One evaluator per operation, each naming its kernel as a comptime
    # parameter. `_eval` below holds no switch, so the set of kernels a binary
    # links is exactly the set its expressions mention. See `EvalFn`.

    @staticmethod
    def _column(
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        return batch.column(payload[String]).copy()

    @staticmethod
    def _literal(
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        return payload[DynScalar].repeat(batch.num_rows())

    @staticmethod
    def _cast(
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        return cast_array(args[0].copy(), payload[DynType])

    @staticmethod
    def _date_trunc(
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        return DateTruncKernel.apply(
            args[0].copy(), CalendarUnit.parse(payload[String])
        )

    @staticmethod
    def _if_else(
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        return case_when_kernel(
            [args[0].as_bool().copy()],
            [args[1].copy()],
            Optional[DynArray](args[2].copy()),
        )

    @staticmethod
    def _is_in(
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        return IsInKernel.dispatch(args[0].copy(), payload[DynArray]).to_dyn()

    @staticmethod
    def _length(
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        return LengthKernel.dispatch(args[0].copy())

    @staticmethod
    def _unary[
        K: UnaryNumericKernel
    ](
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        return K.dispatch(args[0].copy())

    @staticmethod
    def _float_unary[
        K: UnaryFloatKernel
    ](
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        # `sqrt`/`exp`/`ln` are float64 in the fused lane (`FloatUnary`), but
        # these kernels dispatch over floating dtypes only — without the cast
        # `sqrt(int64_col)` raises here and computes there.
        return K.dispatch(_to_float(args[0].copy()))

    @staticmethod
    def _string_unary[
        K: StringMapKernel
    ](
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        return K.dispatch(args[0].copy())

    @staticmethod
    def _temporal[
        K: TemporalExtractKernel
    ](
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        return K.dispatch(args[0].copy())

    @staticmethod
    def _bool_unary[
        K: BoolUnaryKernel
    ](
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        return K.dispatch(args[0].copy())

    @staticmethod
    def _predicate[
        K: UnaryPredicateKernel
    ](
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        """`is_null`/`is_valid`/`is_nan`/`is_inf` — `array -> bool` over any dtype.

        Distinct from `_bool_unary`, whose kernels are `bool -> bool` and so
        require a `BoolArray` operand. These read the validity bitmap (the null
        predicates) or scan the values (the float predicates)."""
        return K.dispatch(args[0].copy())

    @staticmethod
    def _fill_null(
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        """`a` with its nulls taken from `fill`.

        `FillNullKernel` pins both operands to one dtype, and in this lane the
        two arrive independently typed — `col("x").fill_null(lit[Int64Type](0))`
        over an int32 column would raise on a mismatch the caller never wrote.
        Numeric operands are widened first, the same rule `_binary` and
        `_compare` use, so the two lanes agree on what the expression means."""
        var a = args[0].copy()
        var f = args[1].copy()
        if not a.dtype().is_string_like():
            _promote_operands(a, f)
        return fill_null_kernel(a, f)

    @staticmethod
    def _binary[
        K: BinaryNumericKernel
    ](
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        var l = args[0].copy()
        var r = args[1].copy()
        _promote_operands(l, r)
        return K.dispatch(l, r)

    @staticmethod
    def _add(
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        # `+` means *concatenate* when the operands turn out to be strings. The
        # choice cannot be made when the tree is built — an erased column has no
        # dtype until a schema is applied — so it is made here against the
        # materialized operands, the same shape `_compare` uses.
        var l = args[0].copy()
        var r = args[1].copy()
        if l.dtype().is_string_like():
            return ConcatKernel.dispatch(l, r)
        _promote_operands(l, r)
        return AddKernel.dispatch(l, r)

    @staticmethod
    def _float_binary[
        K: BinaryNumericKernel
    ](
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        var l = args[0].copy()
        var r = args[1].copy()
        _promote_float(l, r)
        return K.dispatch(l, r)

    @staticmethod
    def _pow(
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        var l = args[0].copy()
        var r = args[1].copy()
        _promote_float(l, r)
        return PowKernel.dispatch(l, r)

    @staticmethod
    def _compare[
        K: NumericCompareKernel, S: StringPredicateKernel
    ](
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        """One operator, two kernels: the runtime dtype picks.

        The operator is comptime — both halves of the pair are named at the call
        site — so `<` links the numeric *and* the string kernel and nothing
        else. Which of the two runs is the only runtime part.
        """
        var l = args[0].copy()
        var r = args[1].copy()
        if l.dtype().is_string_like():
            return S.dispatch(l, r)
        _promote_operands(l, r)
        return K.dispatch(l, r)

    @staticmethod
    def _bool_binary[
        K: BoolBinaryKernel
    ](
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        return K.dispatch(args[0].copy(), args[1].copy())

    @staticmethod
    def _string_binary[
        K: StringPredicateKernel
    ](
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        return K.dispatch(args[0].copy(), args[1].copy())

    @staticmethod
    def _conditional[
        K: BinaryConditionalKernel
    ](
        args: List[DynArray], payload: DynPayload, batch: RecordBatch
    ) raises -> DynArray:
        return K.combine(args[0].copy(), args[1].copy())

    def _eval(self, batch: RecordBatch) raises -> DynArray:
        """Evaluate this node: children first, then its own evaluator.

        There is no switch. Which kernel runs was decided when the node was
        built, by which `EvalFn` the constructing method named.

        The leaf case is spelled out rather than falling out of an empty loop:
        without it the compiler reads the recursion as unconditional and warns
        `self recursive call will cause an infinite loop`."""
        if len(self._kids) == 0:
            return self._eval_fn(List[DynArray](), self._payload, batch)
        else:
            var args = List[DynArray](capacity=len(self._kids))
            for i in range(len(self._kids)):
                args.append(self._kids[i][]._eval(batch))
            return self._eval_fn(args^, self._payload, batch)

    def materialize(self, batch: RecordBatch) raises -> Datum:
        return Datum(self._eval(batch))

    def execute(self, batch: RecordBatch) raises -> DynArray:
        """The column this expression produces — the relational engine's entry
        point, called once per morsel."""
        return self._eval(batch)

    # --- plan analysis ------------------------------------------------------
    def name(self) -> String:
        if self._tag == "column" and self._payload.isa[String]():
            return self._payload[String].copy()
        else:
            return String()

    def render(self) -> String:
        if len(self._kids) == 0:
            return self.name() if self._tag == "column" else self._tag.copy()
        var out = self._tag.copy() + "("
        for i in range(len(self._kids)):
            if i:
                out += ", "
            out += self._kids[i][].render()
        return out + ")"

    def referenced_columns(self) -> List[String]:
        if self._tag == "column":
            return [self.name()]
        var acc = List[String]()
        for i in range(len(self._kids)):
            var names = self._kids[i][].referenced_columns()
            for j in range(len(names)):
                var seen = False
                for k in range(len(acc)):
                    if acc[k] == names[j]:
                        seen = True
                        break
                if not seen:
                    acc.append(names[j].copy())
        return acc^

    def bound_column(self, schema: Schema) raises -> Int:
        """This expression's column position, or -1 if it is not a bare column.
        """
        if self._tag == "column":
            return schema.get_field_index(self._text())
        else:
            return -1

    def resolve_names(self, schema: Schema) raises -> Self:
        """Bind name references against `schema`.

        A plain recursive walk. The trampoline version had to hand back an
        *erased pointer* — a field whose function type mentioned `DynValue` made
        the struct recursive — and that workaround goes away with the children
        stored directly."""
        if self._tag == "column":
            _ = schema.get_field_index(self._text())
            return self.copy()
        var out = self.copy()
        for i in range(len(out._kids)):
            out._kids[i] = ArcPointer(self._kids[i][].resolve_names(schema))
        return out^

    def prune(self, stats: PruneStats) raises -> Interval:
        """What this expression's value can be, given per-column `[min, max]`.

        The tag twin of the `prune` overrides on the fused nodes — a column
        reports its bounds, a literal a point interval, a comparison the
        min/max rule its operator names, `and`/`or` the boolean combination.
        Every other tag falls through to "unknown", which is the conservative
        answer: a caller only ever skips data it has *proven* cannot match.

        This is what makes row-group and page skipping work for a predicate
        built at run time. Without it a `ParquetScanProcessor` holding a
        `col("x") > lit(1500)` decodes every row group — correct, and much
        slower.
        """
        var t = self._tag
        if t == "column":
            var iv = stats.by_name(self._text())
            return Interval.bounds(iv[0].copy(), iv[1].copy())
        elif t == "literal":
            var v = self._payload[DynScalar].copy()
            return Interval.bounds(Optional(v.copy()), Optional(v^))
        elif len(self._kids) == 2:
            # The rule for each operator is its interval kernel, shared with the
            # fused lane rather than restated here — and the tag is matched
            # against that kernel's own `name`, which is the same string the
            # factory tagged the node with. Neither the rule nor the spelling
            # can drift between the lanes.
            var l = self._kids[0][].prune(stats)
            var r = self._kids[1][].prune(stats)
            if t == AndInterval.name:
                return Interval.truth(AndInterval.apply(l, r))
            elif t == OrInterval.name:
                return Interval.truth(OrInterval.apply(l, r))
            elif t == LtInterval.name:
                return Interval.truth(LtInterval.apply(l, r))
            elif t == LeInterval.name:
                return Interval.truth(LeInterval.apply(l, r))
            elif t == GtInterval.name:
                return Interval.truth(GtInterval.apply(l, r))
            elif t == GeInterval.name:
                return Interval.truth(GeInterval.apply(l, r))
            elif t == EqInterval.name:
                return Interval.truth(EqInterval.apply(l, r))
            else:
                # `not_equal` and `xor` have no usable rule; everything else is
                # arithmetic or a payload op this never modelled.
                return Interval.unknown()
        else:
            return Interval.unknown()

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.render())

    # --- fluent surface -----------------------------------------------------
    #
    # Every method names its evaluator, and that name is the whole reason this
    # lane no longer links every kernel: `__sub__` mentions `SubKernel`, so a
    # program that never subtracts never elaborates it.
    def __add__(self, o: Self) -> Self:
        return Self("add", Self._add, self, o)

    def __sub__(self, o: Self) -> Self:
        return Self(SubKernel.name, Self._binary[SubKernel], self, o)

    def __mul__(self, o: Self) -> Self:
        return Self(MulKernel.name, Self._binary[MulKernel], self, o)

    def __truediv__(self, o: Self) -> Self:
        return Self(DivKernel.name, Self._float_binary[DivKernel], self, o)

    def __mod__(self, o: Self) -> Self:
        return Self(ModKernel.name, Self._binary[ModKernel], self, o)

    def __floordiv__(self, o: Self) -> Self:
        return Self(FloordivKernel.name, Self._binary[FloordivKernel], self, o)

    def __pow__(self, o: Self) -> Self:
        return Self(PowKernel.name, Self._pow, self, o)

    def __lt__(self, o: Self) -> Self:
        return Self(
            LtKernel.name, Self._compare[LtKernel, StringLtKernel], self, o
        )

    def __le__(self, o: Self) -> Self:
        return Self(
            LeKernel.name, Self._compare[LeKernel, StringLeKernel], self, o
        )

    def __gt__(self, o: Self) -> Self:
        return Self(
            GtKernel.name, Self._compare[GtKernel, StringGtKernel], self, o
        )

    def __ge__(self, o: Self) -> Self:
        return Self(
            GeKernel.name, Self._compare[GeKernel, StringGeKernel], self, o
        )

    def __eq__(self, o: Self) -> Self:
        return Self(
            EqKernel.name, Self._compare[EqKernel, StringEqKernel], self, o
        )

    def __ne__(self, o: Self) -> Self:
        return Self(
            NeKernel.name, Self._compare[NeKernel, StringNeKernel], self, o
        )

    def __and__(self, o: Self) -> Self:
        return Self(AndKernel.name, Self._bool_binary[AndKernel], self, o)

    def __or__(self, o: Self) -> Self:
        return Self(OrKernel.name, Self._bool_binary[OrKernel], self, o)

    def __xor__(self, o: Self) -> Self:
        return Self(XorKernel.name, Self._bool_binary[XorKernel], self, o)

    def __neg__(self) -> Self:
        return Self(NegKernel.name, Self._unary[NegKernel], self)

    def __invert__(self) -> Self:
        return Self(NotKernel.name, Self._bool_unary[NotKernel], self)

    def abs(self) -> Self:
        return Self(AbsKernel.name, Self._unary[AbsKernel], self)

    def sign(self) -> Self:
        return Self(SignKernel.name, Self._unary[SignKernel], self)

    def floor(self) -> Self:
        return Self(FloorKernel.name, Self._unary[FloorKernel], self)

    def ceil(self) -> Self:
        return Self(CeilKernel.name, Self._unary[CeilKernel], self)

    def round(self) -> Self:
        return Self(RoundKernel.name, Self._unary[RoundKernel], self)

    def sqrt(self) -> Self:
        return Self(SqrtKernel.name, Self._float_unary[SqrtKernel], self)

    def exp(self) -> Self:
        return Self(ExpKernel.name, Self._float_unary[ExpKernel], self)

    def ln(self) -> Self:
        return Self(LogKernel.name, Self._float_unary[LogKernel], self)

    def upper(self) -> Self:
        return Self(UpperKernel.name, Self._string_unary[UpperKernel], self)

    def lower(self) -> Self:
        return Self(LowerKernel.name, Self._string_unary[LowerKernel], self)

    def strip(self) -> Self:
        return Self(StripKernel.name, Self._string_unary[StripKernel], self)

    def lstrip(self) -> Self:
        return Self(LStripKernel.name, Self._string_unary[LStripKernel], self)

    def rstrip(self) -> Self:
        return Self(RStripKernel.name, Self._string_unary[RStripKernel], self)

    def reverse(self) -> Self:
        return Self(ReverseKernel.name, Self._string_unary[ReverseKernel], self)

    def capitalize(self) -> Self:
        return Self(
            CapitalizeKernel.name, Self._string_unary[CapitalizeKernel], self
        )

    def length(self) -> Self:
        return Self("length", Self._length, self)

    def isin(self, value_set: DynArray) -> Self:
        var out = Self("is_in", Self._is_in, self)
        out._payload = DynPayload(value_set.copy())
        return out^

    def startswith(self, o: Self) -> Self:
        return Self(
            StartsWithKernel.name,
            Self._string_binary[StartsWithKernel],
            self,
            o,
        )

    def endswith(self, o: Self) -> Self:
        return Self(
            EndsWithKernel.name, Self._string_binary[EndsWithKernel], self, o
        )

    def contains(self, o: Self) -> Self:
        return Self(
            ContainsKernel.name, Self._string_binary[ContainsKernel], self, o
        )

    def like(self, var pattern: String) -> Self:
        return Self(
            LikeKernel.name,
            Self._string_binary[LikeKernel],
            self,
            Self.literal(StringScalar(pattern^)),
        )

    def ilike(self, var pattern: String) -> Self:
        return Self(
            ILikeKernel.name,
            Self._string_binary[ILikeKernel],
            self,
            Self.literal(StringScalar(pattern^)),
        )

    def year(self) -> Self:
        return Self(YearKernel.name, Self._temporal[YearKernel], self)

    def month(self) -> Self:
        return Self(MonthKernel.name, Self._temporal[MonthKernel], self)

    def day(self) -> Self:
        return Self(DayKernel.name, Self._temporal[DayKernel], self)

    def hour(self) -> Self:
        return Self(HourKernel.name, Self._temporal[HourKernel], self)

    def minute(self) -> Self:
        return Self(MinuteKernel.name, Self._temporal[MinuteKernel], self)

    def second(self) -> Self:
        return Self(SecondKernel.name, Self._temporal[SecondKernel], self)

    def day_of_week(self) -> Self:
        return Self(DayOfWeekKernel.name, Self._temporal[DayOfWeekKernel], self)

    def quarter(self) -> Self:
        return Self(QuarterKernel.name, Self._temporal[QuarterKernel], self)

    def day_of_year(self) -> Self:
        return Self(DayOfYearKernel.name, Self._temporal[DayOfYearKernel], self)

    def coalesce(self, o: Self) -> Self:
        return Self(
            CoalesceKernel.name, Self._conditional[CoalesceKernel], self, o
        )

    def nullif(self, o: Self) -> Self:
        return Self(NullifKernel.name, Self._conditional[NullifKernel], self, o)

    def fill_null(self, o: Self) -> Self:
        return Self(FillNullKernel.name, Self._fill_null, self, o)

    # --- null / value predicates -------------------------------------------
    #
    # `Value` supplies `is_null`/`is_valid` as defaults returning the *fused*
    # `NullPredicate`, and those are what a `[V: Value]` caller gets. This lane
    # overrides them so the result is another tag node: a predicate that left the
    # lane could not be combined with one that stayed in it —
    # `col("a").is_null() | (col("b") > lit[Int64Type](1))` needs both sides to
    # be the same type, and `BoolValue.__or__` will not take a `DynValue`.
    def is_null(self) -> Self:
        return Self(IsNullKernel.name, Self._predicate[IsNullKernel], self)

    def is_valid(self) -> Self:
        return Self(NotNullKernel.name, Self._predicate[NotNullKernel], self)

    def is_nan(self) -> Self:
        return Self(IsNanKernel.name, Self._predicate[IsNanKernel], self)

    def is_inf(self) -> Self:
        return Self(IsInfKernel.name, Self._predicate[IsInfKernel], self)

    def date_trunc(self, var unit: String) -> Self:
        var out = Self("date_trunc", Self._date_trunc, self)
        out._payload = DynPayload(unit^)
        return out^

    def cast(self, to: DynType) -> Self:
        var out = Self("cast", Self._cast, self)
        out._payload = DynPayload(to.copy())
        return out^

    # --- leaves -------------------------------------------------------------
    @staticmethod
    def column(var name: String) -> Self:
        return Self("column", Self._column, DynPayload(name^))

    @staticmethod
    def literal(var value: DynScalar) -> Self:
        return Self("literal", Self._literal, DynPayload(value^))

    @staticmethod
    def if_else(cond: Self, then_: Self, else_: Self) -> Self:
        """Element-wise conditional — the single-branch `CaseWhen`.

        A static factory rather than a free function in `values.mojo` because
        the evaluator it names is private to this struct; the free `if_else`
        there is a one-line forward."""
        return Self("if_else", Self._if_else, cond, then_, else_)

    # --- aggregates: named here too, resolved against the input dtype -------
    def aggregate(self, var func: String) -> DynAgg:
        return DynAgg(func^, self.copy())

    def sum(self) -> DynAgg:
        return self.aggregate("sum")

    def mean(self) -> DynAgg:
        return self.aggregate("mean")

    def product(self) -> DynAgg:
        return self.aggregate("product")

    def min(self) -> DynAgg:
        return self.aggregate("min")

    def max(self) -> DynAgg:
        return self.aggregate("max")

    def count(self) -> DynAgg:
        return self.aggregate("count")


# ---------------------------------------------------------------------------


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
