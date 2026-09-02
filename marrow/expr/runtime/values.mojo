"""The runtime lane: expressions whose structure lives in fields.

One struct — a tag, its children behind `ArcPointer`, and an optional payload.
Where the comptime lane puts a whole subtree in a type and fuses it into one
SIMD loop, this materialises a `DynArray` **per node, per morsel**.

That is the trade, and it is why the two lanes exist rather than one:

| | comptime | runtime |
|---|---|---|
| structure | in the type | in the fields |
| per node | inlined into one loop | one materialised column |
| built from | Mojo source | anything, including Python at run time |

The runtime lane is what a frontend uses when the query is not known until the
program runs, which is every frontend that is not the Mojo DSL. It is also why
this package has a box at all: a plan holds either lane through `DynValue`, and
that mixing is what keeps an AOT binary off this file entirely. The measured
gap between the two lanes is several times the binary; the current figures live
in `benchmarks/binary_size/` rather than here, so they cannot go stale in a
docstring.

**The tag selects the kernel, and that is deliberate.** `evaluate` switches on
`_tag`, so this file is an interpreter. A program that builds expressions at
run time has already accepted one — it cannot know its kernels at compile time,
and a frontend constructing queries dynamically reaches most of them anyway.
The cost is paid only by binaries that use this lane at all, and the comptime
lane never reaches it.

**Do not replace the switch with a per-node function pointer.** That design
existed, put a thin `fn` field in this self-referential struct, and the compiler
miscompiled it. See `docs/backlog.md`; this docstring described that removed
design, in the present tense, long after it was gone.
"""

from std.memory import ArcPointer
from std.utils import Variant

from ...arrays import (
    BoolArray,
    DynArray,
    Int64Array,
    StringArray,
    StructArray,
)
from ...kernels.aggregate import (
    APPROX_COUNT_DISTINCT,
    COUNT,
    COUNT_DISTINCT,
    MAX,
    MEAN,
    MIN,
    PRODUCT,
    STDDEV,
    STDDEV_SAMP,
    SUM,
    VARIANCE,
    VAR_SAMP,
)
from ...kernels.boolean import (
    AndKernel,
    IsInfKernel,
    IsNanKernel,
    IsNullKernel,
    NotKernel,
    NotNullKernel,
    OrKernel,
    XorKernel,
)
from ...kernels.cast import cast as cast_array
from ...kernels.conditional import case_when as case_when_kernel
from ...kernels.conditional import coalesce as coalesce_kernel
from ...kernels.conditional import fill_null as fill_null_kernel
from ...kernels.conditional import nullif as nullif_kernel
from ...kernels.membership import IsInKernel
from ...kernels.nested import ArrayLengthKernel
from ...kernels.numeric import (
    AbsKernel,
    AddKernel,
    BinaryKernel,
    CeilKernel,
    DivKernel,
    ExpKernel,
    FloorKernel,
    FloordivKernel,
    LogKernel,
    ModKernel,
    MulKernel,
    NegKernel,
    NumericCompareKernel,
    PowKernel,
    RoundKernel,
    SignKernel,
    SqrtKernel,
    SubKernel,
    TruncKernel,
    UnaryFloatKernel,
    EqKernel,
    GeKernel,
    GtKernel,
    LeKernel,
    LtKernel,
    NeKernel,
)
from ...kernels.string import (
    AsciiKernel,
    CapitalizeKernel,
    CharLengthKernel,
    ConcatKernel,
    ContainsKernel,
    EndsWithKernel,
    ILikeKernel,
    LPadKernel,
    LStripKernel,
    LeftKernel,
    LengthKernel,
    LikeKernel,
    LowerKernel,
    PositionKernel,
    RPadKernel,
    RStripKernel,
    RepeatKernel,
    ReplaceKernel,
    ReverseKernel,
    RightKernel,
    SplitPartKernel,
    StartsWithKernel,
    StringOperands,
    StringPredicateKernel,
    StripKernel,
    SubstrKernel,
    TrimCharsKernel,
    UpperKernel,
    StringEqKernel,
    StringGeKernel,
    StringGtKernel,
    StringLeKernel,
    StringLtKernel,
    StringNeKernel,
)
from ...kernels.temporal import (
    CalendarUnit,
    DateTruncKernel,
    DayKernel,
    DayNameKernel,
    DayOfWeekKernel,
    DayOfYearKernel,
    EpochKernel,
    HourKernel,
    IsoYearKernel,
    LastDayKernel,
    MinuteKernel,
    MonthKernel,
    MonthNameKernel,
    QuarterKernel,
    SecondKernel,
    WeekKernel,
    YearKernel,
)
from ...dtypes import DynType, bool_, float64, int64, string
from ...scalars import BoolScalar, DynScalar
from ...schema import Schema
from ...tabular import RecordBatch
from ..logical import DynValue, Shape, Value, merged
from ..bindings import Bindings
from ...kernels.bounds import (
    EqBounds,
    GeBounds,
    GtBounds,
    LeBounds,
    LtBounds,
    NeBounds,
)
from ..pruning import DynBounds, PruneStats, Prunable, Truth, compare_dyn
from ..physical import Datum
from ..physical import Evaluable, DynOperator, EvalOperator
from .aggregates import RuntimeAggregate


def promote_dyn(l: DynType, r: DynType) raises -> DynType:
    """The common numeric domain of two runtime dtypes.

    The runtime twin of `comptime/rules.mojo`'s `promote[L, R]`, and it states
    the same two rules: a float outranks any integer whatever the widths, and
    otherwise the wider one wins. It lives here rather than beside its twin
    because `rules.mojo` belongs to the comptime lane -- reaching across would
    make this file import the package it exists to stay out of.

    **Widening, not "try a cast each way".** The earlier rule cast the right
    operand to the left's type and fell back to the reverse on failure, which
    reads as symmetric and is not: `cast` accepts a *narrowing* integer
    conversion and only rejects it per value, so `int32_col > lit(2**40)`
    tried `int64 -> int32` first, succeeded structurally, and raised on the
    literal instead of comparing. Widening cannot narrow, so there is no
    ordering to get wrong.
    """
    if l == r:
        return l.copy()
    if not l.is_numeric() or not r.is_numeric():
        raise Error("promote: no common numeric type for ", l, " and ", r)
    var l_float = l.is_floating_point()
    if l_float != r.is_floating_point():
        return l.copy() if l_float else r.copy()
    # `byte_width`, not `bit_width`: `DynType` has only the former, and the
    # ordering it induces over the numeric types is the same one.
    return l.copy() if l.byte_width() >= r.byte_width() else r.copy()


comptime Payload = Variant[NoneType, String, DynType, DynArray, DynScalar]
"""What a node carries besides its children — a column name, a cast target, a
literal, an `IsIn` value set.

A closed variant rather than an erased box: the set is small, known, and adding
to it should be a decision someone makes rather than something a caller can do
from outside.
"""


struct RuntimeValue(Evaluable, Movable, Prunable, Value):
    """A runtime-built expression.

    Satisfies `Value` — `Analyzable & Executable & Writable & Copyable &
    Deinitable` — and nothing else. It cannot satisfy `ComptimeValue`, whose
    `Type` and `Bound` it has no way to supply: its type is not known until a
    schema is in hand, and it has no per-batch bound state because it does not
    fuse.

    That is the rule the erasure follows: **erase into a trait of methods,
    never into one with comptime members you cannot supply.**
    """

    comptime shape = Shape.columnar
    """Always columnar — `evaluate` returns a `DynArray` unconditionally.

    The comptime lane can be `Shape.scalar` (a literal stays lazy); this lane
    materialises, so it answers truthfully rather than aspirationally.
    """

    var _tag: String
    """How this node prints and prunes. Never how it evaluates."""

    var _kids: List[ArcPointer[Self]]
    """Children behind `ArcPointer`: a bare `List[Self]` is rejected
    (`field '_kids' has non-implicitly deletable type`), and the indirection
    makes copying a subtree O(1)."""

    var _payload: Payload

    # -- construction -------------------------------------------------------

    def __init__(out self, var tag: String, var payload: Payload):
        self._tag = tag^
        self._kids = List[ArcPointer[Self]]()
        self._payload = payload^

    def __init__(out self, var tag: String, a: Self):
        self._tag = tag^
        self._kids = [ArcPointer(a.copy())]
        self._payload = Payload(NoneType())

    def __init__(out self, var tag: String, a: Self, var payload: Payload):
        """One child *and* a payload — `cast`, `like`, `date_trunc`, `isin`.

        The payload-only constructor above builds a leaf, so without this a
        node whose operand is an expression and whose second operand is a
        constant the kernel takes directly (a target dtype, a compiled
        pattern, a value set) would have to smuggle the constant through a
        `literal` child and unpack it again at evaluation. That would make the
        constant a `DynArray` broadcast to the batch length — one allocation
        per morsel for a value the kernel wants as a scalar.
        """
        self._tag = tag^
        self._kids = [ArcPointer(a.copy())]
        self._payload = payload^

    def __init__(out self, var tag: String, var kids: List[Self]):
        """N children. `if_else` needs three and `case_when` needs `2n + 1`, so
        the fixed-arity constructors below do not cover everything."""
        self._tag = tag^
        self._kids = List[ArcPointer[Self]](capacity=len(kids))
        for ref k in kids:
            self._kids.append(ArcPointer(k.copy()))
        self._payload = Payload(NoneType())

    def __init__(out self, var tag: String, a: Self, b: Self):
        self._tag = tag^
        self._kids = [ArcPointer(a.copy()), ArcPointer(b.copy())]
        self._payload = Payload(NoneType())

    # -- Value --------------------------------------------------------------

    def prune(self, stats: PruneStats, bindings: Bindings) -> Truth:
        """The runtime lane's half of the same contract.

        The tag selects between six one-line `decide` bodies, not between
        kernels: no kernel code is linked by this, so the module's "a tag never
        selects a kernel" rule is not what is at stake here. The dtype ladder
        lives once, in `_ord`, and the readings are the same six the fused lane
        runs — writing them twice is how two lanes drift into disagreeing about
        which row groups to skip.
        """
        if len(self._kids) == 2:
            if self._tag == "and":
                return self._kids[0][].prune(stats, bindings) & self._kids[
                    1
                ][].prune(stats, bindings)
            if self._tag == "or":
                return self._kids[0][].prune(stats, bindings) | self._kids[
                    1
                ][].prune(stats, bindings)
            var l = self._kids[0][]._bounds(stats, bindings)
            var r = self._kids[1][]._bounds(stats, bindings)
            if self._tag == "lt":
                return compare_dyn[LtBounds](l, r)
            if self._tag == "le":
                return compare_dyn[LeBounds](l, r)
            if self._tag == "gt":
                return compare_dyn[GtBounds](l, r)
            if self._tag == "ge":
                return compare_dyn[GeBounds](l, r)
            if self._tag == "eq":
                return compare_dyn[EqBounds](l, r)
            if self._tag == "ne":
                return compare_dyn[NeBounds](l, r)
        return Truth.maybe

    def _bounds(self, stats: PruneStats, bindings: Bindings) -> DynBounds:
        """A leaf's bounds, left erased — this lane has no comptime type to
        unwrap into. Anything composite answers unknown."""
        if len(self._kids) == 0:
            if self._tag == "column" and self._payload.isa[String]():
                return stats.dyn_bounds(self._payload[String])
            if self._tag == "literal" and self._payload.isa[DynScalar]():
                return DynBounds.point(self._payload[DynScalar].copy())
        return DynBounds.unknown()

    def columns(self) -> List[String]:
        # The leaf case is spelled out rather than falling out of an empty
        # loop: without it the compiler reads the recursion below as
        # unconditional and warns `self recursive call will cause an infinite
        # loop`. Same reason `evaluate` has its own early return.
        if len(self._kids) == 0:
            if self._tag == "column" and self._payload.isa[String]():
                return [self._payload[String].copy()]
            return List[String]()

        var out = List[String]()
        if self._tag == "column" and self._payload.isa[String]():
            out.append(self._payload[String].copy())
        for ref kid in self._kids:
            out = merged(out^, kid[].columns())
        return out^

    def constant_bool(self) -> Optional[Bool]:
        """A non-null boolean literal answers; everything else does not.

        Shares `_bool_const`'s rule about nulls: `NULL` is not a constant this
        may fold on, because a filter keeps rows where the predicate is `TRUE`
        and a null predicate is not `FALSE` — it merely fails to select.
        """
        return _bool_const(self)

    def name(self) -> String:
        if self._tag == "column" and self._payload.isa[String]():
            return self._payload[String].copy()
        if self._tag == "literal" and self._payload.isa[DynScalar]():
            return String(self._payload[DynScalar])
        return String()

    def dtype(self, schema: Schema) raises -> DynType:
        """The type this expression produces over `schema`.

        A column reference looks itself up; a literal knows its own scalar.
        Anything computed is answered by **evaluating against a zero-row
        batch** — the only lane that has to, because a runtime node's output
        type follows from a function pointer, and recovering it statically
        would mean a promotion rule per tag, which is a second dispatch table
        keyed on the thing that must never select behaviour.

        the previous expression package probed *every* expression this way,
        comptime ones included.
        Here the comptime lane answers from `Type` for free and only this lane
        pays, which is the asymmetry worth having.
        """
        if self._tag == "column" and self._payload.isa[String]():
            var i = schema.get_field_index(self._payload[String])
            if i == -1:
                raise Error(
                    "column '", self._payload[String], "' not found in schema"
                )
            return schema.fields[i].dtype.copy()
        if self._tag == "literal" and self._payload.isa[DynScalar]():
            return self._payload[DynScalar].type()
        return (
            self.evaluate(
                RecordBatch.empty(schema).to_struct_array(), Bindings()
            )
            .to_array(0)
            .dtype()
        )

    def to_operator(
        self, schema: Schema, grouped: Bool, bindings: Bindings = Bindings()
    ) raises -> DynOperator:
        """The runtime lane's half of the same contract. Its operator is the
        same adapter the comptime lane uses — the lanes differ in how they
        compute, not in how they are turned into something that runs.

        `bindings` is carried for symmetry and reaches nothing: this lane has
        no `Param` node, and a `RuntimeValue`'s children are `RuntimeValue`s.
        """
        return EvalOperator[Self](self.copy(), bindings.copy())

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        """Interpret this node.

        A switch on `_tag`, not a per-node function pointer. The pointer bought
        pay-per-use kernel linking — a binary linked only the kernels its
        expressions named — but it also put a thin fn field in a
        self-referential struct that already holds a nested variant, which is
        miscompiled. The switch costs binary size *in binaries that use this
        lane at all*; the typed API builds comptime nodes and never reaches it.
        """
        # Leaves spelled out before the recursion — see `columns`. Without the
        # guard the compiler reads the recursion below as unconditional and
        # warns `self recursive call will cause an infinite loop`.
        if len(self._kids) == 0:
            if self._tag == "column":
                return Datum(batch.field(self._payload[String]).copy())
            if self._tag == "literal":
                return Datum(self._payload[DynScalar].repeat(len(batch)))
            raise Error("evaluate: unknown runtime leaf '", self._tag, "'")

        var kids = List[DynArray](capacity=len(self._kids))
        for ref kid in self._kids:
            kids.append(kid[].evaluate(batch, bindings).to_array(len(batch)))

        if len(kids) == 1:
            var d = self._unary(kids[0].copy())
            if d:
                return Datum(d.take())
        # Comparisons and boolean connectives. The `_tag` selects the kernel
        # here, which the comptime lane never does -- see this module's
        # docstring on why that trade is right for the interpreted lane.
        if len(kids) == 2:
            var l = kids[0].copy()
            var r = kids[1].copy()
            if self._tag == "eq":
                return Datum(Self._compare[EqKernel, StringEqKernel](l^, r^))
            if self._tag == "ne":
                return Datum(Self._compare[NeKernel, StringNeKernel](l^, r^))
            if self._tag == "lt":
                return Datum(Self._compare[LtKernel, StringLtKernel](l^, r^))
            if self._tag == "le":
                return Datum(Self._compare[LeKernel, StringLeKernel](l^, r^))
            if self._tag == "gt":
                return Datum(Self._compare[GtKernel, StringGtKernel](l^, r^))
            if self._tag == "ge":
                return Datum(Self._compare[GeKernel, StringGeKernel](l^, r^))
            if self._tag == "and":
                return Datum(AndKernel.dispatch(l^, r^))
            if self._tag == "or":
                return Datum(OrKernel.dispatch(l^, r^))
            if self._tag == "xor":
                return Datum(XorKernel.dispatch(l^, r^))
            var d = self._binary(kids[0].copy(), kids[1].copy())
            if d:
                return Datum(d.take())

        # The SQL string functions, whose arity is two *or* three -- so they
        # fit neither `_binary` nor `_unary` and get their own ladder.
        if len(kids) == 2 or len(kids) == 3:
            var d = self._string_fn(kids)
            if d:
                return Datum(d.take())

        if self._tag == "coalesce":
            return Datum(coalesce_kernel(Self._unified(kids^)))
        if self._tag == "case_when":
            var has_else = len(kids) % 2 == 1
            var n = (len(kids) - 1) // 2 if has_else else len(kids) // 2
            var conds = List[BoolArray](capacity=n)
            for i in range(n):
                conds.append(kids[i].as_bool().copy())
            # The conditions are bool and the branches are whatever the caller
            # wrote, so only the branches are unified -- and the else arm is
            # one of them, which is why it goes through the same list.
            var branches = List[DynArray](capacity=n + 1)
            for i in range(n):
                branches.append(kids[n + i].copy())
            if has_else:
                branches.append(kids[len(kids) - 1].copy())
            branches = Self._unified(branches^)
            var otherwise = Optional[DynArray](None)
            if has_else:
                otherwise = branches[n].copy()
            var vals = List[DynArray](capacity=n)
            for i in range(n):
                vals.append(branches[i].copy())
            return Datum(case_when_kernel(conds, vals, otherwise^))
        raise Error("evaluate: unknown runtime node '", self._tag, "'")

    @staticmethod
    def _compare[
        K: NumericCompareKernel, S: StringPredicateKernel
    ](var l: DynArray, var r: DynArray) raises -> DynArray:
        """One operator, two kernels: the runtime dtype picks which runs.

        Both halves are named at the call site, so a binary links the numeric
        *and* the string kernel for every comparison its expressions mention --
        and nothing else. Mixed numeric widths are promoted to the wider domain
        first, so `int32_col > int64_lit` compares rather than raising.
        """
        if l.dtype().is_string_like() or r.dtype().is_string_like():
            if not (l.dtype().is_string_like() and r.dtype().is_string_like()):
                raise Error(
                    "compare: cannot compare ", l.dtype(), " with ", r.dtype()
                )
            return S.dispatch(l^, r^)
        if l.dtype() == r.dtype():
            return K.dispatch(l^, r^)
        if l.dtype().is_numeric() and r.dtype().is_numeric():
            var to = promote_dyn(l.dtype(), r.dtype())
            return K.dispatch(cast_array(l^, to), cast_array(r^, to))
        # Anything else -- two temporal columns at different resolutions, say
        # -- meets on the left operand's type. `promote_dyn` deliberately has
        # no answer outside the numeric domain, and inventing one here would
        # be a second promotion rule that could disagree with it.
        var to = l.dtype()
        return K.dispatch(l^, cast_array(r^, to))

    @staticmethod
    def _arith[
        K: BinaryKernel
    ](var l: DynArray, var r: DynArray) raises -> DynArray:
        """`+ - * // %` — both operands promoted to their common domain first.

        `promote_dyn` is the same rule the comptime lane's `promote[L, R]`
        applies at compile time. Doing it here rather than letting the kernel's
        `expect_same_dtype` raise is what makes `int32_col + int64_lit` add
        instead of failing, and doing it *by widening* rather than by trying a
        cast in each direction is what keeps it from silently narrowing.
        """
        if not l.dtype().is_numeric() or not r.dtype().is_numeric():
            raise Error(
                "arithmetic is not defined for ",
                l.dtype(),
                " and ",
                r.dtype(),
            )
        var to = promote_dyn(l.dtype(), r.dtype())
        return K.dispatch(cast_array(l^, to), cast_array(r^, to))

    @staticmethod
    def _float_binary[
        K: BinaryKernel
    ](var l: DynArray, var r: DynArray) raises -> DynArray:
        """`/` and `**` — always `float64`, whatever went in.

        The runtime half of `FloatBinary`: both operands are cast up *before*
        the kernel, so `5 / 2` is 2.5 rather than an integer division widened
        afterwards. See that struct on why marrow diverges from `pc.divide`
        here.
        """
        var to = DynType(float64)
        return K.dispatch(cast_array(l^, to), cast_array(r^, to))

    @staticmethod
    def _float_unary[K: UnaryFloatKernel](var a: DynArray) raises -> DynArray:
        """`sqrt` / `exp` / `ln` — the runtime half of `FloatUnary`."""
        return K.dispatch(cast_array(a^, DynType(float64)))

    def _unary(self, var a: DynArray) raises -> Optional[DynArray]:
        """The one-operand tags, or `None` when this node is not one of them.

        Split out of `evaluate` rather than inlined into it because that method
        is already the module's one interpreter loop and forty more `if`s in it
        would bury the three shapes that are *not* a plain kernel call
        (`coalesce`, `case_when`, the promotion in `_compare`).

        `Optional` rather than a raise, so `evaluate` keeps ownership of the
        one error message that names the unknown tag.
        """
        if self._tag == "not":
            return NotKernel.dispatch(a^)
        if self._tag == "neg":
            return NegKernel.dispatch(a^)
        if self._tag == "abs":
            return AbsKernel.dispatch(a^)
        if self._tag == "sign":
            return SignKernel.dispatch(a^)
        if self._tag == "floor":
            return FloorKernel.dispatch(a^)
        if self._tag == "ceil":
            return CeilKernel.dispatch(a^)
        if self._tag == "round":
            return RoundKernel.dispatch(a^)
        if self._tag == "trunc":
            return TruncKernel.dispatch(a^)
        if self._tag == "sqrt":
            return Self._float_unary[SqrtKernel](a^)
        if self._tag == "exp":
            return Self._float_unary[ExpKernel](a^)
        if self._tag == "ln":
            return Self._float_unary[LogKernel](a^)

        # Null and value predicates. `is_null`/`is_valid` read the validity
        # bitmap and are never null themselves; `is_nan`/`is_inf` read the
        # values and are null where the input is -- the same split the
        # comptime lane draws between `NullPredicate` and `ValuePredicate`.
        if self._tag == "is_null":
            return IsNullKernel.apply(a).to_dyn()
        if self._tag == "is_valid":
            return NotNullKernel.apply(a).to_dyn()
        if self._tag == "is_nan":
            return IsNanKernel.apply(a).to_dyn()
        if self._tag == "is_inf":
            return IsInfKernel.apply(a).to_dyn()

        if self._tag == "upper":
            return UpperKernel.dispatch(a^)
        if self._tag == "lower":
            return LowerKernel.dispatch(a^)
        if self._tag == "strip":
            return StripKernel.dispatch(a^)
        if self._tag == "lstrip":
            return LStripKernel.dispatch(a^)
        if self._tag == "rstrip":
            return RStripKernel.dispatch(a^)
        if self._tag == "reverse":
            return ReverseKernel.dispatch(a^)
        if self._tag == "capitalize":
            return CapitalizeKernel.dispatch(a^)
        if self._tag == "length":
            return LengthKernel.dispatch(a^)
        if self._tag == "array_length":
            return ArrayLengthKernel.dispatch(a^)

        # The SQL function surface's *nullary* half — the functions whose only
        # operand is the column, so their `StringOperands` is empty.
        #
        # Their argument-carrying siblings reach this lane too, and the way
        # they do is load-bearing: **an argument is a child node, never a
        # `Payload` member.** `Payload` already holds a `DynScalar`, which
        # makes it the `Variant` shape that loses every other element of a
        # `List` when it grows (CLAUDE.md, "Mojo Gotchas"), so an argument
        # slot there would walk straight into it. A child costs nothing
        # extra — this lane has carried children since it was written. See
        # `_string_fn` below.
        if self._tag == "char_length":
            return CharLengthKernel.dispatch(a^, StringOperands())
        if self._tag == "ascii":
            return AsciiKernel.dispatch(a^, StringOperands())

        if self._tag == "week":
            return WeekKernel.dispatch(a^)
        if self._tag == "iso_year":
            return IsoYearKernel.dispatch(a^)
        if self._tag == "epoch":
            return EpochKernel.dispatch(a^)
        if self._tag == "last_day":
            return LastDayKernel.dispatch(a^)
        if self._tag == "day_name":
            return DayNameKernel.dispatch(a^)
        if self._tag == "month_name":
            return MonthNameKernel.dispatch(a^)

        if self._tag == "year":
            return YearKernel.dispatch(a^)
        if self._tag == "month":
            return MonthKernel.dispatch(a^)
        if self._tag == "day":
            return DayKernel.dispatch(a^)
        if self._tag == "hour":
            return HourKernel.dispatch(a^)
        if self._tag == "minute":
            return MinuteKernel.dispatch(a^)
        if self._tag == "second":
            return SecondKernel.dispatch(a^)
        if self._tag == "quarter":
            return QuarterKernel.dispatch(a^)
        if self._tag == "day_of_week":
            return DayOfWeekKernel.dispatch(a^)
        if self._tag == "day_of_year":
            return DayOfYearKernel.dispatch(a^)

        # The payload-carrying unary nodes: the second operand is a constant
        # the kernel takes directly rather than a column it has to broadcast.
        if self._tag == "like":
            return LikeKernel.dispatch(a, self._payload[String])
        if self._tag == "ilike":
            return ILikeKernel.dispatch(a, self._payload[String])
        if self._tag == "date_trunc":
            return DateTruncKernel.apply(
                a, CalendarUnit.parse(self._payload[String])
            )
        if self._tag == "isin":
            return IsInKernel.dispatch(
                a, self._payload[DynArray].copy()
            ).to_dyn()
        if self._tag == "cast" or self._tag == "cast_unsafe":
            return cast_array(
                a^, self._payload[DynType].copy(), self._tag == "cast"
            )
        return None

    def _binary(
        self, var l: DynArray, var r: DynArray
    ) raises -> Optional[DynArray]:
        """The two-operand tags `evaluate` does not spell out itself."""
        if self._tag == "add":
            # `+` over erased operands cannot know at build time whether it
            # means addition or concatenation, so the runtime dtype decides --
            # which is the call `ConcatKernel.dispatch` exists for.
            if l.dtype().is_string_like():
                return ConcatKernel.dispatch(l^, r^)
            return Self._arith[AddKernel](l^, r^)
        if self._tag == "sub":
            return Self._arith[SubKernel](l^, r^)
        if self._tag == "mul":
            return Self._arith[MulKernel](l^, r^)
        if self._tag == "floordiv":
            return Self._arith[FloordivKernel](l^, r^)
        if self._tag == "mod":
            return Self._arith[ModKernel](l^, r^)
        if self._tag == "truediv":
            return Self._float_binary[DivKernel](l^, r^)
        if self._tag == "pow":
            return Self._float_binary[PowKernel](l^, r^)

        if self._tag == "startswith":
            return StartsWithKernel.dispatch(l^, r^)
        if self._tag == "endswith":
            return EndsWithKernel.dispatch(l^, r^)
        if self._tag == "contains":
            return ContainsKernel.dispatch(l^, r^)

        # `NullifKernel` and `FillNullKernel` both call `expect_same_dtype`,
        # so a mixed pair has to meet before the kernel sees it -- otherwise
        # `float_col.fill_null(0)` raises on a literal that infers `int64`.
        if self._tag == "nullif":
            var pair = Self._unified([l^, r^])
            return nullif_kernel(pair[0], pair[1])
        if self._tag == "fill_null":
            var pair = Self._unified([l^, r^])
            return fill_null_kernel(pair[0], pair[1])
        return None

    def _string_fn(self, kids: List[DynArray]) raises -> Optional[DynArray]:
        """The SQL string functions whose arguments are operands.

        Two children or three, so they fit neither `_unary` nor `_binary`.
        Every one of them was **absent from this lane** while its arguments
        were constants on the node: a `RuntimeValue` has no way to spell a
        constant a kernel takes directly except through `Payload`, and
        widening that `Variant` is the change the tree works around rather
        than makes. An argument is a child now, and children are the one thing
        this struct has always carried.

        The operands are normalised to `utf8` and `int64` — the widths
        `StringOperands`' defaults name — so this file instantiates the carrier
        **once**. The comptime lane fills the same four slots with its own
        types and casts nothing; paying a cast here is the same trade
        `_arith` and `_compare` already make, and it keeps a
        stringlike-by-integer dispatch cross product out of the interpreter.
        """
        if self._tag == "left":
            var ops = StringOperands()
            ops.count = Self._as_int64(kids[1])
            return LeftKernel.dispatch(kids[0], ops)
        if self._tag == "right":
            var ops = StringOperands()
            ops.count = Self._as_int64(kids[1])
            return RightKernel.dispatch(kids[0], ops)
        if self._tag == "repeat":
            var ops = StringOperands()
            ops.count = Self._as_int64(kids[1])
            return RepeatKernel.dispatch(kids[0], ops)
        if self._tag == "trim_chars":
            var ops = StringOperands()
            ops.text = Self._as_text(kids[1])
            return TrimCharsKernel.dispatch(kids[0], ops)
        if self._tag == "position":
            var ops = StringOperands()
            ops.text = Self._as_text(kids[1])
            return PositionKernel.dispatch(kids[0], ops)
        if len(kids) == 3:
            if self._tag == "substr":
                var ops = StringOperands()
                ops.start = Self._as_int64(kids[1])
                ops.count = Self._as_int64(kids[2])
                return SubstrKernel.dispatch(kids[0], ops)
            if self._tag == "lpad":
                var ops = StringOperands()
                ops.count = Self._as_int64(kids[1])
                ops.text = Self._as_text(kids[2])
                return LPadKernel.dispatch(kids[0], ops)
            if self._tag == "rpad":
                var ops = StringOperands()
                ops.count = Self._as_int64(kids[1])
                ops.text = Self._as_text(kids[2])
                return RPadKernel.dispatch(kids[0], ops)
            if self._tag == "replace":
                var ops = StringOperands()
                ops.text = Self._as_text(kids[1])
                ops.alt = Self._as_text(kids[2])
                return ReplaceKernel.dispatch(kids[0], ops)
            if self._tag == "split_part":
                var ops = StringOperands()
                ops.text = Self._as_text(kids[1])
                ops.count = Self._as_int64(kids[2])
                return SplitPartKernel.dispatch(kids[0], ops)
        return None

    @staticmethod
    def _as_text(a: DynArray) raises -> StringArray:
        """An argument operand as `utf8`. Identity for a `utf8` column —
        `cast` returns the input when the dtypes already agree."""
        return cast_array(a, DynType(string)).as_string().copy()

    @staticmethod
    def _as_int64(a: DynArray) raises -> Int64Array:
        """An argument operand as `int64`, the width `lit(2)` already infers."""
        return cast_array(a, DynType(int64)).as_int64().copy()

    @staticmethod
    def _unified(var arrays: List[DynArray]) raises -> List[DynArray]:
        """Cast a set of *alternatives* to one common type.

        The n-ary counterpart of `_arith`'s promotion, for the conditionals --
        `coalesce`, `case_when` and the two-armed `nullif` / `fill_null`. Each
        of those kernels picks one of its inputs per row and therefore needs
        them to agree on a dtype, which two independently written expressions
        will not: `coalesce(int_col, lit(0.5))` is an ordinary thing to write.

        A non-numeric mismatch is left alone rather than guessed at, so the
        kernel raises and its message names both dtypes. That is the honest
        answer for `coalesce(string_col, int_col)`, which has no common type.
        """
        var to = arrays[0].dtype()
        var mixed = False
        for ref a in arrays:
            if a.dtype() != to:
                mixed = True
        if not mixed:
            return arrays^
        for ref a in arrays:
            if not a.dtype().is_numeric():
                return arrays^
        for ref a in arrays:
            to = promote_dyn(to, a.dtype())
        var out = List[DynArray](capacity=len(arrays))
        for ref a in arrays:
            out.append(cast_array(a.copy(), to))
        return out^

    # -- Writable -----------------------------------------------------------

    # -- the aggregate surface ----------------------------------------------
    #
    # The mirror of `ComptimeValue`'s, and that symmetry is the point: the same
    # fluent expression works in either lane, chosen by whether the caller knew
    # a dtype.
    #
    #     col("s", string).count_distinct()   # comptime, operand fuses
    #     col("s").count_distinct()           # runtime,  operand erased
    #
    # Every one of them raises, because `RuntimeAggregate` validates its name
    # in `__init__` — which is what makes an unknown aggregate impossible to
    # build from here, and keeps each verb's name literal in exactly one place.
    #
    # **The semantics are documented on the kernels, once.** These verbs, the
    # comptime lane's, and the kernels themselves were three places saying what
    # `min` does, and they had already drifted -- `min` was described three
    # different ways. A verb here names the SQL form and the kernel that
    # answers it; what the kernel *does* belongs to the kernel.

    def sum(self) raises -> RuntimeAggregate:
        """`SUM(self)` — see `SumFold`."""
        return RuntimeAggregate(self.copy(), String(SUM))

    def product(self) raises -> RuntimeAggregate:
        """`PRODUCT(self)` — see `ProductFold`."""
        return RuntimeAggregate(self.copy(), String(PRODUCT))

    def mean(self) raises -> RuntimeAggregate:
        """`AVG(self)` — see `MeanFold`."""
        return RuntimeAggregate(self.copy(), String(MEAN))

    def variance(self) raises -> RuntimeAggregate:
        """`VAR_POP(self)` — the population variance, Arrow's default.

        `var_samp()` is the sample form (`ddof=1`).
        """
        return RuntimeAggregate(self.copy(), String(VARIANCE))

    def var_samp(self) raises -> RuntimeAggregate:
        """`VAR_SAMP(self)` — the sample variance, `ddof=1`."""
        return RuntimeAggregate(self.copy(), String(VAR_SAMP))

    def stddev(self) raises -> RuntimeAggregate:
        """`STDDEV_POP(self)` — see `Dispersion`."""
        return RuntimeAggregate(self.copy(), String(STDDEV))

    def stddev_samp(self) raises -> RuntimeAggregate:
        """`STDDEV_SAMP(self)` — the square root of `var_samp()`."""
        return RuntimeAggregate(self.copy(), String(STDDEV_SAMP))

    def min(self) raises -> RuntimeAggregate:
        """`MIN(self)` — see `MinFold`, or `LexicalExtremum` for strings."""
        return RuntimeAggregate(self.copy(), String(MIN))

    def max(self) raises -> RuntimeAggregate:
        """`MAX(self)` — see `MaxFold`."""
        return RuntimeAggregate(self.copy(), String(MAX))

    def count(self) raises -> RuntimeAggregate:
        """`COUNT(self)` — see `ValidCount`."""
        return RuntimeAggregate(self.copy(), String(COUNT))

    def count_distinct(self) raises -> RuntimeAggregate:
        """`COUNT(DISTINCT self)` — see `DistinctCount`."""
        return RuntimeAggregate(self.copy(), String(COUNT_DISTINCT))

    def approx_count_distinct(self) raises -> RuntimeAggregate:
        """`APPROX_COUNT_DISTINCT(self)` — see `DistinctCount[exact=False]`."""
        return RuntimeAggregate(self.copy(), String(APPROX_COUNT_DISTINCT))

    def write_to[W: Writer](self, mut writer: W):
        var named = self.name()
        if named != "":
            writer.write(named)
            return
        if len(self._kids) == 0:
            writer.write(self._tag)
            return
        writer.write(self._tag, "(")
        for i in range(len(self._kids)):
            if i > 0:
                writer.write(", ")
            writer.write(self._kids[i][])
        writer.write(")")


# ---------------------------------------------------------------------------
# Leaf constructors
# ---------------------------------------------------------------------------
def column(var name: String) -> RuntimeValue:
    """Read `name` from the batch."""

    return RuntimeValue("column", Payload(name^))


def literal(var value: DynScalar) -> RuntimeValue:
    """A constant, broadcast to the batch's length on evaluation."""

    return RuntimeValue("literal", Payload(value^))


# ---------------------------------------------------------------------------
# Comparisons and boolean connectives
# ---------------------------------------------------------------------------
#
# The runtime lane had none of these, which meant it could not express a
# predicate at all -- `filter` was reachable only through a bare bool column.
# Every frontend that builds queries at run time (the Python one, a future SQL
# or wire protocol) needs them, and so does any statistics pruning, which keys
# on `eq` above all.


def eq(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """`l == r`. Numeric or string; the runtime dtype picks the kernel."""
    return RuntimeValue("eq", l, r)


def ne(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """`l != r`."""
    return RuntimeValue("ne", l, r)


def lt(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """`l < r`."""
    return RuntimeValue("lt", l, r)


def le(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """`l <= r`."""
    return RuntimeValue("le", l, r)


def gt(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """`l > r`."""
    return RuntimeValue("gt", l, r)


def ge(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """`l >= r`."""
    return RuntimeValue("ge", l, r)


def _bool_const(v: RuntimeValue) -> Optional[Bool]:
    """`True`/`False` if `v` is a non-null boolean literal, else `None`.

    A **null** boolean literal answers `None` deliberately. Under Kleene
    semantics `x AND NULL` is neither `x` nor `NULL` — it is `FALSE` when `x`
    is false and `NULL` otherwise — so folding a null as if it were false
    changes answers.
    """
    if v._tag != "literal" or not v._payload.isa[DynScalar]():
        return None
    ref boxed = v._payload[DynScalar]
    if boxed.type() != bool_ or not boxed.is_valid():
        return None
    return boxed.as_bool().value()


def and_(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """Three-valued AND, matching the fused lane's Kleene semantics.

    **Folded here, at construction, rather than by a plan rule.** A rule would
    have to look inside a `DynValue`, which the box does not allow and which
    would drag a comptime predicate into the runtime lane if it did. The
    operands are still concrete at this point, so the question is free to ask.

    `x AND TRUE` is `x`, and `x AND FALSE` is `FALSE` — both hold in Kleene
    logic, the second because `FALSE` annihilates even a null. Folding matters
    beyond tidiness: `Filter(FALSE)` is what lets `PropagateEmpty` collapse a
    subtree, and predicates fold to constants far more often than anyone writes
    `LIMIT 0`.
    """
    var lc = _bool_const(l)
    var rc = _bool_const(r)
    if lc:
        return r^ if lc.value() else l^
    if rc:
        return l^ if rc.value() else r^
    return RuntimeValue("and", l, r)


def or_(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """Three-valued OR. `x OR FALSE` is `x`; `x OR TRUE` is `TRUE`."""
    var lc = _bool_const(l)
    var rc = _bool_const(r)
    if lc:
        return l^ if lc.value() else r^
    if rc:
        return r^ if rc.value() else l^
    return RuntimeValue("or", l, r)


def xor(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """Three-valued XOR."""
    return RuntimeValue("xor", l, r)


def not_(var a: RuntimeValue) -> RuntimeValue:
    """Three-valued NOT: null stays null.

    `NOT NOT x` cancels, and a literal negates in place.
    """
    if a._tag == "not" and len(a._kids) == 1:
        return a._kids[0][].copy()
    var c = _bool_const(a)
    if c:
        return literal(BoolScalar(not c.value()).to_dyn())
    return RuntimeValue("not", a)


def coalesce(var values: List[RuntimeValue]) raises -> RuntimeValue:
    """First non-null across N expressions (PyArrow `pc.coalesce`).

    N-ary rather than a fold of binary nodes, because the kernel is already
    n-ary — the previous expression package folded only because its runtime
    node had no way to hold N children.
    """
    if len(values) == 0:
        raise Error("coalesce: needs at least one value")

    return RuntimeValue("coalesce", values^)


def case_when(
    var conditions: List[RuntimeValue],
    var values: List[RuntimeValue],
    var else_: Optional[RuntimeValue] = None,
) raises -> RuntimeValue:
    """Multi-branch `CASE WHEN` (PyArrow `pc.case_when`).

    The first `values[k]` whose `conditions[k]` is **valid and true**; a null
    condition counts as false. With no `else_`, an unmatched row is null.

    Children are laid out as `conditions ++ values ++ [else_]`, and the shape
    is recoverable from the **count's parity** alone: without an else there are
    `2n` children, with one there are `2n + 1`. Even means no else, odd means
    else. That is why this needs no payload — which matters because `EvalFn` is
    a `thin` pointer and cannot capture a flag.
    """
    if len(conditions) != len(values):
        raise Error(
            "case_when: ",
            len(conditions),
            " conditions but ",
            len(values),
            " values",
        )
    if len(conditions) == 0:
        raise Error("case_when: needs at least one condition")

    # Capacity reserved up front, and that is load-bearing rather than tidy:
    # a `RuntimeValue` carries a `Payload`, and when a `List` holding one grows
    # it moves its elements -- which resets `Variant`'s discriminant to 0, so
    # already-stored payloads come back as the variant's *first* member. Five
    # appends into an unreserved list read back as `int64,null,int64,null,
    # int64`. Same reason `StructArray.__getitem__` pre-allocates
    # (`arrays.mojo:1930`), and it is necessary but *not* sufficient here --
    # see `docs/backlog.md` on the separate temporary-lifetime miscompile that
    # corrupts this function's arguments before it ever runs.
    var kids = List[RuntimeValue](
        capacity=len(conditions) + len(values) + (1 if else_ else 0)
    )
    for ref c in conditions:
        kids.append(c.copy())
    for ref v in values:
        kids.append(v.copy())
    if else_:
        kids.append(else_.value().copy())
    return RuntimeValue("case_when", kids^)


def if_else(
    var cond: RuntimeValue, var then: RuntimeValue, var otherwise: RuntimeValue
) raises -> RuntimeValue:
    """`CASE WHEN cond THEN then ELSE otherwise END` — the runtime twin of
    `builders.if_else`, which builds the comptime lane's `CaseWhen[C, T, E]`.

    One branch of `case_when` rather than its own tag: the multi-branch node
    already carries this shape, and a second tag meaning the same thing is how
    two evaluators drift into disagreeing about whether a null condition
    selects the else arm.
    """
    return case_when([cond^], [then^], Optional(otherwise^))


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------
#
# The runtime half of `NumericValue`'s operator surface. Every verb below is
# one line because the work is in `evaluate`'s `_binary` / `_unary`; what
# these fix is the *spelling*, so one expression reads the same in either
# lane:
#
#     col("a", int64) * lit(2, int64)   # comptime, fuses
#     col("a") * lit(2)                 # runtime,  materialises
#
# `+` is the one that cannot be decided here: over erased operands it means
# addition or concatenation according to the runtime dtype, so `_binary`
# picks between `AddKernel` and `ConcatKernel` when the batch arrives.


def add(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """`l + r`, or string concatenation if the operands turn out to be
    strings."""
    return RuntimeValue("add", l, r)


def sub(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """`l - r`."""
    return RuntimeValue("sub", l, r)


def mul(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """`l * r`."""
    return RuntimeValue("mul", l, r)


def truediv(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """`l / r` — always `float64`, so `5 / 2` is 2.5. See `FloatBinary`."""
    return RuntimeValue("truediv", l, r)


def floordiv(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """`l // r` — see `FloordivKernel`."""
    return RuntimeValue("floordiv", l, r)


def mod(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """`l % r` — see `ModKernel`."""
    return RuntimeValue("mod", l, r)


def pow(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """`l ** r` — always `float64`, like `/`."""
    return RuntimeValue("pow", l, r)


def neg(var a: RuntimeValue) -> RuntimeValue:
    """`-a`."""
    return RuntimeValue("neg", a)


# ---------------------------------------------------------------------------
# Elementwise math
# ---------------------------------------------------------------------------


def abs(var a: RuntimeValue) -> RuntimeValue:
    """`|a|`."""
    return RuntimeValue("abs", a)


def sign(var a: RuntimeValue) -> RuntimeValue:
    """-1, 0 or 1 by the sign of `a`."""
    return RuntimeValue("sign", a)


def floor(var a: RuntimeValue) -> RuntimeValue:
    """Round toward minus infinity."""
    return RuntimeValue("floor", a)


def ceil(var a: RuntimeValue) -> RuntimeValue:
    """Round toward plus infinity."""
    return RuntimeValue("ceil", a)


def round(var a: RuntimeValue) -> RuntimeValue:
    """Round half away from zero."""
    return RuntimeValue("round", a)


def trunc(var a: RuntimeValue) -> RuntimeValue:
    """Round toward zero."""
    return RuntimeValue("trunc", a)


def sqrt(var a: RuntimeValue) -> RuntimeValue:
    """The square root, as `float64`."""
    return RuntimeValue("sqrt", a)


def exp(var a: RuntimeValue) -> RuntimeValue:
    """`e ** a`, as `float64`."""
    return RuntimeValue("exp", a)


def ln(var a: RuntimeValue) -> RuntimeValue:
    """The natural logarithm, as `float64`."""
    return RuntimeValue("ln", a)


# ---------------------------------------------------------------------------
# Null and value predicates
# ---------------------------------------------------------------------------
#
# `is_null` / `is_valid` read the *validity bitmap* and are never null
# themselves; `is_nan` / `is_inf` read the *values* and are null where the
# input is. That is the same split the comptime lane draws between
# `NullPredicate` and `ValuePredicate`, and it is why `is_nan(NULL)` is NULL
# while `is_null(NULL)` is TRUE.


def is_null(var a: RuntimeValue) -> RuntimeValue:
    """True where `a` is null. Never null itself."""
    return RuntimeValue("is_null", a)


def is_valid(var a: RuntimeValue) -> RuntimeValue:
    """True where `a` is *not* null — Arrow's spelling of `~is_null`."""
    return RuntimeValue("is_valid", a)


def is_nan(var a: RuntimeValue) -> RuntimeValue:
    """True where this floating column is NaN; null where it is null."""
    return RuntimeValue("is_nan", a)


def is_inf(var a: RuntimeValue) -> RuntimeValue:
    """True where this floating column is +/-infinity."""
    return RuntimeValue("is_inf", a)


def nullif(var a: RuntimeValue, var b: RuntimeValue) -> RuntimeValue:
    """SQL `NULLIF(a, b)` — see `NullifKernel`."""
    return RuntimeValue("nullif", a, b)


def fill_null(var a: RuntimeValue, var fill: RuntimeValue) -> RuntimeValue:
    """`fill` wherever `a` is null — see `FillNullKernel`."""
    return RuntimeValue("fill_null", a, fill)


# ---------------------------------------------------------------------------
# Strings
# ---------------------------------------------------------------------------


def upper(var a: RuntimeValue) -> RuntimeValue:
    """Upper-case each element."""
    return RuntimeValue("upper", a)


def lower(var a: RuntimeValue) -> RuntimeValue:
    """Lower-case each element."""
    return RuntimeValue("lower", a)


def strip(var a: RuntimeValue) -> RuntimeValue:
    """Drop leading and trailing whitespace."""
    return RuntimeValue("strip", a)


def lstrip(var a: RuntimeValue) -> RuntimeValue:
    """Drop leading whitespace."""
    return RuntimeValue("lstrip", a)


def rstrip(var a: RuntimeValue) -> RuntimeValue:
    """Drop trailing whitespace."""
    return RuntimeValue("rstrip", a)


def reverse(var a: RuntimeValue) -> RuntimeValue:
    """Reverse each element."""
    return RuntimeValue("reverse", a)


def capitalize(var a: RuntimeValue) -> RuntimeValue:
    """Upper-case the first character and lower-case the rest."""
    return RuntimeValue("capitalize", a)


def length(var a: RuntimeValue) -> RuntimeValue:
    """Byte length per element, as `int32` — pyarrow's `utf8_length`."""
    return RuntimeValue("length", a)


def startswith(var a: RuntimeValue, var prefix: RuntimeValue) -> RuntimeValue:
    """True where `a` starts with `prefix`."""
    return RuntimeValue("startswith", a, prefix)


def endswith(var a: RuntimeValue, var suffix: RuntimeValue) -> RuntimeValue:
    """True where `a` ends with `suffix`."""
    return RuntimeValue("endswith", a, suffix)


def contains(var a: RuntimeValue, var needle: RuntimeValue) -> RuntimeValue:
    """True where `needle` occurs anywhere in `a`."""
    return RuntimeValue("contains", a, needle)


def like(var a: RuntimeValue, var pattern: String) -> RuntimeValue:
    """SQL `LIKE` — `%` and `_` wildcards, case-sensitive.

    The pattern is a payload rather than a child so `LikeKernel` compiles it
    once per batch instead of once per row.
    """
    return RuntimeValue("like", a, Payload(pattern^))


def ilike(var a: RuntimeValue, var pattern: String) -> RuntimeValue:
    """SQL `ILIKE` — as `like`, case-insensitive."""
    return RuntimeValue("ilike", a, Payload(pattern^))


# --- the SQL function surface ----------------------------------------------
#
# Nine verbs this lane could not spell at all until the arguments stopped being
# constants: their arguments are children, so `RuntimeValue` already had
# everywhere to put them. `char_length` and `ascii` are below with the rest of
# the nullary surface — they take no argument and never needed this.
#
# Every one of them is null-propagating in **every** argument position, which
# is DuckDB's behaviour and the kernels' — see `StringOperands.is_valid`.


def substr(
    var a: RuntimeValue, var start: RuntimeValue, var count: RuntimeValue
) -> RuntimeValue:
    """SQL `substr(a, start, count)` — 1-based, counting characters."""
    return RuntimeValue("substr", [a^, start^, count^])


def left(var a: RuntimeValue, var count: RuntimeValue) -> RuntimeValue:
    """SQL `left(a, count)` — the first `count` characters, or all but the last
    `|count|` when it is negative."""
    return RuntimeValue("left", a, count)


def right(var a: RuntimeValue, var count: RuntimeValue) -> RuntimeValue:
    """SQL `right(a, count)` — `left` from the other end."""
    return RuntimeValue("right", a, count)


def repeat(var a: RuntimeValue, var count: RuntimeValue) -> RuntimeValue:
    """SQL `repeat(a, count)` — `a` concatenated `count` times. Zero and
    negative counts both give the empty string."""
    return RuntimeValue("repeat", a, count)


def lpad(
    var a: RuntimeValue, var width: RuntimeValue, var fill: RuntimeValue
) -> RuntimeValue:
    """SQL `lpad(a, width, fill)` — pad on the left, **truncating** when the
    input is already longer."""
    return RuntimeValue("lpad", [a^, width^, fill^])


def rpad(
    var a: RuntimeValue, var width: RuntimeValue, var fill: RuntimeValue
) -> RuntimeValue:
    """SQL `rpad(a, width, fill)` — `lpad` from the other end."""
    return RuntimeValue("rpad", [a^, width^, fill^])


def replace(
    var a: RuntimeValue,
    var pattern: RuntimeValue,
    var replacement: RuntimeValue,
) -> RuntimeValue:
    """SQL `replace(a, pattern, replacement)` — **every** occurrence,
    literally. An empty pattern returns the input unchanged."""
    return RuntimeValue("replace", [a^, pattern^, replacement^])


def split_part(
    var a: RuntimeValue, var sep: RuntimeValue, var index: RuntimeValue
) -> RuntimeValue:
    """SQL `split_part(a, sep, index)` — the `index`-th field, 1-based. An
    index past the last field answers the empty string, not null."""
    return RuntimeValue("split_part", [a^, sep^, index^])


def trim_chars(
    var a: RuntimeValue, var characters: RuntimeValue
) -> RuntimeValue:
    """SQL `trim(a, characters)` — strip any leading or trailing character that
    is a **member of the set**, not the literal substring."""
    return RuntimeValue("trim_chars", a, characters)


def position(var a: RuntimeValue, var needle: RuntimeValue) -> RuntimeValue:
    """SQL `position(needle IN a)` — the 1-based character index as `int64`,
    and **0** when absent. A *null* needle answers null, which is a different
    fact."""
    return RuntimeValue("position", a, needle)


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------


def year(var a: RuntimeValue) -> RuntimeValue:
    """The calendar year, as `int32`."""
    return RuntimeValue("year", a)


def month(var a: RuntimeValue) -> RuntimeValue:
    """The month, 1-12."""
    return RuntimeValue("month", a)


def day(var a: RuntimeValue) -> RuntimeValue:
    """The day of the month, 1-31."""
    return RuntimeValue("day", a)


def hour(var a: RuntimeValue) -> RuntimeValue:
    """The hour, 0-23."""
    return RuntimeValue("hour", a)


def minute(var a: RuntimeValue) -> RuntimeValue:
    """The minute, 0-59."""
    return RuntimeValue("minute", a)


def second(var a: RuntimeValue) -> RuntimeValue:
    """The second, 0-59."""
    return RuntimeValue("second", a)


def quarter(var a: RuntimeValue) -> RuntimeValue:
    """The quarter, 1-4."""
    return RuntimeValue("quarter", a)


def day_of_week(var a: RuntimeValue) -> RuntimeValue:
    """The day of the week, Monday = 0."""
    return RuntimeValue("day_of_week", a)


def day_of_year(var a: RuntimeValue) -> RuntimeValue:
    """The day of the year, 1-366."""
    return RuntimeValue("day_of_year", a)


def date_trunc(var a: RuntimeValue, var unit: String) raises -> RuntimeValue:
    """Floor to a `CalendarUnit` boundary, keeping the input's type.

    The unit is parsed **here**, at construction, so a bad spelling fails when
    the plan is built rather than on the row that first evaluates it -- which
    is the reason `CalendarUnit` is a type and not a `String` in the first
    place. The parsed value is not what gets stored: `Payload` has no
    `CalendarUnit` member, and adding one to buy a second parse is not worth a
    variant member. `evaluate` re-parses a string this call has already proved
    valid.
    """
    _ = CalendarUnit.parse(unit)
    return RuntimeValue("date_trunc", a, Payload(unit^))


# ---------------------------------------------------------------------------
# Membership, casting, nested
# ---------------------------------------------------------------------------


def isin(var a: RuntimeValue, var value_set: DynArray) -> RuntimeValue:
    """True where `a`'s value appears in `value_set` — SQL `IN (...)`.

    The set is a `DynArray` payload rather than a child: it is the same set on
    every row and every batch, so hashing it into the probe table is work that
    belongs to the plan, not to the morsel. The comptime lane has no
    counterpart -- `IsInKernel` decides membership on the 64-bit hash alone
    and has no typed leaf to fuse into.
    """
    return RuntimeValue("isin", a, Payload(value_set^))


def cast(
    var a: RuntimeValue, var to: DynType, safe: Bool = True
) -> RuntimeValue:
    """Cast to `to`. With `safe` a lossy conversion raises; without it the
    truncating/wrapping conversion is used and an unparseable string is nulled.

    `safe` rides the **tag** rather than the payload, because `Payload` holds
    one value and the target dtype is already it. Two tags for one kernel is
    the cheaper of the two, and `_unary` reads the flag back off the tag it
    matched.
    """
    return RuntimeValue("cast" if safe else "cast_unsafe", a, Payload(to^))


def array_length(var a: RuntimeValue) -> RuntimeValue:
    """The number of elements in each list, as `int32`.

    A list column consumed into a numeric one, which is the only way a list is
    read: a list element is a whole sub-array rather than a value a lane can
    hold. Same reason `builders.array_length` is the comptime lane's only list
    verb.
    """
    return RuntimeValue("array_length", a)


def char_length(var a: RuntimeValue) -> RuntimeValue:
    """SQL `length` — the **character** count, as `int64`, where `length`
    above is the byte count `octet_length` asks for."""
    return RuntimeValue("char_length", a)


def ascii(var a: RuntimeValue) -> RuntimeValue:
    """SQL `ascii` — the first character's code point, 0 for the empty
    string."""
    return RuntimeValue("ascii", a)


def week(var a: RuntimeValue) -> RuntimeValue:
    """The ISO-8601 week number, 1-53, as `int64`."""
    return RuntimeValue("week", a)


def iso_year(var a: RuntimeValue) -> RuntimeValue:
    """The ISO-8601 week-numbering year, as `int64` — not always `year`."""
    return RuntimeValue("iso_year", a)


def epoch(var a: RuntimeValue) -> RuntimeValue:
    """Whole seconds since 1970-01-01, floored, as `int64`."""
    return RuntimeValue("epoch", a)


def last_day(var a: RuntimeValue) -> RuntimeValue:
    """The last day of this value's month, as `date32`."""
    return RuntimeValue("last_day", a)


def day_name(var a: RuntimeValue) -> RuntimeValue:
    """The full English weekday name."""
    return RuntimeValue("day_name", a)


def month_name(var a: RuntimeValue) -> RuntimeValue:
    """The full English month name."""
    return RuntimeValue("month_name", a)
