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

from ...arrays import StructArray, BoolArray, DynArray
from ...kernels.aggregate import (
    APPROX_COUNT_DISTINCT,
    COUNT,
    COUNT_DISTINCT,
    MAX,
    MEAN,
    MIN,
    PRODUCT,
    STDDEV,
    SUM,
    VARIANCE,
)
from ...kernels.boolean import AndKernel, NotKernel, OrKernel, XorKernel
from ...kernels.cast import cast as cast_array
from ...kernels.conditional import case_when as case_when_kernel
from ...kernels.conditional import coalesce as coalesce_kernel
from ...kernels.numeric import (
    NumericCompareKernel,
    EqKernel,
    GeKernel,
    GtKernel,
    LeKernel,
    LtKernel,
    NeKernel,
)
from ...kernels.string import (
    StringPredicateKernel,
    StringEqKernel,
    StringGeKernel,
    StringGtKernel,
    StringLeKernel,
    StringLtKernel,
    StringNeKernel,
)
from ...dtypes import DynType
from ...scalars import DynScalar
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

        `expr/` probed *every* expression this way, comptime ones included.
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
        if len(kids) == 1 and self._tag == "not":
            return Datum(NotKernel.dispatch(kids[0].copy()))

        if self._tag == "coalesce":
            return Datum(coalesce_kernel(kids))
        if self._tag == "case_when":
            var has_else = len(kids) % 2 == 1
            var n = (len(kids) - 1) // 2 if has_else else len(kids) // 2
            var conds = List[BoolArray](capacity=n)
            for i in range(n):
                conds.append(kids[i].as_bool().copy())
            var vals = List[DynArray](capacity=n)
            for i in range(n):
                vals.append(kids[n + i].copy())
            var otherwise = Optional[DynArray](None)
            if has_else:
                otherwise = kids[len(kids) - 1].copy()
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
        if l.dtype().is_string_like():
            return S.dispatch(l^, r^)
        var lt = l.dtype()
        var rt = r.dtype()
        if lt != rt:
            if lt.is_string_like() or rt.is_string_like():
                raise Error("compare: cannot compare ", lt, " with ", rt)
            # Cast the narrower side up. `cast` decides what "wider" means; a
            # pair it rejects raises there rather than comparing raw bits.
            try:
                r = cast_array(r^, lt)
            except:
                l = cast_array(l^, rt)
        return K.dispatch(l^, r^)

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

    def sum(self) raises -> RuntimeAggregate:
        """`SUM(self)`. Integers widen to int64; floats stay float64."""
        return RuntimeAggregate(self.copy(), String(SUM))

    def product(self) raises -> RuntimeAggregate:
        """`PRODUCT(self)`."""
        return RuntimeAggregate(self.copy(), String(PRODUCT))

    def mean(self) raises -> RuntimeAggregate:
        """`AVG(self)`. Accumulates in float64 over the valid values, so nulls
        are excluded rather than counted as zero."""
        return RuntimeAggregate(self.copy(), String(MEAN))

    def variance(self) raises -> RuntimeAggregate:
        """`VAR_POP(self)` — the population variance, Arrow's default.

        The sample form is `Dispersion[1, False]` in the comptime lane. It is
        absent here only because no name is bound to it; adding `var_samp`
        costs one string and one arm of `resolve`.
        """
        return RuntimeAggregate(self.copy(), String(VARIANCE))

    def stddev(self) raises -> RuntimeAggregate:
        """`STDDEV_POP(self)` — the square root of `variance()`."""
        return RuntimeAggregate(self.copy(), String(STDDEV))

    def min(self) raises -> RuntimeAggregate:
        """`MIN(self)`. Keeps the input's dtype — a timestamp's unit and
        timezone included; lexicographic over a string column."""
        return RuntimeAggregate(self.copy(), String(MIN))

    def max(self) raises -> RuntimeAggregate:
        """`MAX(self)`."""
        return RuntimeAggregate(self.copy(), String(MAX))

    def count(self) raises -> RuntimeAggregate:
        """`COUNT(self)` — the *non-null* values of `self`, not the row
        count."""
        return RuntimeAggregate(self.copy(), String(COUNT))

    def count_distinct(self) raises -> RuntimeAggregate:
        """`COUNT(DISTINCT self)` — exact, nulls excluded (SQL semantics)."""
        return RuntimeAggregate(self.copy(), String(COUNT_DISTINCT))

    def approx_count_distinct(self) raises -> RuntimeAggregate:
        """`APPROX_COUNT_DISTINCT(self)` — a HyperLogLog estimate, ~0.65%
        standard error, nulls excluded."""
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


def and_(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """Three-valued AND, matching the fused lane's Kleene semantics."""
    return RuntimeValue("and", l, r)


def or_(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """Three-valued OR."""
    return RuntimeValue("or", l, r)


def xor(var l: RuntimeValue, var r: RuntimeValue) -> RuntimeValue:
    """Three-valued XOR."""
    return RuntimeValue("xor", l, r)


def not_(var a: RuntimeValue) -> RuntimeValue:
    """Three-valued NOT: null stays null."""
    return RuntimeValue("not", a)


def coalesce(var values: List[RuntimeValue]) raises -> RuntimeValue:
    """First non-null across N expressions (PyArrow `pc.coalesce`).

    N-ary rather than a fold of binary nodes, because the kernel is already
    n-ary — `expr/` folds only because its runtime node had no way to hold N
    children.
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
