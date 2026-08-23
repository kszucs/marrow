"""The runtime lane: expressions whose structure lives in fields.

One struct — a tag, its children, a payload, and a pointer to the function that
evaluates it. Where the comptime lane puts a whole subtree in a type and fuses
it into one SIMD loop, this materialises a `DynArray` **per node, per morsel**.

That is the trade, and it is why the two lanes exist rather than one:

| | comptime | runtime |
|---|---|---|
| structure | in the type | in the fields |
| per node | inlined into one loop | one materialised column |
| built from | Mojo source | anything, including Python at run time |
| binary | 1.46 MB | 4.91 MB (same plan) |

The runtime lane is what a frontend uses when the query is not known until the
program runs, which is every frontend that is not the Mojo DSL. It is also why
this package has a box at all: a plan holds either lane through `DynValue`,
and that mixing is what buys the 1.46 MB.

**A tag never selects a kernel.** `_tag` is how a node prints and how it
prunes; `_eval` is how it computes, and it is a function pointer bound at
construction. Routing on the tag would put every kernel in every binary that
builds any expression — the same closed-erasure property that keeps unused
operators out of a plan.
"""

from std.memory import ArcPointer
from std.utils import Variant

from ...arrays import DynArray
from ...dtypes import DynType
from ...scalars import DynScalar
from ...schema import Schema
from ...tabular import RecordBatch
from ..core import Analyzable, Datum, Executable, Shape
from ..physical import Evaluable, DynOperator, EvalOperator


comptime Payload = Variant[NoneType, String, DynType, DynArray, DynScalar]
"""What a node carries besides its children — a column name, a cast target, a
literal, an `IsIn` value set.

A closed variant rather than an erased box: the set is small, known, and adding
to it should be a decision someone makes rather than something a caller can do
from outside.
"""

comptime EvalFn = def(
    List[DynArray], Payload, RecordBatch
) thin raises -> DynArray
"""How one node computes its column, given its children's already-computed
columns, its payload, and the batch.

A `thin` pointer, bound at construction, so a binary links exactly the kernels
its expressions name and nothing else.
"""


struct RuntimeValue(
    Analyzable, Copyable, Deinitable, Evaluable, Executable, Movable, Writable
):
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
    var _eval: EvalFn

    # -- construction -------------------------------------------------------

    def __init__(out self, var tag: String, ev: EvalFn, var payload: Payload):
        self._tag = tag^
        self._kids = List[ArcPointer[Self]]()
        self._payload = payload^
        self._eval = ev

    def __init__(out self, var tag: String, ev: EvalFn, a: Self):
        self._tag = tag^
        self._kids = [ArcPointer(a.copy())]
        self._payload = Payload(NoneType())
        self._eval = ev

    def __init__(out self, var tag: String, ev: EvalFn, a: Self, b: Self):
        self._tag = tag^
        self._kids = [ArcPointer(a.copy()), ArcPointer(b.copy())]
        self._payload = Payload(NoneType())
        self._eval = ev

    # -- Analyzable ---------------------------------------------------------

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
            for ref n in kid[].columns():
                var seen = False
                for ref have in out:
                    if have == n:
                        seen = True
                        break
                if not seen:
                    out.append(n.copy())
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
        return self.evaluate(RecordBatch.empty(schema)).to_array(0).dtype()

    # -- Executable ----------------------------------------------------------

    def to_operator(self, grouped: Bool) raises -> DynOperator[Datum]:
        """The runtime lane's half of the same contract. Its processor is the
        same adapter the comptime lane uses — the lanes differ in how they
        compute, not in how they are turned into something that runs."""
        return EvalOperator[Self](self.copy())

    def evaluate(self, batch: RecordBatch) raises -> Datum:
        # Leaf spelled out — see `columns`. There is no switch here: which
        # kernel runs was decided when the node was built, by which `EvalFn`
        # the constructing method named.
        if len(self._kids) == 0:
            return self._eval(List[DynArray](), self._payload, batch)

        var kids = List[DynArray]()
        for ref kid in self._kids:
            kids.append(kid[].evaluate(batch).to_array(batch.num_rows()))
        return self._eval(kids, self._payload, batch)

    # -- Writable -----------------------------------------------------------

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

    def eval(
        kids: List[DynArray], p: Payload, b: RecordBatch
    ) raises -> DynArray:
        return b.column(p[String]).copy()

    return RuntimeValue("column", eval, Payload(name^))


def literal(var value: DynScalar) -> RuntimeValue:
    """A constant, broadcast to the batch's length on evaluation."""

    def eval(
        kids: List[DynArray], p: Payload, b: RecordBatch
    ) raises -> DynArray:
        return p[DynScalar].repeat(b.num_rows())

    return RuntimeValue("literal", eval, Payload(value^))
