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
"""

from ..arrays import AnyArray
from ..dtypes import AnyDataType, NumericType
from ..scalars import AnyScalar, PrimitiveScalar
from ..schema import Schema
from ..tabular import RecordBatch
from .pruning import PruneStats, PruneBound
from ..kernels.arithmetic import (
    AddKernel,
    SubKernel,
    MulKernel,
    DivKernel,
    NegKernel,
    AbsKernel,
)
from ..kernels.boolean import AndKernel, OrKernel, NotKernel, is_null, select
from ..kernels.compare import (
    equal,
    NeKernel,
    LtKernel,
    LeKernel,
    GtKernel,
    GeKernel,
)
from ..kernels.string import LengthKernel
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

    def __init__(
        out self,
        tag: UInt8,
        var args: List[DynValue],
        kind_data: UInt8,
        var value: Optional[AnyScalar],
        var name: String,
        var cast_to: Optional[AnyDataType] = None,
    ):
        self._tag = tag
        self._args = args^
        self._kind_data = kind_data
        self._value = value.copy()
        self._name = name^
        self._cast_to = cast_to^

    def __init__(out self, *, copy: Self):
        self._tag = copy._tag
        self._args = List[DynValue]()
        for i in range(len(copy._args)):
            self._args.append(copy._args[i].copy())
        self._kind_data = copy._kind_data
        self._value = copy._value.copy()
        self._name = copy._name.copy()
        self._cast_to = copy._cast_to.copy()

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
        """Return the column name for a named LOAD node (empty otherwise)."""
        return self._name

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
        elif self._tag == EQ:
            return equal(self._args[0].eval(batch), self._args[1].eval(batch))
        elif self._tag == NE:
            return NeKernel.dispatch(
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == LT:
            return LtKernel.dispatch(
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == LE:
            return LeKernel.dispatch(
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == GT:
            return GtKernel.dispatch(
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == GE:
            return GeKernel.dispatch(
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
        elif self._tag == NEG:
            return NegKernel.dispatch(self._args[0].eval(batch))
        elif self._tag == ABS:
            return AbsKernel.dispatch(self._args[0].eval(batch))
        elif self._tag == NOT:
            return NotKernel.dispatch(self._args[0].eval(batch))
        elif self._tag == IS_NULL:
            return is_null(self._args[0].eval(batch))
        elif self._tag == LENGTH:
            return LengthKernel.dispatch(self._args[0].eval(batch))
        elif self._tag == CAST:
            return cast_array(self._args[0].eval(batch), self._cast_to.value())
        elif self._tag == IF_ELSE:
            return select(
                self._args[0].eval(batch),
                self._args[1].eval(batch),
                self._args[2].eval(batch),
            )
        else:
            raise Error("DynValue.eval: unknown expression kind ", self._tag)

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
        elif self._tag == NEG:
            return "neg"
        elif self._tag == ABS:
            return "abs"
        elif self._tag == NOT:
            return "not"
        elif self._tag == IS_NULL:
            return "is_null"
        elif self._tag == LENGTH:
            return "length"
        elif self._tag == CAST:
            return "cast"
        elif self._tag == IF_ELSE:
            return "if_else"
        else:
            return String()

    def write_to[W: Writer](self, mut writer: W):
        if self._tag == LOAD:
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

    def abs(self) -> DynValue:
        return self._unary(ABS)

    def length(self) -> DynValue:
        return self._unary(LENGTH)


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
