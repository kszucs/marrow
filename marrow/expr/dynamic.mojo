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
CAST - Type cast (not yet implemented — see DynValue.eval)
LENGTH - String byte length (dispatches to kernels.string.string_lengths)
"""

from marrow.arrays import AnyArray
from marrow.dtypes import AnyDataType, NumericType
from marrow.scalars import AnyScalar, PrimitiveScalar
from marrow.schema import Schema
from marrow.tabular import RecordBatch
from marrow.expr.values import Value
from marrow.kernels.arithmetic import add, subtract, multiply, divide, neg, abs_
from marrow.kernels.boolean import and_, or_, not_, is_null, select
from marrow.kernels.compare import (
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
)
from marrow.kernels.string import string_lengths


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
comptime CAST: UInt8 = 19
comptime LENGTH: UInt8 = 20


# ---------------------------------------------------------------------------
# DynValue - unified n-ary term expression node
# ---------------------------------------------------------------------------


struct DynValue(
    Copyable,
    ImplicitlyCopyable,
    ImplicitlyDeletable,
    Movable,
    Value,
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

    def __init__(
        out self,
        tag: UInt8,
        var args: List[DynValue],
        kind_data: UInt8,
        var value: Optional[AnyScalar],
        var name: String,
    ):
        self._tag = tag
        self._args = args^
        self._kind_data = kind_data
        self._value = value.copy()
        self._name = name^

    def __init__(out self, *, copy: Self):
        self._tag = copy._tag
        self._args = List[DynValue]()
        for i in range(len(copy._args)):
            self._args.append(copy._args[i].copy())
        self._kind_data = copy._kind_data
        self._value = copy._value.copy()
        self._name = copy._name.copy()

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

    def to_array(self, batch: RecordBatch) raises -> AnyArray:
        """Evaluate against *batch*, resolving named ``col(...)`` references
        against its schema first. This is the ``AnyValue``-box entry point (the
        fused/streaming relations call it directly), so the executor's separate
        ``resolve_names`` pass is folded in here."""
        return self.resolve_names(batch.schema).eval(batch)

    def field_name(self) -> String:
        """Output column name (the LOAD name; empty for computed nodes)."""
        return self._name.copy()

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
        return None

    def inputs(self) -> List[DynValue]:
        var result = List[DynValue](capacity=len(self._args))
        for ref a in self._args:
            result.append(a.copy())
        return result^

    def eval(self, batch: RecordBatch) raises -> AnyArray:
        """Evaluate this expression tree against *batch*, dispatching on tag."""
        if self._tag == LOAD:
            return batch.columns[Int(self._kind_data)].copy()
        elif self._tag == LITERAL:
            return self._value.value().repeat(batch.num_rows())
        elif self._tag == ADD:
            return add(self._args[0].eval(batch), self._args[1].eval(batch))
        elif self._tag == SUB:
            return subtract(
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == MUL:
            return multiply(
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == DIV:
            return divide(self._args[0].eval(batch), self._args[1].eval(batch))
        elif self._tag == EQ:
            return equal(self._args[0].eval(batch), self._args[1].eval(batch))
        elif self._tag == NE:
            return not_equal(
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == LT:
            return less(self._args[0].eval(batch), self._args[1].eval(batch))
        elif self._tag == LE:
            return less_equal(
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == GT:
            return greater(self._args[0].eval(batch), self._args[1].eval(batch))
        elif self._tag == GE:
            return greater_equal(
                self._args[0].eval(batch), self._args[1].eval(batch)
            )
        elif self._tag == AND:
            return and_(self._args[0].eval(batch), self._args[1].eval(batch))
        elif self._tag == OR:
            return or_(self._args[0].eval(batch), self._args[1].eval(batch))
        elif self._tag == NEG:
            return neg(self._args[0].eval(batch))
        elif self._tag == ABS:
            return abs_(self._args[0].eval(batch))
        elif self._tag == NOT:
            return not_(self._args[0].eval(batch))
        elif self._tag == IS_NULL:
            return is_null(self._args[0].eval(batch))
        elif self._tag == LENGTH:
            return string_lengths(self._args[0].eval(batch)).to_any()
        elif self._tag == IF_ELSE:
            return select(
                self._args[0].eval(batch),
                self._args[1].eval(batch),
                self._args[2].eval(batch),
            )
        elif self._tag == CAST:
            raise Error("DynValue.eval: cast not implemented")
        else:
            raise Error("DynValue.eval: unknown expression kind ", self._tag)

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
        elif self._tag == IF_ELSE:
            return "if_else"
        elif self._tag == CAST:
            return "cast"
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

    def cast(self, to: AnyDataType) -> DynValue:
        return self._unary(CAST)


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
