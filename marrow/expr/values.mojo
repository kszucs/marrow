"""Scalar expression nodes for the marrow expression system.

``Expr``        — unified n-ary term expression node
``Value``       — trait every scalar expression node must implement

Factory functions
-----------------
``col(index)``  / ``col(name)`` — column reference
``lit[T](value)``               — typed scalar literal
``if_else(cond, then_, else_)`` — conditional

Operator overloads on ``Expr``: ``+``, ``-``, ``*``, ``/``, ``>``,
``<``, ``>=``, ``<=``, ``==``, ``!=``, ``&``, ``|``, ``~`` (NOT),
unary ``-``.  Instance methods: ``.abs()``, ``.is_null()``, ``.cast(to)``.

Expression tags
---------------
LOAD    - Column reference
LITERAL - Constant value
ADD/SUB/MUL/DIV/EQ/NE/LT/LE/GT/GE/AND/OR - Binary operations
NEG/ABS/NOT - Unary operations
IS_NULL - Null check
IF_ELSE - Conditional
CAST - Type cast
"""

from std.memory import ArcPointer
from marrow.arrays import AnyArray
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import AnyDataType, NumericType
from marrow.schema import Schema


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
comptime FUSED: UInt8 = 20
"""Tag for Expr nodes that carry a comptime-fused expression in _fused."""


# ---------------------------------------------------------------------------
# Expr - unified n-ary term expression node
# ---------------------------------------------------------------------------


struct Expr(Copyable, ImplicitlyCopyable, ImplicitlyDestructible, Movable, Writable):
    """Unified expression node using tag-based dispatch."""

    var _tag: UInt8
    var _args: List[Expr]
    var _kind_data: UInt8
    var _value: Optional[AnyArray]
    var _name: String
    var dispatch: UInt8
    var _fused: Optional[ArcPointer[NoneType]]
    var _virt_fused_dtype: def(ArcPointer[NoneType]) thin -> Optional[AnyDataType]
    var _virt_fused_write: def(ArcPointer[NoneType]) thin -> String

    def __init__(
        out self,
        tag: UInt8,
        var args: List[Expr],
        kind_data: UInt8,
        var value: Optional[AnyArray],
        var name: String,
    ):
        self._tag = tag
        self._args = args^
        self._kind_data = kind_data
        self._value = value.copy()
        self._name = name^
        self.dispatch = 0
        self._fused = None
        self._virt_fused_dtype = Self._tramp_fused_dtype_default
        self._virt_fused_write = Self._tramp_fused_write_default

    def __init__(out self, *, copy: Self):
        self._tag = copy._tag
        self._args = List[Expr]()
        for i in range(len(copy._args)):
            self._args.append(Expr(copy=copy._args[i]))
        self._kind_data = copy._kind_data
        self._value = copy._value.copy()
        self._name = copy._name.copy()
        self.dispatch = copy.dispatch
        self._fused = copy._fused
        self._virt_fused_dtype = copy._virt_fused_dtype
        self._virt_fused_write = copy._virt_fused_write

    @staticmethod
    def _tramp_fused_dtype_default(ptr: ArcPointer[NoneType]) -> Optional[AnyDataType]:
        return None

    @staticmethod
    def _tramp_fused_write_default(ptr: ArcPointer[NoneType]) -> String:
        return String()

    def kind(self) -> UInt8:
        return self._tag

    def dtype(self) -> Optional[AnyDataType]:
        if self._fused:
            try:
                return self._virt_fused_dtype(self._fused[])
            except:
                return None
        if self._tag == LITERAL:
            return self._value.value().dtype()
        return None

    def inputs(self) -> List[Expr]:
        var result = List[Expr](capacity=len(self._args))
        for ref a in self._args:
            result.append(a.copy())
        return result^

    def write_to[W: Writer](self, mut writer: W):
        if self._fused:
            writer.write("fused(...)")
        elif self._tag == LOAD:
            writer.write(t"input({self._kind_data})")
        elif self._tag == LITERAL:
            writer.write("literal(...)")
        elif self._tag >= ADD and self._tag <= OR:
            var op_name = String("?")
            if self._tag == ADD: op_name = "add"
            elif self._tag == SUB: op_name = "sub"
            elif self._tag == MUL: op_name = "mul"
            elif self._tag == DIV: op_name = "div"
            elif self._tag == EQ: op_name = "equal"
            elif self._tag == NE: op_name = "not_equal"
            elif self._tag == LT: op_name = "less"
            elif self._tag == LE: op_name = "less_equal"
            elif self._tag == GT: op_name = "greater"
            elif self._tag == GE: op_name = "greater_equal"
            elif self._tag == AND: op_name = "and"
            elif self._tag == OR: op_name = "or"
            writer.write(t"{op_name}(")
            if len(self._args) >= 1: self._args[0].write_to(writer)
            writer.write(t", ")
            if len(self._args) >= 2: self._args[1].write_to(writer)
            writer.write(t")")
        elif self._tag >= NEG and self._tag <= NOT:
            var op_name = String("?")
            if self._tag == NEG: op_name = "neg"
            elif self._tag == ABS: op_name = "abs"
            elif self._tag == NOT: op_name = "not"
            writer.write(t"{op_name}(")
            if len(self._args) >= 1: self._args[0].write_to(writer)
            writer.write(t")")
        elif self._tag == IS_NULL:
            writer.write(t"is_null(")
            if len(self._args) >= 1: self._args[0].write_to(writer)
            writer.write(t")")
        elif self._tag == IF_ELSE:
            writer.write(t"if_else(")
            if len(self._args) >= 1: self._args[0].write_to(writer)
            writer.write(t", ")
            if len(self._args) >= 2: self._args[1].write_to(writer)
            writer.write(t", ")
            if len(self._args) >= 3: self._args[2].write_to(writer)
            writer.write(t")")
        elif self._tag == CAST:
            writer.write(t"cast(")
            if len(self._args) >= 1: self._args[0].write_to(writer)
            writer.write(t")")
        else:
            writer.write(t"<unknown>({self._tag})")

    # Operator overloads (methods on Expr)
    def __add__(self, rhs: Expr) -> Expr:
        return Expr(tag=ADD, args=[self.copy(), rhs.copy()], kind_data=0, value=None, name=String())

    def __sub__(self, rhs: Expr) -> Expr:
        return Expr(tag=SUB, args=[self.copy(), rhs.copy()], kind_data=0, value=None, name=String())

    def __mul__(self, rhs: Expr) -> Expr:
        return Expr(tag=MUL, args=[self.copy(), rhs.copy()], kind_data=0, value=None, name=String())

    def __truediv__(self, rhs: Expr) -> Expr:
        return Expr(tag=DIV, args=[self.copy(), rhs.copy()], kind_data=0, value=None, name=String())

    def __gt__(self, rhs: Expr) -> Expr:
        return Expr(tag=GT, args=[self.copy(), rhs.copy()], kind_data=0, value=None, name=String())

    def __lt__(self, rhs: Expr) -> Expr:
        return Expr(tag=LT, args=[self.copy(), rhs.copy()], kind_data=0, value=None, name=String())

    def __ge__(self, rhs: Expr) -> Expr:
        return Expr(tag=GE, args=[self.copy(), rhs.copy()], kind_data=0, value=None, name=String())

    def __le__(self, rhs: Expr) -> Expr:
        return Expr(tag=LE, args=[self.copy(), rhs.copy()], kind_data=0, value=None, name=String())

    def __eq__(self, rhs: Expr) -> Expr:
        return Expr(tag=EQ, args=[self.copy(), rhs.copy()], kind_data=0, value=None, name=String())

    def __ne__(self, rhs: Expr) -> Expr:
        return Expr(tag=NE, args=[self.copy(), rhs.copy()], kind_data=0, value=None, name=String())

    def __neg__(self) -> Expr:
        return Expr(tag=NEG, args=[self.copy()], kind_data=0, value=None, name=String())

    def __invert__(self) -> Expr:
        return Expr(tag=NOT, args=[self.copy()], kind_data=0, value=None, name=String())

    def __and__(self, rhs: Expr) -> Expr:
        return Expr(tag=AND, args=[self.copy(), rhs.copy()], kind_data=0, value=None, name=String())

    def __or__(self, rhs: Expr) -> Expr:
        return Expr(tag=OR, args=[self.copy(), rhs.copy()], kind_data=0, value=None, name=String())

    def is_null(self) -> Expr:
        return Expr(tag=IS_NULL, args=[self.copy()], kind_data=0, value=None, name=String())

    def abs(self) -> Expr:
        return Expr(tag=ABS, args=[self.copy()], kind_data=0, value=None, name=String())

    def cast(self, to: AnyDataType) -> Expr:
        return Expr(tag=CAST, args=[self.copy()], kind_data=0, value=None, name=String())


# ---------------------------------------------------------------------------
# Value trait — interface every concrete expression node must implement
# ---------------------------------------------------------------------------


trait Value(ImplicitlyDestructible, Movable):
    """Interface for immutable scalar expression nodes."""

    def kind(self) -> UInt8:
        """Return the node-kind constant."""
        ...

    def dtype(self) -> Optional[AnyDataType]:
        """Return the output data type, or None if not yet inferred."""
        ...

    def inputs(self) -> List[Expr]:
        """Return child expressions (empty for leaf nodes)."""
        ...

    def write_to[W: Writer](self, mut writer: W):
        """Format this node for display."""
        ...


# ---------------------------------------------------------------------------
# Factory functions (return Expr)
# ---------------------------------------------------------------------------


def col(index: Int) -> Expr:
    """Reference to the ``index``-th input column."""
    return Expr(tag=LOAD, args=List[Expr](), kind_data=index, value=None, name=String())

def col(var name: String) -> Expr:
    """Reference to a named column."""
    return Expr(tag=LOAD, args=List[Expr](), kind_data=-1, value=None, name=name^)

def lit[T: NumericType](value: Scalar[T.native]) raises -> Expr:
    """A scalar constant."""
    var builder = PrimitiveBuilder[T](T(), 1)
    builder.unsafe_append(value)
    return Expr(tag=LITERAL, args=List[Expr](), kind_data=0, value=builder.finish().to_any(), name=String())

def if_else(cond: Expr, then_: Expr, else_: Expr) -> Expr:
    """Element-wise conditional."""
    return Expr(tag=IF_ELSE, args=[cond.copy(), then_.copy(), else_.copy()], kind_data=0, value=None, name=String())
