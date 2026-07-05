"""Type-erased runtime expression nodes for the marrow expression system.

``Expr`` is the runtime counterpart to the comptime-typed layer in
``values.mojo``.  It exists so that query plans can be built and executed
without knowing concrete comptime types — this is what the Python bindings
(and any other runtime-typed caller) drive.  A single ``Expr`` node carries
a tag plus its child args, and dispatches its own execution by tag in
``eval()`` — there is no separate "processor" hierarchy mirroring the tree.

A comptime-typed node from ``values.mojo`` can be boxed into an ``Expr`` via
``NumericValue.to_expr()`` (tag ``FUSED``); ``eval()``/``dtype()``/
``write_to()`` on a boxed node all delegate back to the concrete comptime
node through trampolines, so a fused subtree keeps its single fused pass
even when driven through this type-erased path.

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
CAST - Type cast (not yet implemented — see Expr.eval)
FUSED - Carries a boxed comptime-typed node (see values.mojo)
"""

from std.memory import ArcPointer
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
"""Tag for Expr nodes that carry a boxed comptime-typed node in _fused."""


# ---------------------------------------------------------------------------
# Expr - unified n-ary term expression node
# ---------------------------------------------------------------------------


struct Expr(
    Copyable,
    ImplicitlyCopyable,
    ImplicitlyDestructible,
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
    var _args: List[Expr]
    var _kind_data: UInt8
    var _value: Optional[AnyScalar]
    var _name: String
    var _fused: Optional[ArcPointer[NoneType]]
    var _virt_fused_dtype: def(ArcPointer[NoneType]) thin -> Optional[
        AnyDataType
    ]
    var _virt_fused_write: def(ArcPointer[NoneType]) thin -> String
    var _virt_fused_eval: def(
        ArcPointer[NoneType], RecordBatch
    ) thin raises -> AnyArray

    def __init__(
        out self,
        tag: UInt8,
        var args: List[Expr],
        kind_data: UInt8,
        var value: Optional[AnyScalar],
        var name: String,
    ):
        self._tag = tag
        self._args = args^
        self._kind_data = kind_data
        self._value = value.copy()
        self._name = name^
        self._fused = None
        self._virt_fused_dtype = Self._tramp_fused_dtype_default
        self._virt_fused_write = Self._tramp_fused_write_default
        self._virt_fused_eval = Self._tramp_fused_eval_default

    def __init__(out self, *, copy: Self):
        self._tag = copy._tag
        self._args = List[Expr]()
        for i in range(len(copy._args)):
            self._args.append(copy._args[i].copy())
        self._kind_data = copy._kind_data
        self._value = copy._value.copy()
        self._name = copy._name.copy()
        self._fused = copy._fused
        self._virt_fused_dtype = copy._virt_fused_dtype
        self._virt_fused_write = copy._virt_fused_write
        self._virt_fused_eval = copy._virt_fused_eval

    @staticmethod
    def _tramp_fused_dtype_default(
        ptr: ArcPointer[NoneType],
    ) -> Optional[AnyDataType]:
        return None

    @staticmethod
    def _tramp_fused_write_default(ptr: ArcPointer[NoneType]) -> String:
        return String()

    @staticmethod
    def _tramp_fused_eval_default(
        ptr: ArcPointer[NoneType], batch: RecordBatch
    ) raises -> AnyArray:
        raise Error("Expr.eval: FUSED tag set without a bound comptime node")

    def kind(self) -> UInt8:
        return self._tag

    def kind_data(self) -> UInt8:
        """Return the kind-specific data (e.g. column index for LOAD nodes)."""
        return self._kind_data

    def name(self) -> String:
        """Return the column name for a named LOAD node (empty otherwise)."""
        return self._name

    def resolve_names(self, schema: Schema) raises -> Expr:
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
        if self._fused:
            try:
                return self._virt_fused_dtype(self._fused[])
            except:
                return None
        if self._tag == LITERAL:
            return self._value.value().type()
        return None

    def inputs(self) -> List[Expr]:
        var result = List[Expr](capacity=len(self._args))
        for ref a in self._args:
            result.append(a.copy())
        return result^

    def eval(self, batch: RecordBatch) raises -> AnyArray:
        """Evaluate this expression tree against *batch*, dispatching on tag.

        A boxed comptime-typed node (``FUSED`` tag) delegates to its own
        ``execute()`` via a trampoline, running as a single fused pass with
        no intermediate arrays even though it arrived through this
        type-erased tree.
        """
        if self._fused:
            return self._virt_fused_eval(self._fused.value(), batch)
        elif self._tag == LOAD:
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
        elif self._tag == IF_ELSE:
            return select(
                self._args[0].eval(batch),
                self._args[1].eval(batch),
                self._args[2].eval(batch),
            )
        elif self._tag == CAST:
            raise Error("Expr.eval: cast not implemented")
        else:
            raise Error("Expr.eval: unknown expression kind ", self._tag)

    def write_to[W: Writer](self, mut writer: W):
        if self._fused:
            writer.write(self._virt_fused_write(self._fused.value()))
        elif self._tag == LOAD:
            writer.write(t"input({self._kind_data})")
        elif self._tag == LITERAL:
            writer.write("literal(...)")
        elif self._tag >= ADD and self._tag <= OR:
            var op_name = String("?")
            if self._tag == ADD:
                op_name = "add"
            elif self._tag == SUB:
                op_name = "sub"
            elif self._tag == MUL:
                op_name = "mul"
            elif self._tag == DIV:
                op_name = "div"
            elif self._tag == EQ:
                op_name = "equal"
            elif self._tag == NE:
                op_name = "not_equal"
            elif self._tag == LT:
                op_name = "less"
            elif self._tag == LE:
                op_name = "less_equal"
            elif self._tag == GT:
                op_name = "greater"
            elif self._tag == GE:
                op_name = "greater_equal"
            elif self._tag == AND:
                op_name = "and"
            elif self._tag == OR:
                op_name = "or"
            writer.write(t"{op_name}(")
            if len(self._args) >= 1:
                self._args[0].write_to(writer)
            writer.write(t", ")
            if len(self._args) >= 2:
                self._args[1].write_to(writer)
            writer.write(t")")
        elif self._tag >= NEG and self._tag <= NOT:
            var op_name = String("?")
            if self._tag == NEG:
                op_name = "neg"
            elif self._tag == ABS:
                op_name = "abs"
            elif self._tag == NOT:
                op_name = "not"
            writer.write(t"{op_name}(")
            if len(self._args) >= 1:
                self._args[0].write_to(writer)
            writer.write(t")")
        elif self._tag == IS_NULL:
            writer.write(t"is_null(")
            if len(self._args) >= 1:
                self._args[0].write_to(writer)
            writer.write(t")")
        elif self._tag == IF_ELSE:
            writer.write(t"if_else(")
            if len(self._args) >= 1:
                self._args[0].write_to(writer)
            writer.write(t", ")
            if len(self._args) >= 2:
                self._args[1].write_to(writer)
            writer.write(t", ")
            if len(self._args) >= 3:
                self._args[2].write_to(writer)
            writer.write(t")")
        elif self._tag == CAST:
            writer.write(t"cast(")
            if len(self._args) >= 1:
                self._args[0].write_to(writer)
            writer.write(t")")
        else:
            writer.write(t"<unknown>({self._tag})")

    # Operator overloads (methods on Expr)
    def __add__(self, rhs: Expr) -> Expr:
        return Expr(
            tag=ADD,
            args=[self.copy(), rhs.copy()],
            kind_data=0,
            value=None,
            name=String(),
        )

    def __sub__(self, rhs: Expr) -> Expr:
        return Expr(
            tag=SUB,
            args=[self.copy(), rhs.copy()],
            kind_data=0,
            value=None,
            name=String(),
        )

    def __mul__(self, rhs: Expr) -> Expr:
        return Expr(
            tag=MUL,
            args=[self.copy(), rhs.copy()],
            kind_data=0,
            value=None,
            name=String(),
        )

    def __truediv__(self, rhs: Expr) -> Expr:
        return Expr(
            tag=DIV,
            args=[self.copy(), rhs.copy()],
            kind_data=0,
            value=None,
            name=String(),
        )

    def __gt__(self, rhs: Expr) -> Expr:
        return Expr(
            tag=GT,
            args=[self.copy(), rhs.copy()],
            kind_data=0,
            value=None,
            name=String(),
        )

    def __lt__(self, rhs: Expr) -> Expr:
        return Expr(
            tag=LT,
            args=[self.copy(), rhs.copy()],
            kind_data=0,
            value=None,
            name=String(),
        )

    def __ge__(self, rhs: Expr) -> Expr:
        return Expr(
            tag=GE,
            args=[self.copy(), rhs.copy()],
            kind_data=0,
            value=None,
            name=String(),
        )

    def __le__(self, rhs: Expr) -> Expr:
        return Expr(
            tag=LE,
            args=[self.copy(), rhs.copy()],
            kind_data=0,
            value=None,
            name=String(),
        )

    def __eq__(self, rhs: Expr) -> Expr:
        return Expr(
            tag=EQ,
            args=[self.copy(), rhs.copy()],
            kind_data=0,
            value=None,
            name=String(),
        )

    def __ne__(self, rhs: Expr) -> Expr:
        return Expr(
            tag=NE,
            args=[self.copy(), rhs.copy()],
            kind_data=0,
            value=None,
            name=String(),
        )

    def __neg__(self) -> Expr:
        return Expr(
            tag=NEG, args=[self.copy()], kind_data=0, value=None, name=String()
        )

    def __invert__(self) -> Expr:
        return Expr(
            tag=NOT, args=[self.copy()], kind_data=0, value=None, name=String()
        )

    def __and__(self, rhs: Expr) -> Expr:
        return Expr(
            tag=AND,
            args=[self.copy(), rhs.copy()],
            kind_data=0,
            value=None,
            name=String(),
        )

    def __or__(self, rhs: Expr) -> Expr:
        return Expr(
            tag=OR,
            args=[self.copy(), rhs.copy()],
            kind_data=0,
            value=None,
            name=String(),
        )

    def is_null(self) -> Expr:
        return Expr(
            tag=IS_NULL,
            args=[self.copy()],
            kind_data=0,
            value=None,
            name=String(),
        )

    def abs(self) -> Expr:
        return Expr(
            tag=ABS, args=[self.copy()], kind_data=0, value=None, name=String()
        )

    def cast(self, to: AnyDataType) -> Expr:
        return Expr(
            tag=CAST, args=[self.copy()], kind_data=0, value=None, name=String()
        )


# ---------------------------------------------------------------------------
# Factory functions (return Expr)
# ---------------------------------------------------------------------------


def col(index: Int) -> Expr:
    """Reference to the ``index``-th input column."""
    return Expr(
        tag=LOAD,
        args=List[Expr](),
        kind_data=UInt8(index),
        value=None,
        name=String(),
    )


def col(var name: String) -> Expr:
    """Reference to a named column."""
    return Expr(
        tag=LOAD, args=List[Expr](), kind_data=0, value=None, name=name^
    )


def lit[T: NumericType](value: Scalar[T.native]) raises -> Expr:
    """A scalar constant."""
    return Expr(
        tag=LITERAL,
        args=List[Expr](),
        kind_data=0,
        value=PrimitiveScalar[T](value).to_any(),
        name=String(),
    )


def if_else(cond: Expr, then_: Expr, else_: Expr) -> Expr:
    """Element-wise conditional."""
    return Expr(
        tag=IF_ELSE,
        args=[cond.copy(), then_.copy(), else_.copy()],
        kind_data=0,
        value=None,
        name=String(),
    )
