"""Type-erased runtime expression nodes for the marrow expression system.

``Expr`` is the runtime counterpart to the comptime-typed layer in
``marrow.aot.values``.  It exists so that query plans can be built and
executed without knowing concrete comptime types — this is what the Python
bindings (and any other runtime-typed caller) drive.  A single ``Expr`` node
carries a tag plus its child args, and dispatches its own execution by tag in
``eval()`` — there is no separate "processor" hierarchy mirroring the tree.

A comptime-typed node from ``marrow.aot.values`` can be boxed into an
``Expr`` via the ``Expr(value)`` constructor (tag ``FUSED``); ``eval()``/
``dtype()``/``write_to()`` on a boxed node all delegate back to the concrete
comptime node through trampolines, so a fused subtree keeps its single fused
pass even when driven through this type-erased path. This is the one
dependency between the two ``marrow.aot`` / ``marrow.dyn`` packages — ``dyn``
imports the ``NumericValue``/``BoolValue`` traits from ``aot.values`` to
declare the boxing constructors' generic bounds; ``aot`` never imports
anything from ``dyn``.

Factory functions
-----------------
``col(index)``  / ``col(name)`` — column reference
``lit[T](value)``               — typed scalar literal
``if_else(cond, then_, else_)`` — conditional

Operator overloads on ``Expr``: ``+``, ``-``, ``*``, ``/``, ``>``,
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
CAST - Type cast (not yet implemented — see Expr.eval)
FUSED - Carries a boxed comptime-typed node (see marrow.aot.values)
LENGTH - String byte length (dispatches to kernels.string.string_lengths)
"""

from std.memory import ArcPointer
from marrow.arrays import AnyArray
from marrow.dtypes import AnyDataType, NumericType
from marrow.scalars import AnyScalar, PrimitiveScalar
from marrow.schema import Schema
from marrow.tabular import RecordBatch
from marrow.aot.values import Value, NumericValue, BoolValue
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
comptime FUSED: UInt8 = 20
"""Tag for Expr nodes that carry a boxed comptime-typed node in _fused."""
comptime LENGTH: UInt8 = 21


# ---------------------------------------------------------------------------
# Trampoline helpers for boxing comptime-typed expressions into Expr
# ---------------------------------------------------------------------------


def _fused_dtype_tramp[
    T: NumericValue
](ptr: ArcPointer[NoneType],) -> Optional[AnyDataType]:
    """Thin trampoline: delegate dtype() to a concrete NumericValue."""
    var typed = rebind[ArcPointer[T]](ptr)
    return typed[].dtype()


def _fused_write_tramp[
    T: NumericValue
](ptr: ArcPointer[NoneType],) -> String:
    """Thin trampoline: delegate write_to() to a concrete NumericValue."""
    var typed = rebind[ArcPointer[T]](ptr)
    var s = String()
    typed[].write_to(s)
    return s^


def _fused_eval_tramp[
    T: NumericValue
](ptr: ArcPointer[NoneType], batch: RecordBatch) raises -> AnyArray:
    """Thin trampoline: delegate execute() to a concrete NumericValue."""
    var typed = rebind[ArcPointer[T]](ptr)
    return typed[].execute(batch).to_any()


def _fused_dtype_tramp_bool[
    T: BoolValue
](ptr: ArcPointer[NoneType],) -> Optional[AnyDataType]:
    """Thin trampoline: delegate dtype() to a concrete BoolValue."""
    var typed = rebind[ArcPointer[T]](ptr)
    return typed[].dtype()


def _fused_write_tramp_bool[
    T: BoolValue
](ptr: ArcPointer[NoneType],) -> String:
    """Thin trampoline: delegate write_to() to a concrete BoolValue."""
    var typed = rebind[ArcPointer[T]](ptr)
    var s = String()
    typed[].write_to(s)
    return s^


def _fused_eval_tramp_bool[
    T: BoolValue
](ptr: ArcPointer[NoneType], batch: RecordBatch) raises -> AnyArray:
    """Thin trampoline: delegate execute() to a concrete BoolValue."""
    var typed = rebind[ArcPointer[T]](ptr)
    return typed[].execute(batch).to_any()


# ---------------------------------------------------------------------------
# Expr - unified n-ary term expression node
# ---------------------------------------------------------------------------


struct Expr(
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

    def __init__[T: NumericValue](out self, value: T):
        """Box a comptime-typed expression node into a runtime Expr.

        The resulting ``Expr`` carries the comptime node in its ``_fused``
        slot, so ``dtype()``, ``write_to()``, and ``eval()`` all delegate to
        the comptime-typed implementation — including a single fused pass
        for ``eval()``, with no intermediate arrays.
        """
        var ptr = ArcPointer[T](value.copy())
        self._tag = FUSED
        self._args = List[Expr]()
        self._kind_data = 0
        self._value = None
        self._name = String()
        self._fused = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_fused_dtype = _fused_dtype_tramp[T]
        self._virt_fused_write = _fused_write_tramp[T]
        self._virt_fused_eval = _fused_eval_tramp[T]

    def __init__[T: BoolValue](out self, value: T):
        """Box a comptime-typed predicate node (``Lt``/``Gt``/``Eq``) into a
        runtime ``Expr``. Mirrors the ``NumericValue`` overload above — same
        ``FUSED`` tag, same trampoline mechanism, just a ``BoolValue``
        trampoline set so ``eval()`` produces a bit-packed ``BoolArray``
        instead of a ``PrimitiveArray``.
        """
        var ptr = ArcPointer[T](value.copy())
        self._tag = FUSED
        self._args = List[Expr]()
        self._kind_data = 0
        self._value = None
        self._name = String()
        self._fused = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_fused_dtype = _fused_dtype_tramp_bool[T]
        self._virt_fused_write = _fused_write_tramp_bool[T]
        self._virt_fused_eval = _fused_eval_tramp_bool[T]

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

    # Explicit (empty) destructor so this self-referential struct
    # (`_args: List[Expr]`) is ImplicitlyDeletable; fields are still destroyed
    # automatically after the body runs.
    def __del__(deinit self):
        pass

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
        elif self._tag == LENGTH:
            return string_lengths(self._args[0].eval(batch)).to_any()
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
        if self._fused:
            writer.write(self._virt_fused_write(self._fused.value()))
        elif self._tag == LOAD:
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
    def _binary(self, tag: UInt8, rhs: Expr) -> Expr:
        return Expr(
            tag=tag,
            args=[self.copy(), rhs.copy()],
            kind_data=0,
            value=None,
            name=String(),
        )

    def _unary(self, tag: UInt8) -> Expr:
        return Expr(
            tag=tag,
            args=[self.copy()],
            kind_data=0,
            value=None,
            name=String(),
        )

    # Operator overloads (methods on Expr)
    def __add__(self, rhs: Expr) -> Expr:
        return self._binary(ADD, rhs)

    def __sub__(self, rhs: Expr) -> Expr:
        return self._binary(SUB, rhs)

    def __mul__(self, rhs: Expr) -> Expr:
        return self._binary(MUL, rhs)

    def __truediv__(self, rhs: Expr) -> Expr:
        return self._binary(DIV, rhs)

    def __gt__(self, rhs: Expr) -> Expr:
        return self._binary(GT, rhs)

    def __lt__(self, rhs: Expr) -> Expr:
        return self._binary(LT, rhs)

    def __ge__(self, rhs: Expr) -> Expr:
        return self._binary(GE, rhs)

    def __le__(self, rhs: Expr) -> Expr:
        return self._binary(LE, rhs)

    def __eq__(self, rhs: Expr) -> Expr:
        return self._binary(EQ, rhs)

    def __ne__(self, rhs: Expr) -> Expr:
        return self._binary(NE, rhs)

    def __and__(self, rhs: Expr) -> Expr:
        return self._binary(AND, rhs)

    def __or__(self, rhs: Expr) -> Expr:
        return self._binary(OR, rhs)

    def __neg__(self) -> Expr:
        return self._unary(NEG)

    def __invert__(self) -> Expr:
        return self._unary(NOT)

    def is_null(self) -> Expr:
        return self._unary(IS_NULL)

    def abs(self) -> Expr:
        return self._unary(ABS)

    def length(self) -> Expr:
        return self._unary(LENGTH)

    def cast(self, to: AnyDataType) -> Expr:
        return self._unary(CAST)


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
