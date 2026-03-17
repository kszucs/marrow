"""Scalar expression nodes for the marrow expression system.

``Value``  — the trait every scalar expression node must implement.
``Expr``   — the type-erased, ArcPointer-backed container (analogous to
             ``AnyBuilder``).  Copies are O(1) ref-count bumps; the
             expression DAG is fully shareable.

Concrete node types
-------------------
``Column``  — input column reference (LOAD)
``Literal`` — scalar constant broadcast to array length (LITERAL)
``Binary``  — binary arithmetic / comparison / boolean (ADD … OR)
``Unary``   — unary arithmetic / boolean (NEG, ABS, NOT)
``IsNull``  — null check (IS_NULL)
``IfElse``  — conditional select (IF_ELSE)
``Cast``    — explicit type cast (CAST)

Node-kind / op constants
------------------------
BinaryOp : ADD SUB MUL DIV EQ NE LT LE GT GE AND OR
UnaryOp  : NEG ABS NOT
Other    : LOAD LITERAL IS_NULL IF_ELSE CAST

Dispatch hints
--------------
DISPATCH_AUTO — executor decides
DISPATCH_CPU  — always CPU SIMD path
DISPATCH_GPU  — always GPU; raises if no accelerator

E-graph compatibility
---------------------
Nodes are immutable after construction and are designed to support hash
consing (structural hash / equality) as a prerequisite for equality
saturation.  A future ``ExprContext`` intern table and ``EGraph`` runner
can be added without changing node representations or ``Rewrite``
rule definitions.
"""

from std.memory import ArcPointer
from marrow.dtypes import DataType


# ---------------------------------------------------------------------------
# Node-kind / op constants
# ---------------------------------------------------------------------------

# Leaf nodes
comptime LOAD:    UInt8 = 0
comptime LITERAL: UInt8 = 1

# BinaryOp values — also used as kind() for Binary nodes
comptime ADD: UInt8 = 2
comptime SUB: UInt8 = 3
comptime MUL: UInt8 = 4
comptime DIV: UInt8 = 5
comptime EQ:  UInt8 = 6
comptime NE:  UInt8 = 7
comptime LT:  UInt8 = 8
comptime LE:  UInt8 = 9
comptime GT:  UInt8 = 10
comptime GE:  UInt8 = 11
comptime AND: UInt8 = 12
comptime OR:  UInt8 = 13

# UnaryOp values — also used as kind() for Unary nodes
comptime NEG: UInt8 = 14
comptime ABS: UInt8 = 15
comptime NOT: UInt8 = 16

# Other nodes
comptime IS_NULL: UInt8 = 17
comptime IF_ELSE: UInt8 = 18
comptime CAST:    UInt8 = 19


# ---------------------------------------------------------------------------
# Dispatch hint constants
# ---------------------------------------------------------------------------

comptime DISPATCH_AUTO: UInt8 = 0
"""Executor decides: GPU if DeviceContext present and array large enough."""

comptime DISPATCH_CPU: UInt8 = 1
"""Always use CPU SIMD path; DeviceContext is ignored."""

comptime DISPATCH_GPU: UInt8 = 2
"""Always use GPU; raises if no accelerator or no DeviceContext."""


# ---------------------------------------------------------------------------
# Value trait — interface every concrete expression node must implement
# ---------------------------------------------------------------------------


trait Value(ImplicitlyDestructible, Movable):
    """Interface for immutable scalar expression nodes.

    Nodes are designed for e-graph compatibility: they must be immutable
    after construction.  Implementors should also implement ``Hashable``
    and ``Equatable`` (structural hash / equality over fields) to enable
    hash consing and equality saturation in a future ``EGraph``.
    """

    fn kind(self) -> UInt8:
        """Return the node-kind constant (LOAD, ADD, NEG, …)."""
        ...

    fn dtype(self) -> Optional[DataType]:
        """Return the output data type, or None if not yet inferred."""
        ...

    fn inputs(self) -> List[Expr]:
        """Return child expressions (empty for leaf nodes)."""
        ...

    fn write_to[W: Writer](self, mut writer: W):
        """Format this node for display (children formatted recursively)."""
        ...


# ---------------------------------------------------------------------------
# Expr — type-erased, ArcPointer-backed expression container
# ---------------------------------------------------------------------------


struct Expr(ImplicitlyCopyable, Movable, Writable):
    """Type-erased scalar expression node.

    Wraps any ``Value``-conforming type on the heap behind an
    ``ArcPointer`` so copies are O(1) ref-count bumps.  Composite nodes
    (``Binary``, ``IfElse``) hold ``Expr`` children — the entire DAG is
    shared by reference.
    """

    var _data: ArcPointer[NoneType]
    var _virt_kind: fn(ArcPointer[NoneType]) -> UInt8
    var _virt_dtype: fn(ArcPointer[NoneType]) -> Optional[DataType]
    var _virt_inputs: fn(ArcPointer[NoneType]) -> List[Expr]
    var _virt_write_to_string: fn(ArcPointer[NoneType]) -> String
    var _virt_drop: fn(var ArcPointer[NoneType])
    var dispatch: UInt8
    """Dispatch hint for the executor (DISPATCH_AUTO / CPU / GPU)."""

    # --- trampolines ---

    @staticmethod
    fn _tramp_kind[T: Value](ptr: ArcPointer[NoneType]) -> UInt8:
        return rebind[ArcPointer[T]](ptr)[].kind()

    @staticmethod
    fn _tramp_dtype[
        T: Value
    ](ptr: ArcPointer[NoneType]) -> Optional[DataType]:
        return rebind[ArcPointer[T]](ptr)[].dtype()

    @staticmethod
    fn _tramp_inputs[T: Value](ptr: ArcPointer[NoneType]) -> List[Expr]:
        return rebind[ArcPointer[T]](ptr)[].inputs()

    @staticmethod
    fn _tramp_write_to_string[T: Value](ptr: ArcPointer[NoneType]) -> String:
        var s = String()
        rebind[ArcPointer[T]](ptr)[].write_to(s)
        return s^

    @staticmethod
    fn _tramp_drop[T: Value](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    # --- construction ---

    @implicit
    fn __init__[T: Value](out self, var value: T):
        var ptr = ArcPointer(value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_kind = Self._tramp_kind[T]
        self._virt_dtype = Self._tramp_dtype[T]
        self._virt_inputs = Self._tramp_inputs[T]
        self._virt_write_to_string = Self._tramp_write_to_string[T]
        self._virt_drop = Self._tramp_drop[T]
        self.dispatch = DISPATCH_AUTO

    fn __copyinit__(out self, read copy: Self):
        self._data = copy._data.copy()
        self._virt_kind = copy._virt_kind
        self._virt_dtype = copy._virt_dtype
        self._virt_inputs = copy._virt_inputs
        self._virt_write_to_string = copy._virt_write_to_string
        self._virt_drop = copy._virt_drop
        self.dispatch = copy.dispatch

    # --- public API ---

    fn kind(self) -> UInt8:
        return self._virt_kind(self._data)

    fn dtype(self) -> Optional[DataType]:
        return self._virt_dtype(self._data)

    fn inputs(self) -> List[Expr]:
        return self._virt_inputs(self._data)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self._virt_write_to_string(self._data))

    fn with_dispatch(self, hint: UInt8) -> Expr:
        """Return a copy of this expression with the given dispatch hint."""
        var copy = self
        copy.dispatch = hint
        return copy^

    # --- downcast helpers ---

    fn downcast[T: Value](self) -> ArcPointer[T]:
        return rebind[ArcPointer[T]](self._data.copy())

    @always_inline
    fn as_column(self) -> ArcPointer[Column]:
        return self.downcast[Column]()

    @always_inline
    fn as_literal(self) -> ArcPointer[Literal]:
        return self.downcast[Literal]()

    @always_inline
    fn as_binary(self) -> ArcPointer[Binary]:
        return self.downcast[Binary]()

    @always_inline
    fn as_unary(self) -> ArcPointer[Unary]:
        return self.downcast[Unary]()

    @always_inline
    fn as_is_null(self) -> ArcPointer[IsNull]:
        return self.downcast[IsNull]()

    @always_inline
    fn as_if_else(self) -> ArcPointer[IfElse]:
        return self.downcast[IfElse]()

    @always_inline
    fn as_cast(self) -> ArcPointer[Cast]:
        return self.downcast[Cast]()

    # --- static factory methods (preserve existing call sites) ---

    @staticmethod
    fn input(idx: Int) -> Expr:
        """Reference to the ``idx``-th input array."""
        return Column(index=idx, name=String(), dtype_=None)

    @staticmethod
    fn literal[T: DataType](value: Scalar[T.native]) -> Expr:
        """A scalar constant broadcast to the length of the first input."""
        return Literal(value=Float64(value), dtype_=T)

    @staticmethod
    fn _binary(op: UInt8, var left: Expr, var right: Expr) -> Expr:
        return Binary(op=op, left=left^, right=right^)

    @staticmethod
    fn add(var left: Expr, var right: Expr) -> Expr:
        """Element-wise addition."""
        return Expr._binary(ADD, left^, right^)

    @staticmethod
    fn sub(var left: Expr, var right: Expr) -> Expr:
        """Element-wise subtraction."""
        return Expr._binary(SUB, left^, right^)

    @staticmethod
    fn mul(var left: Expr, var right: Expr) -> Expr:
        """Element-wise multiplication."""
        return Expr._binary(MUL, left^, right^)

    @staticmethod
    fn div(var left: Expr, var right: Expr) -> Expr:
        """Element-wise division."""
        return Expr._binary(DIV, left^, right^)

    @staticmethod
    fn equal(var left: Expr, var right: Expr) -> Expr:
        """Element-wise equality (→ bool array)."""
        return Expr._binary(EQ, left^, right^)

    @staticmethod
    fn not_equal(var left: Expr, var right: Expr) -> Expr:
        """Element-wise inequality (→ bool array)."""
        return Expr._binary(NE, left^, right^)

    @staticmethod
    fn less(var left: Expr, var right: Expr) -> Expr:
        """Element-wise less-than (→ bool array)."""
        return Expr._binary(LT, left^, right^)

    @staticmethod
    fn less_equal(var left: Expr, var right: Expr) -> Expr:
        """Element-wise less-or-equal (→ bool array)."""
        return Expr._binary(LE, left^, right^)

    @staticmethod
    fn greater(var left: Expr, var right: Expr) -> Expr:
        """Element-wise greater-than (→ bool array)."""
        return Expr._binary(GT, left^, right^)

    @staticmethod
    fn greater_equal(var left: Expr, var right: Expr) -> Expr:
        """Element-wise greater-or-equal (→ bool array)."""
        return Expr._binary(GE, left^, right^)

    @staticmethod
    fn and_(var left: Expr, var right: Expr) -> Expr:
        """Logical AND of two boolean expressions."""
        return Expr._binary(AND, left^, right^)

    @staticmethod
    fn or_(var left: Expr, var right: Expr) -> Expr:
        """Logical OR of two boolean expressions."""
        return Expr._binary(OR, left^, right^)

    @staticmethod
    fn _unary(op: UInt8, var child: Expr) -> Expr:
        return Unary(op=op, child=child^)

    @staticmethod
    fn neg(var child: Expr) -> Expr:
        """Element-wise negation."""
        return Expr._unary(NEG, child^)

    @staticmethod
    fn abs_(var child: Expr) -> Expr:
        """Element-wise absolute value."""
        return Expr._unary(ABS, child^)

    @staticmethod
    fn not_(var child: Expr) -> Expr:
        """Logical NOT of a boolean expression."""
        return Expr._unary(NOT, child^)

    @staticmethod
    fn is_null(var child: Expr) -> Expr:
        """True where the child expression produces a null value."""
        return IsNull(child=child^)

    @staticmethod
    fn if_else(var cond: Expr, var then_: Expr, var else_: Expr) -> Expr:
        """Element-wise conditional: result[i] = then_[i] if cond[i] else else_[i]."""
        return IfElse(cond=cond^, then_=then_^, else_=else_^)

    fn __del__(deinit self):
        self._virt_drop(self._data^)


# ---------------------------------------------------------------------------
# Concrete expression nodes
# ---------------------------------------------------------------------------


struct Column(Value):
    """Input column reference (LOAD node).

    ``index``  — positional index into the executor's input array list.
    ``name``   — optional display name (may be empty).
    ``dtype_`` — declared data type; None until type inference runs.
    """

    var index: Int
    var name: String
    var dtype_: Optional[DataType]

    fn __init__(
        out self, *, index: Int, var name: String, dtype_: Optional[DataType]
    ):
        self.index = index
        self.name = name^
        self.dtype_ = dtype_

    fn kind(self) -> UInt8:
        return LOAD

    fn dtype(self) -> Optional[DataType]:
        return self.dtype_

    fn inputs(self) -> List[Expr]:
        return List[Expr]()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(t"input({self.index})")


struct Literal(Value):
    """Scalar constant broadcast to the length of the first input (LITERAL node)."""

    var value: Float64
    var dtype_: DataType

    fn __init__(out self, *, value: Float64, dtype_: DataType):
        self.value = value
        self.dtype_ = dtype_

    fn kind(self) -> UInt8:
        return LITERAL

    fn dtype(self) -> Optional[DataType]:
        return self.dtype_

    fn inputs(self) -> List[Expr]:
        return List[Expr]()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(t"literal({self.value})")


struct Binary(Value):
    """Binary operation node (arithmetic, comparison, boolean).

    ``op``    — BinaryOp constant (ADD, SUB, … OR).
    ``left``  — left child expression.
    ``right`` — right child expression.

    ``kind()`` returns ``op`` so the executor can dispatch on the kind
    constant directly without downcasting.
    """

    var op: UInt8
    var left: Expr
    var right: Expr

    fn __init__(out self, *, op: UInt8, var left: Expr, var right: Expr):
        self.op = op
        self.left = left^
        self.right = right^

    fn kind(self) -> UInt8:
        return self.op

    fn dtype(self) -> Optional[DataType]:
        return None  # filled in by type inference

    fn inputs(self) -> List[Expr]:
        var result = List[Expr](capacity=2)
        result.append(self.left)
        result.append(self.right)
        return result^

    fn write_to[W: Writer](self, mut writer: W):
        if self.op == ADD:
            writer.write(t"add(")
        elif self.op == SUB:
            writer.write(t"sub(")
        elif self.op == MUL:
            writer.write(t"mul(")
        elif self.op == DIV:
            writer.write(t"div(")
        elif self.op == EQ:
            writer.write(t"equal(")
        elif self.op == NE:
            writer.write(t"not_equal(")
        elif self.op == LT:
            writer.write(t"less(")
        elif self.op == LE:
            writer.write(t"less_equal(")
        elif self.op == GT:
            writer.write(t"greater(")
        elif self.op == GE:
            writer.write(t"greater_equal(")
        elif self.op == AND:
            writer.write(t"and_(")
        elif self.op == OR:
            writer.write(t"or_(")
        else:
            writer.write(t"<unknown_binary>(")
        self.left.write_to(writer)
        writer.write(t", ")
        self.right.write_to(writer)
        writer.write(t")")


struct Unary(Value):
    """Unary operation node (arithmetic negation, absolute value, logical NOT).

    ``op``    — UnaryOp constant (NEG, ABS, NOT).
    ``child`` — operand expression.

    ``kind()`` returns ``op``.
    """

    var op: UInt8
    var child: Expr

    fn __init__(out self, *, op: UInt8, var child: Expr):
        self.op = op
        self.child = child^

    fn kind(self) -> UInt8:
        return self.op

    fn dtype(self) -> Optional[DataType]:
        return None  # filled in by type inference

    fn inputs(self) -> List[Expr]:
        var result = List[Expr](capacity=1)
        result.append(self.child)
        return result^

    fn write_to[W: Writer](self, mut writer: W):
        if self.op == NEG:
            writer.write(t"neg(")
        elif self.op == ABS:
            writer.write(t"abs(")
        elif self.op == NOT:
            writer.write(t"not_(")
        else:
            writer.write(t"<unknown_unary>(")
        self.child.write_to(writer)
        writer.write(t")")


struct IsNull(Value):
    """Null check — produces a bool array: True where child is null."""

    var child: Expr

    fn __init__(out self, *, var child: Expr):
        self.child = child^

    fn kind(self) -> UInt8:
        return IS_NULL

    fn dtype(self) -> Optional[DataType]:
        return None  # bool_ after type inference

    fn inputs(self) -> List[Expr]:
        var result = List[Expr](capacity=1)
        result.append(self.child)
        return result^

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(t"is_null(")
        self.child.write_to(writer)
        writer.write(t")")


struct IfElse(Value):
    """Element-wise conditional: result[i] = then_[i] if cond[i] else else_[i]."""

    var cond: Expr
    var then_: Expr
    var else_: Expr

    fn __init__(out self, *, var cond: Expr, var then_: Expr, var else_: Expr):
        self.cond = cond^
        self.then_ = then_^
        self.else_ = else_^

    fn kind(self) -> UInt8:
        return IF_ELSE

    fn dtype(self) -> Optional[DataType]:
        return None  # filled in by type inference

    fn inputs(self) -> List[Expr]:
        var result = List[Expr](capacity=3)
        result.append(self.cond)
        result.append(self.then_)
        result.append(self.else_)
        return result^

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(t"if_else(")
        self.cond.write_to(writer)
        writer.write(t", ")
        self.then_.write_to(writer)
        writer.write(t", ")
        self.else_.write_to(writer)
        writer.write(t")")


struct Cast(Value):
    """Explicit type cast."""

    var child: Expr
    var to: DataType

    fn __init__(out self, *, var child: Expr, to: DataType):
        self.child = child^
        self.to = to

    fn kind(self) -> UInt8:
        return CAST

    fn dtype(self) -> Optional[DataType]:
        return self.to

    fn inputs(self) -> List[Expr]:
        var result = List[Expr](capacity=1)
        result.append(self.child)
        return result^

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(t"cast(")
        self.child.write_to(writer)
        writer.write(t", {self.to})")
