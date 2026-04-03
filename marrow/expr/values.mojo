"""Scalar expression nodes for the marrow expression system.

``Value``       — the trait every scalar expression node must implement.
``AnyValue``    — the type-erased, ArcPointer-backed container.
Concrete node types
-------------------
``Column``  — input column reference (LOAD)
``Literal`` — scalar constant stored as a length-1 AnyArray (LITERAL)
``Binary``  — binary arithmetic / comparison / boolean (ADD … OR)
``Unary``   — unary arithmetic / boolean (NEG, ABS, NOT)
``IsNull``  — null check (IS_NULL)
``IfElse``  — conditional select (IF_ELSE)
``Cast``    — explicit type cast (CAST)

Factory functions
-----------------
``col(index)``  / ``col(name)`` — column reference
``lit[T](value)``               — typed scalar literal
``if_else(cond, then_, else_)`` — conditional

Operator overloads on ``AnyValue``: ``+``, ``-``, ``*``, ``/``, ``>``,
``<``, ``>=``, ``<=``, ``==``, ``!=``, ``&``, ``|``, ``~`` (NOT),
unary ``-``.  Instance methods: ``.abs()``, ``.is_null()``, ``.cast(to)``.
"""

from std.memory import ArcPointer
from marrow.arrays import AnyArray, PrimitiveArray
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import ArrowType, PrimitiveType
from marrow.schema import Schema


# ---------------------------------------------------------------------------
# Node-kind / op constants
# ---------------------------------------------------------------------------

# Leaf nodes
comptime LOAD: UInt8 = 0
comptime LITERAL: UInt8 = 1

# BinaryOp values — also used as kind() for Binary nodes
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

# UnaryOp values — also used as kind() for Unary nodes
comptime NEG: UInt8 = 14
comptime ABS: UInt8 = 15
comptime NOT: UInt8 = 16

# Other nodes
comptime IS_NULL: UInt8 = 17
comptime IF_ELSE: UInt8 = 18
comptime CAST: UInt8 = 19


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

    def kind(self) -> UInt8:
        """Return the node-kind constant (LOAD, ADD, NEG, …)."""
        ...

    def dtype(self) -> Optional[ArrowType]:
        """Return the output data type, or None if not yet inferred."""
        ...

    def inputs(self) -> List[AnyValue]:
        """Return child expressions (empty for leaf nodes)."""
        ...

    def write_to[W: Writer](self, mut writer: W):
        """Format this node for display (children formatted recursively)."""
        ...


# ---------------------------------------------------------------------------
# AnyValue — type-erased, ArcPointer-backed expression container
# ---------------------------------------------------------------------------


struct AnyValue(ImplicitlyCopyable, Movable, Writable):
    """Type-erased scalar expression node.

    Wraps any ``Value``-conforming type on the heap behind an
    ``ArcPointer`` so copies are O(1) ref-count bumps.  Composite nodes
    (``Binary``, ``IfElse``) hold ``AnyValue`` children — the entire DAG is
    shared by reference.
    """

    var _data: ArcPointer[NoneType]
    var _virt_kind: def(ArcPointer[NoneType]) -> UInt8
    var _virt_dtype: def(ArcPointer[NoneType]) -> Optional[ArrowType]
    var _virt_inputs: def(ArcPointer[NoneType]) -> List[AnyValue]
    var _virt_write_to_string: def(ArcPointer[NoneType]) -> String
    var _virt_drop: def(var ArcPointer[NoneType])
    var dispatch: UInt8
    """Dispatch hint for the executor (DISPATCH_AUTO / CPU / GPU)."""

    # --- trampolines ---

    @staticmethod
    def _tramp_kind[T: Value](ptr: ArcPointer[NoneType]) -> UInt8:
        return rebind[ArcPointer[T]](ptr)[].kind()

    @staticmethod
    def _tramp_dtype[T: Value](ptr: ArcPointer[NoneType]) -> Optional[ArrowType]:
        return rebind[ArcPointer[T]](ptr)[].dtype()

    @staticmethod
    def _tramp_inputs[T: Value](ptr: ArcPointer[NoneType]) -> List[AnyValue]:
        return rebind[ArcPointer[T]](ptr)[].inputs()

    @staticmethod
    def _tramp_write_to_string[T: Value](ptr: ArcPointer[NoneType]) -> String:
        var s = String()
        rebind[ArcPointer[T]](ptr)[].write_to(s)
        return s^

    @staticmethod
    def _tramp_drop[T: Value](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    # --- construction ---

    @implicit
    def __init__[T: Value](out self, var value: T):
        var ptr = ArcPointer(value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_kind = Self._tramp_kind[T]
        self._virt_dtype = Self._tramp_dtype[T]
        self._virt_inputs = Self._tramp_inputs[T]
        self._virt_write_to_string = Self._tramp_write_to_string[T]
        self._virt_drop = Self._tramp_drop[T]
        self.dispatch = DISPATCH_AUTO

    def __init__(out self, *, copy: Self):
        self._data = copy._data
        self._virt_kind = copy._virt_kind
        self._virt_dtype = copy._virt_dtype
        self._virt_inputs = copy._virt_inputs
        self._virt_write_to_string = copy._virt_write_to_string
        self._virt_drop = copy._virt_drop
        self.dispatch = copy.dispatch

    # --- public API ---

    def kind(self) -> UInt8:
        return self._virt_kind(self._data)

    def dtype(self) -> Optional[ArrowType]:
        return self._virt_dtype(self._data)

    def inputs(self) -> List[AnyValue]:
        return self._virt_inputs(self._data)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._virt_write_to_string(self._data))

    def with_dispatch(self, hint: UInt8) -> AnyValue:
        """Return a copy of this expression with the given dispatch hint."""
        var copy = self
        copy.dispatch = hint
        return copy^

    # --- downcast helpers ---

    def downcast[T: Value](self) -> ArcPointer[T]:
        return rebind[ArcPointer[T]](self._data.copy())

    # --- operator overloads (ibis-style) ---

    def __add__(self, other: AnyValue) -> AnyValue:
        return Binary(op=ADD, left=self, right=other)

    def __sub__(self, other: AnyValue) -> AnyValue:
        return Binary(op=SUB, left=self, right=other)

    def __mul__(self, other: AnyValue) -> AnyValue:
        return Binary(op=MUL, left=self, right=other)

    def __truediv__(self, other: AnyValue) -> AnyValue:
        return Binary(op=DIV, left=self, right=other)

    def __gt__(self, other: AnyValue) -> AnyValue:
        return Binary(op=GT, left=self, right=other)

    def __lt__(self, other: AnyValue) -> AnyValue:
        return Binary(op=LT, left=self, right=other)

    def __ge__(self, other: AnyValue) -> AnyValue:
        return Binary(op=GE, left=self, right=other)

    def __le__(self, other: AnyValue) -> AnyValue:
        return Binary(op=LE, left=self, right=other)

    def __eq__(self, other: AnyValue) -> AnyValue:
        return Binary(op=EQ, left=self, right=other)

    def __ne__(self, other: AnyValue) -> AnyValue:
        return Binary(op=NE, left=self, right=other)

    def __neg__(self) -> AnyValue:
        return Unary(op=NEG, child=self)

    def __invert__(self) -> AnyValue:
        """Logical NOT."""
        return Unary(op=NOT, child=self)

    def __and__(self, other: AnyValue) -> AnyValue:
        return Binary(op=AND, left=self, right=other)

    def __or__(self, other: AnyValue) -> AnyValue:
        return Binary(op=OR, left=self, right=other)

    def is_null(self) -> AnyValue:
        """True where this expression produces a null value."""
        return IsNull(child=self)

    def abs(self) -> AnyValue:
        """Element-wise absolute value."""
        return Unary(op=ABS, child=self)

    def cast(self, to: ArrowType) -> AnyValue:
        """Explicit type cast."""
        return Cast(child=self, to=to)

    def __del__(deinit self):
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
    var dtype_: Optional[ArrowType]

    def __init__(
        out self, *, index: Int, var name: String, dtype_: Optional[ArrowType]
    ):
        self.index = index
        self.name = name^
        self.dtype_ = dtype_.copy()

    def kind(self) -> UInt8:
        return LOAD

    def dtype(self) -> Optional[ArrowType]:
        return self.dtype_.copy()

    def inputs(self) -> List[AnyValue]:
        return List[AnyValue]()

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"input({self.index})")


struct Literal(Value):
    """Scalar constant broadcast to the length of the first input (LITERAL node).

    Stores the scalar as a length-1 AnyArray, supporting any dtype (numeric,
    string, bool).  Similar to arrow-rs ``Scalar<T>``.
    """

    var value: AnyArray

    def __init__(out self, *, var value: AnyArray):
        self.value = value^

    def kind(self) -> UInt8:
        return LITERAL

    def dtype(self) -> Optional[ArrowType]:
        return self.value.dtype()

    def inputs(self) -> List[AnyValue]:
        return List[AnyValue]()

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"literal(...)")


struct Binary(Value):
    """Binary operation node (arithmetic, comparison, boolean).

    ``op``    — BinaryOp constant (ADD, SUB, … OR).
    ``left``  — left child expression.
    ``right`` — right child expression.

    ``kind()`` returns ``op`` so the executor can dispatch on the kind
    constant directly without downcasting.
    """

    var op: UInt8
    var left: AnyValue
    var right: AnyValue

    def __init__(
        out self, *, op: UInt8, var left: AnyValue, var right: AnyValue
    ):
        self.op = op
        self.left = left^
        self.right = right^

    def kind(self) -> UInt8:
        return self.op

    def dtype(self) -> Optional[ArrowType]:
        return None  # filled in by type inference

    def inputs(self) -> List[AnyValue]:
        return [self.left, self.right]

    def write_to[W: Writer](self, mut writer: W):
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
    var child: AnyValue

    def __init__(out self, *, op: UInt8, var child: AnyValue):
        self.op = op
        self.child = child^

    def kind(self) -> UInt8:
        return self.op

    def dtype(self) -> Optional[ArrowType]:
        return None  # filled in by type inference

    def inputs(self) -> List[AnyValue]:
        return [self.child]

    def write_to[W: Writer](self, mut writer: W):
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

    var child: AnyValue

    def __init__(out self, *, var child: AnyValue):
        self.child = child^

    def kind(self) -> UInt8:
        return IS_NULL

    def dtype(self) -> Optional[ArrowType]:
        return None  # bool_ after type inference

    def inputs(self) -> List[AnyValue]:
        return [self.child]

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"is_null(")
        self.child.write_to(writer)
        writer.write(t")")


struct IfElse(Value):
    """Element-wise conditional: result[i] = then_[i] if cond[i] else else_[i].
    """

    var cond: AnyValue
    var then_: AnyValue
    var else_: AnyValue

    def __init__(
        out self,
        *,
        var cond: AnyValue,
        var then_: AnyValue,
        var else_: AnyValue,
    ):
        self.cond = cond^
        self.then_ = then_^
        self.else_ = else_^

    def kind(self) -> UInt8:
        return IF_ELSE

    def dtype(self) -> Optional[ArrowType]:
        return None  # filled in by type inference

    def inputs(self) -> List[AnyValue]:
        return [self.cond, self.then_, self.else_]

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"if_else(")
        self.cond.write_to(writer)
        writer.write(t", ")
        self.then_.write_to(writer)
        writer.write(t", ")
        self.else_.write_to(writer)
        writer.write(t")")


struct Cast(Value):
    """Explicit type cast."""

    var child: AnyValue
    var to: ArrowType

    def __init__(out self, *, var child: AnyValue, to: ArrowType):
        self.child = child^
        self.to = to.copy()

    def kind(self) -> UInt8:
        return CAST

    def dtype(self) -> Optional[ArrowType]:
        return self.to.copy()

    def inputs(self) -> List[AnyValue]:
        return [self.child]

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"cast(")
        self.child.write_to(writer)
        writer.write(t", {self.to})")


# ---------------------------------------------------------------------------
# Literal helpers
# ---------------------------------------------------------------------------


def _make_literal[T: PrimitiveType](value: Scalar[T.native]) raises -> AnyValue:
    """Create a Literal expression from a typed scalar value."""
    var builder = PrimitiveBuilder[T](1)
    builder.unsafe_append(value)
    return Literal(value=builder.finish().to_any())


# ---------------------------------------------------------------------------
# Free-standing factory functions
# ---------------------------------------------------------------------------


def col(index: Int) -> AnyValue:
    """Reference to the ``index``-th input column."""
    return Column(index=index, name=String(), dtype_=None)


def col(var name: String) -> AnyValue:
    """Reference to a named column (resolved against the schema at execution).
    """
    return Column(index=-1, name=name^, dtype_=None)


def lit[T: PrimitiveType](value: Scalar[T.native]) raises -> AnyValue:
    """A scalar constant broadcast to the length of the first input."""
    return _make_literal[T](value)


def if_else(cond: AnyValue, then_: AnyValue, else_: AnyValue) -> AnyValue:
    """Element-wise conditional: result[i] = then_[i] if cond[i] else else_[i].
    """
    return IfElse(cond=cond, then_=then_, else_=else_)


# ---------------------------------------------------------------------------
# Expression tree utilities
# ---------------------------------------------------------------------------


def rebuild(expr: AnyValue, children: List[AnyValue]) -> AnyValue:
    """Reconstruct an expression node with replaced children.

    Leaves (Column, Literal) are returned unchanged.
    """
    var k = expr.kind()
    if k == LOAD or k == LITERAL:
        return expr
    if k >= ADD and k <= OR:  # Binary ops
        var bin = expr.downcast[Binary]()
        return Binary(op=bin[].op, left=children[0], right=children[1])
    if k >= NEG and k <= NOT:  # Unary ops
        var un = expr.downcast[Unary]()
        return Unary(op=un[].op, child=children[0])
    if k == IS_NULL:
        return IsNull(child=children[0])
    if k == IF_ELSE:
        return IfElse(cond=children[0], then_=children[1], else_=children[2])
    if k == CAST:
        var c = expr.downcast[Cast]()
        return Cast(child=children[0], to=c[].to)
    return expr  # unknown: pass through


def resolve_columns(expr: AnyValue, schema: Schema) raises -> AnyValue:
    """Resolve name-based Column references (index == -1) to positional
    indices using the schema.

    Walks the expression tree bottom-up, replacing ``col("name")`` with
    ``col(index)`` where ``index`` is the field position in ``schema``.
    """
    var k = expr.kind()
    if k == LOAD:
        var c = expr.downcast[Column]()
        if c[].index == -1:
            var idx = schema.get_field_index(c[].name)
            if idx == -1:
                raise Error(
                    "resolve_columns: column '"
                    + c[].name
                    + "' not found in schema"
                )
            return col(idx)
        return expr

    if k == LITERAL:
        return expr

    # Recursively resolve children and rebuild
    var children = expr.inputs()
    for i in range(len(children)):
        children[i] = resolve_columns(children[i], schema)
    return rebuild(expr, children)
