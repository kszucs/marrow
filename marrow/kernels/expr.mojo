"""Expression tree for compute kernels.

Defines an ``Expr`` tree that represents arithmetic and predicate expressions
over input arrays.

``DispatchHint`` (``DISPATCH_AUTO`` / ``DISPATCH_CPU`` / ``DISPATCH_GPU``) on
each ``Expr`` node signals the ``PipelineExecutor`` whether to route the
expression to the CPU or GPU path.  The default is ``DISPATCH_AUTO``.

Example::

    var a = Expr.input(0)
    var b = Expr.input(1)

    # Arithmetic: abs(a - b)
    var arith = Expr.abs_(Expr.sub(a, b))

    # Predicate: a < literal(10.0)
    var pred = Expr.less(a, Expr.literal[float64](10.0))

    # Conditional: if a < 10 then a else b
    var cond_expr = Expr.if_else(pred, a, b)
"""

from marrow.dtypes import DataType


# ---------------------------------------------------------------------------
# Expression node kind constants
# ---------------------------------------------------------------------------

comptime LOAD:    UInt8 = 0
comptime ADD:     UInt8 = 1
comptime SUB:     UInt8 = 2
comptime MUL:     UInt8 = 3
comptime DIV:     UInt8 = 4
comptime NEG:     UInt8 = 5
comptime ABS:     UInt8 = 6
comptime LITERAL: UInt8 = 7   # scalar constant → PrimitiveArray[T]
comptime EQ:      UInt8 = 8   # equal           → PrimitiveArray[bool_]
comptime NE:      UInt8 = 9   # not-equal       → PrimitiveArray[bool_]
comptime LT:      UInt8 = 10  # less-than       → PrimitiveArray[bool_]
comptime LE:      UInt8 = 11  # less-or-equal   → PrimitiveArray[bool_]
comptime GT:      UInt8 = 12  # greater-than    → PrimitiveArray[bool_]
comptime GE:      UInt8 = 13  # greater-or-equal → PrimitiveArray[bool_]
comptime AND:     UInt8 = 14  # logical AND (bool operands) → bool
comptime OR:      UInt8 = 15  # logical OR  (bool operands) → bool
comptime NOT:     UInt8 = 16  # logical NOT (bool operand)  → bool
comptime IS_NULL: UInt8 = 17  # null check                  → bool
comptime IF_ELSE: UInt8 = 18  # conditional (cond, then_, else_) → T


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
# Expr — arithmetic + predicate expression tree
# ---------------------------------------------------------------------------


struct Expr(Copyable, ImplicitlyCopyable, Movable, Writable):
    """A node in an arithmetic or predicate expression tree.

    Leaf nodes (``LOAD``) reference an input array by index.  ``LITERAL``
    nodes hold a scalar constant stored as ``Float64``.  Interior nodes carry
    an operation kind and one, two, or three children.

    Each node carries a ``dispatch`` hint (``DISPATCH_AUTO``, ``DISPATCH_CPU``,
    or ``DISPATCH_GPU``) that ``PipelineExecutor`` uses to decide whether to
    route the entire expression to the CPU or GPU.  Sub-expressions inherit
    the parent's routing decision; per-node GPU dispatch is a future extension.
    """

    var kind: UInt8
    var input_idx: Int
    var literal_value: Float64   # used only when kind == LITERAL
    var dispatch: UInt8          # DISPATCH_AUTO by default
    var children: List[Expr]

    # --- copy constructor (manual: List[Expr] prevents auto-synthesis) ------

    fn __init__(out self, *, copy: Self):
        self.kind = copy.kind
        self.input_idx = copy.input_idx
        self.literal_value = copy.literal_value
        self.dispatch = copy.dispatch
        self.children = List[Expr](capacity=len(copy.children))
        for ref child in copy.children:
            self.children.append(Expr(copy=child))

    # --- base constructor ---------------------------------------------------

    fn __init__(
        out self,
        *,
        kind: UInt8,
        input_idx: Int,
        literal_value: Float64,
        dispatch: UInt8,
        var children: List[Expr],
    ):
        self.kind = kind
        self.input_idx = input_idx
        self.literal_value = literal_value
        self.dispatch = dispatch
        self.children = children^

    # --- leaf ---------------------------------------------------------------

    @staticmethod
    fn input(idx: Int) -> Expr:
        """Reference to the ``idx``-th input array."""
        return Expr(
            kind=LOAD,
            input_idx=idx,
            literal_value=0.0,
            dispatch=DISPATCH_AUTO,
            children=List[Expr](),
        )

    @staticmethod
    fn literal[T: DataType](value: Scalar[T.native]) -> Expr:
        """A scalar constant broadcast to the length of the first input."""
        return Expr(
            kind=LITERAL,
            input_idx=0,
            literal_value=Float64(value),
            dispatch=DISPATCH_AUTO,
            children=List[Expr](),
        )

    # --- dispatch hint ------------------------------------------------------

    fn with_dispatch(self, hint: UInt8) -> Expr:
        """Return a copy of this expression with the given dispatch hint.

        The hint controls how ``PipelineExecutor`` routes the *entire* tree
        rooted at this node (``DISPATCH_AUTO`` / ``DISPATCH_CPU`` /
        ``DISPATCH_GPU``).
        """
        var copy = self
        copy.dispatch = hint
        return copy^

    # --- binary helpers -----------------------------------------------------

    @staticmethod
    fn _binary(kind: UInt8, var left: Expr, var right: Expr) -> Expr:
        var children = List[Expr]()
        children.append(left^)
        children.append(right^)
        return Expr(
            kind=kind,
            input_idx=0,
            literal_value=0.0,
            dispatch=DISPATCH_AUTO,
            children=children^,
        )

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

    # --- arithmetic unary ---------------------------------------------------

    @staticmethod
    fn _unary(kind: UInt8, var child: Expr) -> Expr:
        var children = List[Expr]()
        children.append(child^)
        return Expr(
            kind=kind,
            input_idx=0,
            literal_value=0.0,
            dispatch=DISPATCH_AUTO,
            children=children^,
        )

    @staticmethod
    fn neg(var child: Expr) -> Expr:
        """Element-wise negation."""
        return Expr._unary(NEG, child^)

    @staticmethod
    fn abs_(var child: Expr) -> Expr:
        """Element-wise absolute value."""
        return Expr._unary(ABS, child^)

    # --- comparison (return bool array) -------------------------------------

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

    # --- boolean (bool × bool → bool) ---------------------------------------

    @staticmethod
    fn and_(var left: Expr, var right: Expr) -> Expr:
        """Logical AND of two boolean expressions."""
        return Expr._binary(AND, left^, right^)

    @staticmethod
    fn or_(var left: Expr, var right: Expr) -> Expr:
        """Logical OR of two boolean expressions."""
        return Expr._binary(OR, left^, right^)

    @staticmethod
    fn not_(var child: Expr) -> Expr:
        """Logical NOT of a boolean expression."""
        return Expr._unary(NOT, child^)

    # --- null check ---------------------------------------------------------

    @staticmethod
    fn is_null(var child: Expr) -> Expr:
        """True where the child expression produces a null value."""
        return Expr._unary(IS_NULL, child^)

    # --- conditional --------------------------------------------------------

    @staticmethod
    fn if_else(var cond: Expr, var then_: Expr, var else_: Expr) -> Expr:
        """Element-wise conditional: result[i] = then_[i] if cond[i] else else_[i]."""
        var children = List[Expr]()
        children.append(cond^)
        children.append(then_^)
        children.append(else_^)
        return Expr(
            kind=IF_ELSE,
            input_idx=0,
            literal_value=0.0,
            dispatch=DISPATCH_AUTO,
            children=children^,
        )

    # --- display ------------------------------------------------------------

    fn write_to[W: Writer](self, mut writer: W):
        if self.kind == LOAD:
            writer.write(t"input({self.input_idx})")
        elif self.kind == LITERAL:
            writer.write(t"literal({self.literal_value})")
        elif self.kind == ADD:
            writer.write(t"add(")
            self.children[0].write_to(writer)
            writer.write(t", ")
            self.children[1].write_to(writer)
            writer.write(t")")
        elif self.kind == SUB:
            writer.write(t"sub(")
            self.children[0].write_to(writer)
            writer.write(t", ")
            self.children[1].write_to(writer)
            writer.write(t")")
        elif self.kind == MUL:
            writer.write(t"mul(")
            self.children[0].write_to(writer)
            writer.write(t", ")
            self.children[1].write_to(writer)
            writer.write(t")")
        elif self.kind == DIV:
            writer.write(t"div(")
            self.children[0].write_to(writer)
            writer.write(t", ")
            self.children[1].write_to(writer)
            writer.write(t")")
        elif self.kind == NEG:
            writer.write(t"neg(")
            self.children[0].write_to(writer)
            writer.write(t")")
        elif self.kind == ABS:
            writer.write(t"abs(")
            self.children[0].write_to(writer)
            writer.write(t")")
        elif self.kind == EQ:
            writer.write(t"equal(")
            self.children[0].write_to(writer)
            writer.write(t", ")
            self.children[1].write_to(writer)
            writer.write(t")")
        elif self.kind == NE:
            writer.write(t"not_equal(")
            self.children[0].write_to(writer)
            writer.write(t", ")
            self.children[1].write_to(writer)
            writer.write(t")")
        elif self.kind == LT:
            writer.write(t"less(")
            self.children[0].write_to(writer)
            writer.write(t", ")
            self.children[1].write_to(writer)
            writer.write(t")")
        elif self.kind == LE:
            writer.write(t"less_equal(")
            self.children[0].write_to(writer)
            writer.write(t", ")
            self.children[1].write_to(writer)
            writer.write(t")")
        elif self.kind == GT:
            writer.write(t"greater(")
            self.children[0].write_to(writer)
            writer.write(t", ")
            self.children[1].write_to(writer)
            writer.write(t")")
        elif self.kind == GE:
            writer.write(t"greater_equal(")
            self.children[0].write_to(writer)
            writer.write(t", ")
            self.children[1].write_to(writer)
            writer.write(t")")
        elif self.kind == AND:
            writer.write(t"and_(")
            self.children[0].write_to(writer)
            writer.write(t", ")
            self.children[1].write_to(writer)
            writer.write(t")")
        elif self.kind == OR:
            writer.write(t"or_(")
            self.children[0].write_to(writer)
            writer.write(t", ")
            self.children[1].write_to(writer)
            writer.write(t")")
        elif self.kind == NOT:
            writer.write(t"not_(")
            self.children[0].write_to(writer)
            writer.write(t")")
        elif self.kind == IS_NULL:
            writer.write(t"is_null(")
            self.children[0].write_to(writer)
            writer.write(t")")
        elif self.kind == IF_ELSE:
            writer.write(t"if_else(")
            self.children[0].write_to(writer)
            writer.write(t", ")
            self.children[1].write_to(writer)
            writer.write(t", ")
            self.children[2].write_to(writer)
            writer.write(t")")
        else:
            writer.write(t"<unknown>")
