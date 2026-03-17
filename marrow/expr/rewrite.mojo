"""Rule-based expression rewriting.

``Rewrite``    — the trait every rewrite rule must implement.
``AnyRewrite`` — the type-erased, ArcPointer-backed rule container.
``Rewriter``   — drives bottom-up fixed-point iteration over a list of rules.

Rules are **non-destructive**: ``apply`` returns a new ``Expr`` or ``None``
(no match).  This is a hard requirement for e-graph / equality-saturation
compatibility — the same rule list can be handed to a future ``EGraph``
runner without modification.

Example
-------
    # Fold x + 0 → x
    struct FoldAddZero(Rewrite):
        fn name(self) -> StringLiteral:
            return "fold_add_zero"

        fn apply(self, expr: Expr) -> Optional[Expr]:
            if expr.kind() != ADD:
                return None
            var bin = expr.as_binary()[]
            if bin.right.kind() != LITERAL:
                return None
            if bin.right.as_literal()[].value == 0.0:
                return bin.left
            return None

    var rewriter = Rewriter(List[AnyRewrite](FoldAddZero()))
    var simplified = rewriter.rewrite(expr)
"""

from std.memory import ArcPointer
from marrow.expr.expr import (
    Expr,
    Binary,
    Unary,
    IsNull,
    IfElse,
    Cast,
    LOAD,
    LITERAL,
    IS_NULL,
    IF_ELSE,
    CAST,
)


# ---------------------------------------------------------------------------
# Rewrite trait
# ---------------------------------------------------------------------------


trait Rewrite(ImplicitlyDestructible, Movable):
    """A single, non-destructive expression rewrite rule.

    ``apply`` must NOT mutate its argument.  Return ``None`` when the rule
    does not match; return a new ``Expr`` when it fires.
    """

    fn name(self) -> StringLiteral:
        """Short name for diagnostics and tracing."""
        ...

    fn apply(self, expr: Expr) -> Optional[Expr]:
        """Try to rewrite ``expr``.

        Returns:
            A new expression if the rule fired, else ``None``.
        """
        ...


# ---------------------------------------------------------------------------
# AnyRewrite — type-erased rule container
# ---------------------------------------------------------------------------


struct AnyRewrite(ImplicitlyCopyable, Movable):
    """Type-erased rewrite rule container."""

    var _data: ArcPointer[NoneType]
    var _virt_name: fn(ArcPointer[NoneType]) -> StringLiteral
    var _virt_apply: fn(ArcPointer[NoneType], Expr) -> Optional[Expr]
    var _virt_drop: fn(var ArcPointer[NoneType])

    # --- trampolines ---

    @staticmethod
    fn _tramp_name[T: Rewrite](ptr: ArcPointer[NoneType]) -> StringLiteral:
        return rebind[ArcPointer[T]](ptr)[].name()

    @staticmethod
    fn _tramp_apply[
        T: Rewrite
    ](ptr: ArcPointer[NoneType], expr: Expr) -> Optional[Expr]:
        return rebind[ArcPointer[T]](ptr)[].apply(expr)

    @staticmethod
    fn _tramp_drop[T: Rewrite](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    # --- construction ---

    @implicit
    fn __init__[T: Rewrite](out self, var value: T):
        var ptr = ArcPointer(value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_name = Self._tramp_name[T]
        self._virt_apply = Self._tramp_apply[T]
        self._virt_drop = Self._tramp_drop[T]

    fn __copyinit__(out self, read copy: Self):
        self._data = copy._data.copy()
        self._virt_name = copy._virt_name
        self._virt_apply = copy._virt_apply
        self._virt_drop = copy._virt_drop

    # --- public API ---

    fn name(self) -> StringLiteral:
        return self._virt_name(self._data)

    fn apply(self, expr: Expr) -> Optional[Expr]:
        return self._virt_apply(self._data, expr)

    fn __del__(deinit self):
        self._virt_drop(self._data^)


# ---------------------------------------------------------------------------
# Rewriter — bottom-up fixed-point driver
# ---------------------------------------------------------------------------


struct Rewriter:
    """Bottom-up fixed-point rule-based expression rewriter.

    Walks the expression tree bottom-up (children before parent), applying
    all rules to each node repeatedly until no rule fires (fixed point).
    This is the classical approach used by DataFusion and Polars.

    Future: the same ``rules`` list can be fed to an ``EGraph`` runner for
    equality saturation without any change to rule definitions or call sites.
    """

    var rules: List[AnyRewrite]

    fn __init__(out self, var rules: List[AnyRewrite]):
        self.rules = rules^

    fn rewrite(self, expr: Expr) raises -> Expr:
        """Rewrite ``expr`` bottom-up to a fixed point."""
        # Rewrite children first (bottom-up)
        var children = expr.inputs()
        var children_changed = False
        for i in range(len(children)):
            var rewritten = self.rewrite(children[i])
            if rewritten.kind() != children[i].kind():
                children_changed = True
            children[i] = rewritten

        # Rebuild the node with rewritten children if any changed
        var current = expr
        if children_changed:
            current = _rebuild(expr, children)

        # Apply rules to this node until fixed point
        var changed = True
        while changed:
            changed = False
            for ref rule in self.rules:
                var result = rule.apply(current)
                if result:
                    current = result.value()
                    changed = True
                    break  # restart rule sweep after any match
        return current


fn _rebuild(expr: Expr, children: List[Expr]) -> Expr:
    """Reconstruct a node with replaced children.

    Leaves (Column, Literal) are returned unchanged.
    """
    var k = expr.kind()
    if k == LOAD or k == LITERAL:
        return expr
    if k >= 2 and k <= 13:  # Binary ops (ADD..OR)
        var bin = expr.as_binary()[]
        return Binary(op=bin.op, left=children[0], right=children[1])
    if k >= 14 and k <= 16:  # Unary ops (NEG, ABS, NOT)
        var un = expr.as_unary()[]
        return Unary(op=un.op, child=children[0])
    if k == IS_NULL:
        return IsNull(child=children[0])
    if k == IF_ELSE:
        return IfElse(cond=children[0], then_=children[1], else_=children[2])
    if k == CAST:
        var cast = expr.as_cast()[]
        return Cast(child=children[0], to=cast.to)
    return expr  # unknown: pass through
