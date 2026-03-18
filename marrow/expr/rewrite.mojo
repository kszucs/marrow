"""Rule-based expression rewriting.

``Rewrite``    — the trait every rewrite rule must implement.
``AnyRewrite`` — the type-erased, ArcPointer-backed rule container.
``Rewriter``   — drives bottom-up fixed-point iteration over a list of rules.

Rules are **non-destructive**: ``apply`` returns a new ``AnyValue`` or ``None``
(no match).  This is a hard requirement for e-graph / equality-saturation
compatibility — the same rule list can be handed to a future ``EGraph``
runner without modification.

Example
-------
    # Fold x + 0 → x
    struct FoldAddZero(Rewrite):
        fn name(self) -> StringLiteral:
            return "fold_add_zero"

        fn apply(self, expr: AnyValue) -> Optional[AnyValue]:
            if expr.kind() != ADD:
                return None
            var bin = expr.downcast[Binary]()[]
            if bin.right.kind() != LITERAL:
                return None
            # Literal.value is a length-1 Array; inspect element 0
            # to check whether it is zero.
            return None

    var rewriter = Rewriter(List[AnyRewrite](FoldAddZero()))
    var simplified = rewriter.rewrite(expr)
"""

from std.memory import ArcPointer
from marrow.expr.values import (
    AnyValue,
    rebuild,
    LOAD,
    LITERAL,
    ADD,
)


# ---------------------------------------------------------------------------
# Rewrite trait
# ---------------------------------------------------------------------------


trait Rewrite(ImplicitlyDestructible, Movable):
    """A single, non-destructive expression rewrite rule.

    ``apply`` must NOT mutate its argument.  Return ``None`` when the rule
    does not match; return a new ``AnyValue`` when it fires.
    """

    fn name(self) -> StringLiteral:
        """Short name for diagnostics and tracing."""
        ...

    fn apply(self, expr: AnyValue) -> Optional[AnyValue]:
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
    var _virt_apply: fn(ArcPointer[NoneType], AnyValue) -> Optional[AnyValue]
    var _virt_drop: fn(var ArcPointer[NoneType])

    # --- trampolines ---

    @staticmethod
    fn _tramp_name[T: Rewrite](ptr: ArcPointer[NoneType]) -> StringLiteral:
        return rebind[ArcPointer[T]](ptr)[].name()

    @staticmethod
    fn _tramp_apply[
        T: Rewrite
    ](ptr: ArcPointer[NoneType], expr: AnyValue) -> Optional[AnyValue]:
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

    fn apply(self, expr: AnyValue) -> Optional[AnyValue]:
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

    fn rewrite(self, expr: AnyValue) raises -> AnyValue:
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
            current = rebuild(expr, children)

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


