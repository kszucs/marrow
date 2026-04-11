"""Generic Variant dispatch utilities.

These helpers drive runtime dispatch over a `Variant[*Ts]` without dynamic
dispatch or vtables.  The active type is detected via `v.isa[T]()` in a
compile-time loop; the value is then reinterpreted as *Trait* through
`trait_downcast` and forwarded to *func*.

Three overloads are provided — distinguished by whether *func* raises and
whether it takes its argument by value or by mutable reference:

  variant_dispatch            — *func* is non-raising, argument by value
  variant_dispatch_raises     — *func* raises,         argument by value
  variant_dispatch_raises     — *func* raises,         argument by mut-ref

Note: a single `ref[_] v` overload would unify all three, but the Mojo
compiler currently crashes when `ref[_]` is used here (tracked as a TODO).
"""

from std.utils import Variant
from std.builtin.variadics import Variadic, TypeList, _TypePredicateGenerator
from std.builtin.rebind import trait_downcast
from std.os import abort
from std.sys import has_accelerator, CompilationTarget


def has_accelerator_support[dtype: DType]() -> Bool:
    """Check if there is accelerator support for the given dtype.

    For example Metal doesn't support float64 as of April 2026.
    """
    if not has_accelerator():
        return False
    if not CompilationTarget.is_apple_silicon():
        return True
    return dtype != DType.float64


comptime _always_true[T: Movable & ImplicitlyDestructible] = True


def variant_dispatch[
    R: AnyType, //,
    Trait: type_of(AnyType),
    *Ts: Movable & ImplicitlyDestructible,
    predicate: _TypePredicateGenerator[Movable & ImplicitlyDestructible] = _always_true,
    func: def[T: Trait](T) capturing[_] -> R,
](ref v: Variant[*Ts]) -> R:
    """Dispatch *func* to the active type in *v*, reinterpreted as *Trait*.

    Only types matching *predicate* are dispatched. Defaults to all types,
    so passing no predicate covers the full variant.
    """
    comptime FilteredTs = Variadic.filter_types[*Ts, predicate=predicate]
    comptime for i in range(TypeList[*FilteredTs].size):
        comptime T = FilteredTs[i]
        if v.isa[T]():
            return func(trait_downcast[Trait](v[T]))
    abort("unreachable: variant_dispatch")


def variant_dispatch_raises[
    R: AnyType, //,
    Trait: type_of(AnyType),
    *Ts: Movable & ImplicitlyDestructible,
    predicate: _TypePredicateGenerator[Movable & ImplicitlyDestructible] = _always_true,
    func: def[T: Trait](T) raises capturing[_] -> R,
](v: Variant[*Ts]) raises -> R:
    """Like *variant_dispatch* but *func* may raise."""
    comptime FilteredTs = Variadic.filter_types[*Ts, predicate=predicate]
    comptime for i in range(TypeList[*FilteredTs].size):
        comptime T = FilteredTs[i]
        if v.isa[T]():
            return func(trait_downcast[Trait](v[T]))
    abort("unreachable: variant_dispatch_raises")


# TODO: using `ref v` should support both `read` and `mut` args but the compiler crashes
def variant_dispatch_raises[
    R: AnyType, //,
    Trait: type_of(AnyType),
    *Ts: Movable & ImplicitlyDestructible,
    predicate: _TypePredicateGenerator[Movable & ImplicitlyDestructible] = _always_true,
    func: def[T: Trait](mut T) raises capturing[_] -> R,
](mut v: Variant[*Ts]) raises -> R:
    """Like *variant_dispatch_raises* but *func* takes a mutable reference."""
    comptime FilteredTs = Variadic.filter_types[*Ts, predicate=predicate]
    comptime for i in range(TypeList[*FilteredTs].size):
        comptime T = FilteredTs[i]
        if v.isa[T]():
            return func(trait_downcast[Trait](v[T]))
    abort("unreachable: variant_dispatch_raises")
