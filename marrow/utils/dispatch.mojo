"""Generic Variant dispatch utilities.

`variant_dispatch(v, func)` drives runtime dispatch over a `Variant[*Ts]`
without dynamic dispatch or vtables: the active type is detected via
`v.isa[T]()` in a compile-time loop and the value is forwarded to `func`.

**`func` is bound on `Movable`, not on a trait of the caller's choosing**, and
that is deliberate rather than a limitation of the loop. A closure type cannot
be generic over its own trait bound, so a `Trait` parameter here would be
declarable and never satisfiable (see CLAUDE.md). Binding on `Movable` — which
`Variant` already requires of every member — removes the parameter entirely and
leaves *narrowing* to the caller, which is where the trait is concrete anyway:

```mojo
def narrow[T: Movable](t: T) raises {imm} -> R:
    comptime if conforms_to(T, Array):
        return func(rebind[downcast[T, Array]](t))
    else:
        raise Error("...")
return variant_dispatch(self._v, narrow)
```

`DynType`, `DynArray`, `DynScalar` and `DynBuilder` each wrap it in exactly one
such adapter (`_dispatch`), and `DynType.dispatch_*` narrows a second step to a
dtype family. `variant_dispatch` takes a non-raising `func`; `variant_dispatch_raises`
takes a raising one, by value or by mutable reference.
"""

from std.utils import Variant
from std.os import abort


def variant_dispatch[
    R: Movable, //, *Ts: Movable, Func: def[T: Movable](T) -> R
](ref v: Variant[*Ts], func: Func) -> R:
    """Run `func` on the active member of `v`. See the module docstring."""
    comptime for i in range(len(Ts)):
        comptime T = Ts[i]
        if v.isa[T]():
            return func(v[T])
    abort("unreachable: variant_dispatch")


def variant_dispatch_raises[
    R: Movable, //, *Ts: Movable, Func: def[T: Movable](T) raises -> R
](ref v: Variant[*Ts], func: Func) raises -> R:
    """Like `variant_dispatch` but `func` may raise.

    Named apart rather than overloaded: a non-raising closure also satisfies
    `raises`, so a single overload set is ambiguous at every call site.
    """
    comptime for i in range(len(Ts)):
        comptime T = Ts[i]
        if v.isa[T]():
            return func(v[T])
    raise Error("variant_dispatch: no arm matched the active variant type")


# TODO: using `ref v` should support both `read` and `mut` args but the compiler crashes
def variant_dispatch_raises[
    R: Movable, //, *Ts: Movable, Func: def[T: Movable](mut T) raises -> R
](mut v: Variant[*Ts], func: Func) raises -> R:
    """Like `variant_dispatch` but `func` takes a mutable reference."""
    comptime for i in range(len(Ts)):
        comptime T = Ts[i]
        if v.isa[T]():
            return func(v[T])
    raise Error("variant_dispatch: no arm matched the active variant type")
