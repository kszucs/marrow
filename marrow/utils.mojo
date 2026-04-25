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
from std.builtin.rebind import trait_downcast
from std.os import abort
from std.sys import has_accelerator, CompilationTarget
from std.sys.info import _accelerator_arch


# `_TypePredicateGenerator` was moved from a top-level alias in
# `std.builtin.variadics` to a member of `TypeList` in Mojo 1.0.0b1; redeclare
# the underlying MLIR generator type here so our variant-dispatch helpers keep
# accepting an arbitrary type predicate.
comptime _TypePredicateGenerator[T: type_of(AnyType)] = __mlir_type[
    `!lit.generator<<"Type": `,
    T,
    `>`,
    Bool,
    `>`,
]


def has_accelerator_support[*dtypes: DType]() -> Bool:
    """Check if there is accelerator support for all given dtypes.

    For example Metal doesn't support float64 as of April 2026.

    Also guards against Mojo toolchain regressions where the GPU architecture
    string is malformed (e.g. 'metal:2-metal4' on an M2 with Metal 4 API).
    The valid Metal targets are 'metal:1'–'metal:4'; anything else indicates
    the toolchain cannot compile GPU kernels for this device and we fall back
    to CPU.
    """
    if not has_accelerator():
        return False
    if not CompilationTarget.is_apple_silicon():
        return True
    # Validate the GPU architecture string before attempting to compile any
    # GPU kernel.  A malformed target (e.g. 'metal:2-metal4') causes a hard
    # constraint failure deep inside simd_width_of, so we gate it out here.
    comptime arch = _accelerator_arch()
    comptime if (
        arch != "metal:1"
        and arch != "metal:2"
        and arch != "metal:3"
        and arch != "metal:4"
    ):
        return False
    comptime for dtype in dtypes:
        if dtype == DType.float64:
            return False
    return True


comptime _always_true[T: Movable] = True


def variant_dispatch[
    R: AnyType,
    //,
    Trait: type_of(AnyType),
    *Ts: Movable,
    predicate: _TypePredicateGenerator[Movable] = _always_true,
    func: def[T: Trait](T) capturing[_] -> R,
](ref v: Variant[*Ts]) -> R:
    """Dispatch *func* to the active type in *v*, reinterpreted as *Trait*.

    Only types matching *predicate* are dispatched. Defaults to all types,
    so passing no predicate covers the full variant.
    """
    comptime for i in range(len(Ts)):
        comptime T = Ts[i]
        comptime if predicate[T]:
            if v.isa[T]():
                return func(trait_downcast[Trait](v[T]))
    abort("unreachable: variant_dispatch")


def variant_dispatch_raises[
    R: AnyType,
    //,
    Trait: type_of(AnyType),
    *Ts: Movable,
    predicate: _TypePredicateGenerator[Movable] = _always_true,
    func: def[T: Trait](T) raises capturing[_] -> R,
](v: Variant[*Ts]) raises -> R:
    """Like *variant_dispatch* but *func* may raise."""
    comptime for i in range(len(Ts)):
        comptime T = Ts[i]
        comptime if predicate[T]:
            if v.isa[T]():
                return func(trait_downcast[Trait](v[T]))
    abort("unreachable: variant_dispatch_raises")


# TODO: using `ref v` should support both `read` and `mut` args but the compiler crashes
def variant_dispatch_raises[
    R: AnyType,
    //,
    Trait: type_of(AnyType),
    *Ts: Movable,
    predicate: _TypePredicateGenerator[Movable] = _always_true,
    func: def[T: Trait](mut T) raises capturing[_] -> R,
](mut v: Variant[*Ts]) raises -> R:
    """Like *variant_dispatch_raises* but *func* takes a mutable reference."""
    comptime for i in range(len(Ts)):
        comptime T = Ts[i]
        comptime if predicate[T]:
            if v.isa[T]():
                return func(trait_downcast[Trait](v[T]))
    abort("unreachable: variant_dispatch_raises")
