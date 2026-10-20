# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`Enum`, the trait behind every type-erased handle.

A handle holds one of its family's members in a storage field — a `UniqueVariant`
when copies are independent, a `SharedVariant` when copies share one value —
and writes the implicit conversion from its family:

```mojo
struct DynType(Copyable, Enum):
    comptime Storage = UniqueVariant[DataType, NullType, BoolType, Int8Type, ...]
    var _storage: Self.Storage

    @implicit
    def __init__[T: DataType](out self, var value: T):
        self._storage = Self.Storage(value^)
```

`Enum` supplies the rest: `isa[T]()`, `get[T]()`, and four ways to run a
closure, a
`def[T: F](T) raises -> R`, on the member. `visit[R, F](func)` runs it when
every member is an `F`, aborting if it raises; `try_visit[R, F](func, message)`
runs it when the member is an `F` and raises a `TypeError` carrying `message`
otherwise. `visit_mut` and `try_visit_mut` pass the member `mut`.
"""

from std.builtin.rebind import downcast
from std.hash import Hasher
from std.memory import ArcPointer
from std.os import abort
from std.utils import Variant

from ..errors import TypeError


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


# The visits take the closure as a plain `Func` and prove its type in their
# body, with `R` explicit: an inherited default spelling a closure type in its
# signature crashes the compiler when called on a concrete handle. Every
# closure is declared `raises`: reinterpreting a non-raising one fails with
# "does not have witness table for trait '__mojo_closure__'", so the total
# visits catch once and abort.


trait EnumStorage:
    """Exactly what `Enum`'s defaults call on the storage field."""

    def isa[T: AnyType](self) -> Bool:
        ...

    def get[T: AnyType](ref self) -> ref[self] T:
        ...

    def visit[R: Movable, *, Func: AnyType](self, func: Func) -> R:
        ...

    def visit[
        R: Movable, F: type_of(AnyType), *, Func: AnyType
    ](self, func: Func) -> R:
        ...

    def visit_mut[R: Movable, *, Func: AnyType](mut self, func: Func) -> R:
        ...

    def visit_mut[
        R: Movable, F: type_of(AnyType), *, Func: AnyType
    ](mut self, func: Func) -> R:
        ...

    def try_visit[
        R: Movable, F: type_of(AnyType), Func: AnyType
    ](self, func: Func, message: StaticString) raises -> R:
        ...

    def try_visit_mut[
        R: Movable, F: type_of(AnyType), Func: AnyType
    ](mut self, func: Func, message: StaticString) raises -> R:
        ...


struct UniqueVariant[Trait: type_of(AnyType), *Ts: Movable & Deinitable](
    Copyable where conforms_to(Variant[*Ts], Copyable),
    EnumStorage,
    Equatable where conforms_to(Variant[*Ts], Equatable),
    Hashable where conforms_to(Variant[*Ts], Hashable),
    Movable,
    Writable where conforms_to(Variant[*Ts], Writable),
):
    """One of `Ts`, members of the family `Trait` — checked where a visit
    over the whole family is instantiated — by value: copies are
    independent. `SharedVariant` is the pointer-like counterpart. Copyable,
    equatable, hashable and writable when every member is.

    ```mojo
    comptime Storage = UniqueVariant[Array, BoolArray, Int64Array, StringArray]
    ```
    """

    comptime Visitor[R: Movable] = def[T: Self.Trait](T) raises -> R
    """What a visit over the whole family runs on the member."""

    comptime MutVisitor[R: Movable] = def[T: Self.Trait](mut T) raises -> R
    """What a mutating visit over the whole family runs on the member."""

    var _value: Variant[*Self.Ts]

    @implicit
    @always_inline
    def __init__[T: Movable](out self, var value: T):
        self._value = Variant[*Self.Ts](value^)

    def __eq__(
        self, other: Self
    ) -> Bool where conforms_to(Variant[*Self.Ts], Equatable):
        comptime V = downcast[Variant[*Self.Ts], Equatable]
        return rebind[V](self._value) == rebind[V](other._value)

    def __hash__[
        H: Hasher
    ](self, mut hasher: H) where conforms_to(Variant[*Self.Ts], Hashable):
        rebind[downcast[Variant[*Self.Ts], Hashable]](self._value).__hash__(
            hasher
        )

    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Variant[*Self.Ts], Writable):
        rebind[downcast[Variant[*Self.Ts], Writable]](self._value).write_to(
            writer
        )

    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Variant[*Self.Ts], Writable):
        rebind[downcast[Variant[*Self.Ts], Writable]](
            self._value
        ).write_repr_to(writer)

    def isa[T: AnyType](self) -> Bool:
        return self._value.isa[T]()

    def get[T: AnyType](ref self) -> ref[self] T:
        """The member as the concrete `T` — a borrow. A wrong `T` aborts, in
        every build mode."""
        if not self._value.isa[T]():
            abort("UniqueVariant.get: wrong member type")
        return widen[origin_of(self)](self._value.unsafe_get[T]())

    def visit[R: Movable, *, Func: AnyType](self, func: Func) -> R:
        """Run `func`, a `Visitor[R]`, on the member; a raise aborts."""
        ref f = rebind[downcast[Func, Self.Visitor[R]]](func)
        try:
            comptime for i in range(len(Self.Ts)):
                comptime T = Self.Ts[i]
                if self._value.isa[T]():
                    return f(
                        rebind[downcast[T, Self.Trait]](
                            self._value.unsafe_get[T]()
                        )
                    )
        except e:
            abort(String("UniqueVariant.visit: ", e))
        abort("UniqueVariant.visit: no member matched")

    def visit[
        R: Movable, F: type_of(AnyType), *, Func: AnyType
    ](self, func: Func) -> R:
        """`visit` over a sub-family `F` every member belongs to: `func` is a
        `def[T: F](T) raises -> R`."""
        comptime assert Self.Ts.all_conforms_to[
            F
        ](), "visit: not every member is an F"
        ref f = rebind[downcast[Func, def[U: F](U) raises -> R]](func)
        try:
            comptime for i in range(len(Self.Ts)):
                comptime T = Self.Ts[i]
                if self._value.isa[T]():
                    return f(
                        rebind[downcast[T, F]](self._value.unsafe_get[T]())
                    )
        except e:
            abort(String("UniqueVariant.visit: ", e))
        abort("UniqueVariant.visit: no member matched")

    def visit_mut[R: Movable, *, Func: AnyType](mut self, func: Func) -> R:
        """`visit` with the member passed `mut`: `func` is a
        `MutVisitor[R]`."""
        ref f = rebind[downcast[Func, Self.MutVisitor[R]]](func)
        try:
            comptime for i in range(len(Self.Ts)):
                comptime T = Self.Ts[i]
                if self._value.isa[T]():
                    return f(
                        rebind[downcast[T, Self.Trait]](
                            self._value.unsafe_get[T]()
                        )
                    )
        except e:
            abort(String("UniqueVariant.visit_mut: ", e))
        abort("UniqueVariant.visit_mut: no member matched")

    def visit_mut[
        R: Movable, F: type_of(AnyType), *, Func: AnyType
    ](mut self, func: Func) -> R:
        """`visit_mut` over a family `F` every member belongs to."""
        comptime assert Self.Ts.all_conforms_to[
            F
        ](), "visit_mut: not every member is an F"
        ref f = rebind[downcast[Func, def[U: F](mut U) raises -> R]](func)
        try:
            comptime for i in range(len(Self.Ts)):
                comptime T = Self.Ts[i]
                if self._value.isa[T]():
                    return f(
                        rebind[downcast[T, F]](self._value.unsafe_get[T]())
                    )
        except e:
            abort(String("UniqueVariant.visit_mut: ", e))
        abort("UniqueVariant.visit_mut: no member matched")

    def try_visit[
        R: Movable, F: type_of(AnyType), Func: AnyType
    ](self, func: Func, message: StaticString) raises -> R:
        """Run `func`, a `def[T: F](T) raises -> R`, on the member if it is an
        `F`, else raise a `TypeError` carrying `message`."""
        ref f = rebind[downcast[Func, def[U: F](U) raises -> R]](func)
        comptime for i in range(len(Self.Ts)):
            comptime T = Self.Ts[i]
            comptime if conforms_to(T, F):
                if self._value.isa[T]():
                    return f(
                        rebind[downcast[T, F]](self._value.unsafe_get[T]())
                    )
        raise TypeError(message)

    @always_inline
    def try_visit_mut[
        R: Movable, F: type_of(AnyType), Func: AnyType
    ](mut self, func: Func, message: StaticString) raises -> R:
        """`try_visit` with the member passed `mut`: `func` is a
        `def[T: F](mut T) raises -> R`."""
        ref f = rebind[downcast[Func, def[U: F](mut U) raises -> R]](func)
        comptime for i in range(len(Self.Ts)):
            comptime T = Self.Ts[i]
            comptime if conforms_to(T, F):
                if self._value.isa[T]():
                    return f(
                        rebind[downcast[T, F]](self._value.unsafe_get[T]())
                    )
        raise TypeError(message)


struct SharedVariant[Trait: type_of(AnyType), *Ts: Movable & Deinitable](
    EnumStorage, ImplicitlyCopyable
):
    """One of `Ts`, shared: a copy points at the same value, so a change made
    through one handle is seen through every other. `UniqueVariant` is the
    value-like counterpart, whose copies are independent."""

    var _ptr: ArcPointer[UniqueVariant[Self.Trait, *Self.Ts]]

    @implicit
    @always_inline
    def __init__[T: Movable](out self, var value: T):
        self._ptr = ArcPointer(UniqueVariant[Self.Trait, *Self.Ts](value^))

    def isa[T: AnyType](self) -> Bool:
        return self._ptr[].isa[T]()

    def get[T: AnyType](ref self) -> ref[self] T:
        # The member carries `self`'s origin rather than the pointee's: the
        # handle keeps the pointer alive.
        return widen[origin_of(self)](self._ptr[].get[T]())

    def visit[R: Movable, *, Func: AnyType](self, func: Func) -> R:
        return self._ptr[].visit[R](func)

    def visit[
        R: Movable, F: type_of(AnyType), *, Func: AnyType
    ](self, func: Func) -> R:
        return self._ptr[].visit[R, F](func)

    def visit_mut[R: Movable, *, Func: AnyType](mut self, func: Func) -> R:
        return self._ptr[].visit_mut[R](func)

    def visit_mut[
        R: Movable, F: type_of(AnyType), *, Func: AnyType
    ](mut self, func: Func) -> R:
        return self._ptr[].visit_mut[R, F](func)

    def try_visit[
        R: Movable, F: type_of(AnyType), Func: AnyType
    ](self, func: Func, message: StaticString) raises -> R:
        return self._ptr[].try_visit[R, F](func, message)

    def try_visit_mut[
        R: Movable, F: type_of(AnyType), Func: AnyType
    ](mut self, func: Func, message: StaticString) raises -> R:
        return self._ptr[].try_visit_mut[R, F](func, message)


# ---------------------------------------------------------------------------
# The handle: Enum
# ---------------------------------------------------------------------------


@always_inline
def widen[T: AnyType, //, o: Origin](ref value: T) -> ref[o] T:
    """`value` borrowed with the origin `o` — a field with its owner's origin,
    which a field access cannot be widened to."""
    return rebind[Pointer[T, o]](Pointer(to=value))[]


trait Enum(Movable):
    """A type-erased handle: one of its family's members, in a storage field.

    A handle names its storage type as `Storage` and hands the field out with
    `storage()`; `Enum`'s defaults reach it through that, since a trait
    cannot name a conformer's field, and call it through `EnumStorage`. A handle declares any further conformance it wants, and
    gets each one its `UniqueVariant` has: `UniqueVariant` conforms to `Copyable`,
    `Equatable`, `Hashable` and `Writable` when every member does.
    """

    # No implicit constructor here: it could only be bound on `Movable`, and
    # a generic `__init__[T: Movable]` wins over a handle's own one-argument
    # constructors (`DynBuilder(dtype)`). A trait method takes no `where`
    # clause, and a default bound on an associated trait matches no conformer.

    # A handle also writes its copy constructor (`copy._storage` copied into
    # `_storage`): the synthesized one measured ~70 KB larger per size gate.

    comptime Storage: EnumStorage
    """The storage field's type: a `UniqueVariant` or a `SharedVariant`."""

    def storage(ref self) -> ref[self] Self.Storage:
        """The storage field, borrowed with `self`'s origin:
        `return widen[origin_of(self)](self._storage)`."""
        ...

    def isa[T: AnyType](self) -> Bool:
        return self.storage().isa[T]()

    def get[T: AnyType](ref self) -> ref[self] T:
        """The member as the concrete `T` — a borrow. A wrong `T` aborts."""
        return self.storage().get[T]()

    def visit[R: Movable, *, Func: AnyType](self, func: Func) -> R:
        """Run `func`, a `def[T: Trait](T) raises -> R` over the storage's
        family, on the member; a raise aborts."""
        return self.storage().visit[R](func)

    def visit[
        R: Movable, F: type_of(AnyType), *, Func: AnyType
    ](self, func: Func) -> R:
        """`visit` over a family `F` every member belongs to."""
        return self.storage().visit[R, F](func)

    def visit_mut[R: Movable, *, Func: AnyType](mut self, func: Func) -> R:
        """`visit` with the member passed `mut`."""
        return self.storage().visit_mut[R](func)

    def visit_mut[
        R: Movable, F: type_of(AnyType), *, Func: AnyType
    ](mut self, func: Func) -> R:
        """`visit` with the member passed `mut`: `func` is a
        `def[T: F](mut T) raises -> R`."""
        return self.storage().visit_mut[R, F](func)

    def try_visit[
        R: Movable, F: type_of(AnyType), *, Func: AnyType
    ](self, func: Func, message: StaticString) raises -> R:
        """Run `func`, a `def[T: F](T) raises -> R`, on the member if it is an
        `F`, else raise a `TypeError` carrying `message`."""
        return self.storage().try_visit[R, F](func, message)

    def try_visit_mut[
        R: Movable, F: type_of(AnyType), *, Func: AnyType
    ](mut self, func: Func, message: StaticString) raises -> R:
        """`try_visit` with the member passed `mut`: `func` is a
        `def[T: F](mut T) raises -> R`."""
        return self.storage().try_visit_mut[R, F](func, message)
