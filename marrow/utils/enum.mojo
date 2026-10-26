# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`Enum`, the trait behind every type-erased handle, and `EnumVariant`, the
one storage behind it.

A handle holds one of its family's members in an `EnumVariant` and writes the
implicit conversion from its family:

```mojo
struct DynType(Copyable, Enum):
    comptime Storage = ArcMembers[DataType, NullType, BoolType, ...]
    var _storage: Self.Storage

    def storage(ref self) -> ref[self] Self.Storage:
        return widen[origin_of(self)](self._storage)

    @implicit
    def __init__[T: DataType](out self, var value: T):
        self._storage = Self.Storage(value^)
```

`EnumVariant[Family, P, Ts, *Slots]` is a `Variant` with one slot per member
of `Ts`. The policy `P` says what a slot is and how a member goes in, comes out
and is copied; `_Members[P, Family, *Ts]` spells the slots as `P.Box` of each
member:

- `Members` (`Inline`): the member itself. Copies are independent; the
  variant is as large as its largest member.
- `ArcMembers` (`Arc`): an `ArcPointer` to the member. A copy is one atomic
  increment and shares the member, so a change made through one copy is seen
  through every other.
- `OwnedMembers` (`Owned`): an `OwnedPointer` to the member. A copy allocates
  a copy.

The slots are spelled by the alias rather than inside `EnumVariant` so the
pointer is visible where the variant is declared: under `ArcMembers` and
`OwnedMembers` a member may hold its own handle by value — a `ListArray`'s
child, a `DictionaryType`'s value type, a `ListBuilder`'s child. Under
`Members` it may not: the member would contain itself.

Three reinterpretations remain, each forced by the compiler (`backlog.md`
§1.3):

- `widen` gives a borrow of a field its owner's origin, which a field access
  does not carry and a method returning `ref[self]` needs.
- A visit takes its closure as a plain `Func` and reinterprets it as a
  `def[U: F](U) raises -> R`: a capturing closure does not convert to a
  parameter of that type, which is also why `R` is explicit.
- A visit hands the closure the member reinterpreted as `downcast[T, F]`:
  `comptime if conforms_to(T, F)` narrows for a concrete trait, not for a
  trait parameter `F`.

Every member is boxed, zero-sized ones included: a slot type chosen by the
member's size (`size_of`) never finishes building over marrow's recursive
dtypes, since answering it needs the layout being defined.

Visit closures are declared `raises` and return a value: a non-raising or
`None`-returning closure does not build when reinterpreted.
"""

from std.builtin.rebind import downcast
from std.hash import Hasher
from std.memory import ArcPointer, OwnedPointer
from std.os import abort
from std.utils import Variant

from ..errors import TypeError


@always_inline
def widen[T: AnyType, //, o: Origin](ref value: T) -> ref[o] T:
    """`value` borrowed with origin `o`. Sound only when `value` lives exactly
    as long as what `o` names."""
    return rebind[Pointer[T, o]](Pointer(to=value))[]


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


trait Policy:
    """How an `EnumVariant` holds a member."""

    comptime Box[From: Movable & Deinitable]: Movable & Deinitable
    """What a slot is for a member of type `From`."""

    @staticmethod
    def box[T: Movable & Deinitable](var value: T) -> Self.Box[T]:
        ...

    @staticmethod
    def unbox[T: Movable & Deinitable](ref box: Self.Box[T]) -> ref[box] T:
        ...

    @staticmethod
    def copy[T: Movable & Deinitable](box: Self.Box[T]) -> Self.Box[T]:
        ...


struct Inline(Policy):
    comptime Box[From: Movable & Deinitable] = From

    @staticmethod
    def box[T: Movable & Deinitable](var value: T) -> T:
        return value^

    @staticmethod
    def unbox[T: Movable & Deinitable](ref box: T) -> ref[box] T:
        return box

    @staticmethod
    def copy[T: Movable & Deinitable](box: T) -> T:
        comptime assert conforms_to(T, Copyable), "a member is not copyable"
        comptime if conforms_to(T, Copyable):
            return box.copy()
        else:
            abort("unreachable")


struct Arc(Policy):
    comptime Box[From: Movable & Deinitable] = ArcPointer[From]

    @staticmethod
    def box[T: Movable & Deinitable](var value: T) -> ArcPointer[T]:
        return ArcPointer(value^)

    @staticmethod
    def unbox[T: Movable & Deinitable](ref box: ArcPointer[T]) -> ref[box] T:
        return widen[origin_of(box)](box[])

    @staticmethod
    def copy[T: Movable & Deinitable](box: ArcPointer[T]) -> ArcPointer[T]:
        return box.copy()


struct Owned(Policy):
    comptime Box[From: Movable & Deinitable] = OwnedPointer[From]

    # Not `@always_inline`: inlined over a recursive member (`StructArray`
    # holding `DynArray`s) it sends `mojo build -O3` into an endless loop.
    @staticmethod
    def box[T: Movable & Deinitable](var value: T) -> OwnedPointer[T]:
        return OwnedPointer(value^)

    @staticmethod
    def unbox[T: Movable & Deinitable](ref box: OwnedPointer[T]) -> ref[box] T:
        return widen[origin_of(box)](box[])

    @staticmethod
    def copy[T: Movable & Deinitable](box: OwnedPointer[T]) -> OwnedPointer[T]:
        return Self.box(Inline.copy[T](box[]))


# ---------------------------------------------------------------------------
# The storage
# ---------------------------------------------------------------------------


trait EnumStorage(Copyable, Equatable, Hashable, Writable):
    """What `Enum` calls on a handle's storage; `EnumVariant` is the one
    conformer."""

    comptime Trait: type_of(AnyType)
    """The family every member belongs to: what a visit runs over unless it
    names a sub-family."""

    def isa[T: Movable & Deinitable](self) -> Bool:
        ...

    def get[T: Movable & Deinitable](ref self) -> ref[self] T:
        ...

    def visit[
        R: Movable, F: type_of(AnyType), *, Func: AnyType
    ](self, func: Func) -> R:
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


struct EnumVariant[
    Family: type_of(AnyType),
    P: Policy,
    Ts: TypeList[Trait=Movable & Deinitable, ...],
    *Slots: Movable & Deinitable,
](EnumStorage):
    """One of the members `Ts`, of the family `Family`, held as `P` says:
    member `Ts[i]` in slot `Slots[i]`, which is `P.Box[Ts[i]]`. The slots
    are spelled by the aliases rather than here, so a member may hold its own
    handle behind `P`'s pointer. Equality, hashing and writing need every
    member to have them, checked where they are used."""

    comptime Trait = Self.Family

    var _value: Variant[*Self.Slots]

    # --- lifecycle ---

    # Not `@always_inline`: inlined over a recursive member (`StructArray`
    # holding `DynArray`s), `mojo build -O3` never finishes.
    @implicit
    def __init__[T: Movable & Deinitable](out self, var value: T):
        comptime assert Self.Ts.contains[T](), "not a member type"
        self._value = type_of(self._value)(Self.P.box(value^))

    def __init__(out self, *, copy: Self):
        comptime for i in range(Self.Ts.length):
            comptime T = Self.Ts[i]
            if copy.isa[T]():
                self._value = type_of(self._value)(
                    Self.P.copy[T](copy._value[Self.P.Box[T]])
                )
                return
        abort("EnumVariant: no member matched")

    # --- access ---

    @always_inline
    def isa[T: Movable & Deinitable](self) -> Bool:
        comptime if Self.Ts.contains[T]():
            return self._value.isa[Self.P.Box[T]]()
        else:
            return False

    @always_inline
    def get[T: Movable & Deinitable](ref self) -> ref[self] T:
        """The member as `T`, borrowed; a wrong `T` aborts in every build."""
        comptime assert Self.Ts.contains[T](), "get: not a member type"
        return widen[origin_of(self)](
            Self.P.unbox[T](self._value[Self.P.Box[T]])
        )

    # --- visits ---

    def visit[
        R: Movable, F: type_of(AnyType), *, Func: AnyType
    ](self, func: Func) -> R:
        """Run `func`, a `def[T: F](T) raises -> R`, on the member; a raise
        aborts. Every member must be an `F`."""
        ref f = rebind[downcast[Func, def[U: F](U) raises -> R]](func)
        try:
            comptime for i in range(Self.Ts.length):
                comptime T = Self.Ts[i]
                comptime assert conforms_to(T, F), "visit: a member is not an F"
                if self.isa[T]():
                    return f(rebind[downcast[T, F]](self.get[T]()))
        except e:
            abort(String("visit: ", e))
        abort("visit: no member matched")

    def visit_mut[
        R: Movable, F: type_of(AnyType), *, Func: AnyType
    ](mut self, func: Func) -> R:
        """`visit` with the member passed `mut`."""
        ref f = rebind[downcast[Func, def[U: F](mut U) raises -> R]](func)
        try:
            comptime for i in range(Self.Ts.length):
                comptime T = Self.Ts[i]
                comptime assert conforms_to(T, F), "visit: a member is not an F"
                if self.isa[T]():
                    return f(rebind[downcast[T, F]](self.get[T]()))
        except e:
            abort(String("visit_mut: ", e))
        abort("visit_mut: no member matched")

    def try_visit[
        R: Movable, F: type_of(AnyType), Func: AnyType
    ](self, func: Func, message: StaticString) raises -> R:
        """Run `func`, a `def[T: F](T) raises -> R`, on the member if it is an
        `F`, else raise a `TypeError` carrying `message`."""
        ref f = rebind[downcast[Func, def[U: F](U) raises -> R]](func)
        comptime for i in range(Self.Ts.length):
            comptime T = Self.Ts[i]
            comptime if conforms_to(T, F):
                if self.isa[T]():
                    return f(rebind[downcast[T, F]](self.get[T]()))
        raise TypeError(message)

    def try_visit_mut[
        R: Movable, F: type_of(AnyType), Func: AnyType
    ](mut self, func: Func, message: StaticString) raises -> R:
        """`try_visit` with the member passed `mut`."""
        ref f = rebind[downcast[Func, def[U: F](mut U) raises -> R]](func)
        comptime for i in range(Self.Ts.length):
            comptime T = Self.Ts[i]
            comptime if conforms_to(T, F):
                if self.isa[T]():
                    return f(rebind[downcast[T, F]](self.get[T]()))
        raise TypeError(message)

    # --- equality, hashing, writing ---

    def __eq__(self, other: Self) -> Bool:
        comptime for i in range(Self.Ts.length):
            comptime T = Self.Ts[i]
            comptime assert conforms_to(
                T, Equatable
            ), "a member is not Equatable"
            comptime if conforms_to(T, Equatable):
                if self.isa[T]():
                    return other.isa[T]() and self.get[T]() == other.get[T]()
        return False

    def __hash__[H: Hasher](self, mut hasher: H):
        comptime for i in range(Self.Ts.length):
            comptime T = Self.Ts[i]
            comptime assert conforms_to(T, Hashable), "a member is not Hashable"
            comptime if conforms_to(T, Hashable):
                if self.isa[T]():
                    UInt8(i).__hash__(hasher)
                    self.get[T]().__hash__(hasher)

    def write_to(self, mut writer: Some[Writer]):
        comptime for i in range(Self.Ts.length):
            comptime T = Self.Ts[i]
            comptime assert conforms_to(T, Writable), "a member is not Writable"
            comptime if conforms_to(T, Writable):
                if self.isa[T]():
                    self.get[T]().write_to(writer)

    def write_repr_to(self, mut writer: Some[Writer]):
        comptime for i in range(Self.Ts.length):
            comptime T = Self.Ts[i]
            comptime assert conforms_to(T, Writable), "a member is not Writable"
            comptime if conforms_to(T, Writable):
                if self.isa[T]():
                    self.get[T]().write_repr_to(writer)


comptime _Members[
    P: Policy, Family: type_of(AnyType), *Ts: Movable & Deinitable
] = EnumVariant[Family, P, Ts, *Ts.map[P.Box]()]
"""Members `Ts` of `Family`, each held as `P` says."""

comptime Members[
    Family: type_of(AnyType), *Ts: Movable & Deinitable
] = _Members[Inline, Family, *Ts]
"""Members held in the variant: copies are independent."""

comptime ArcMembers[
    Family: type_of(AnyType), *Ts: Movable & Deinitable
] = _Members[Arc, Family, *Ts]
"""Members behind an `ArcPointer`: copies share the member."""

comptime OwnedMembers[
    Family: type_of(AnyType), *Ts: Movable & Deinitable
] = _Members[Owned, Family, *Ts]
"""Members behind an `OwnedPointer`: a copy allocates a copy."""


# ---------------------------------------------------------------------------
# The handle
# ---------------------------------------------------------------------------


trait Enum(Movable):
    """A type-erased handle: one of its family's members, in a storage.

    A handle names its storage type `Storage` and lends it out with
    `storage()`, since a trait cannot name a conformer's field. It writes its
    own `@implicit` constructor bound on its family, and its own copy
    constructor (`self._storage = copy._storage.copy()`).
    """

    # Mojo 1.2 added `match` / `case .Name(x)` over types conforming to
    # `std.builtin.EnumLike`. `Variant` does not conform yet; once it (or
    # `EnumVariant`) can, the visits below could become pattern matches.
    comptime Storage: EnumStorage

    def storage(ref self) -> ref[self] Self.Storage:
        ...

    def isa[T: Movable & Deinitable](self) -> Bool:
        return self.storage().isa[T]()

    def get[T: Movable & Deinitable](ref self) -> ref[self] T:
        return self.storage().get[T]()

    def visit[
        R: Movable, F: type_of(AnyType) = Self.Storage.Trait, *, Func: AnyType
    ](self, func: Func) -> R:
        return self.storage().visit[R, F](func)

    def visit_mut[
        R: Movable, F: type_of(AnyType) = Self.Storage.Trait, *, Func: AnyType
    ](mut self, func: Func) -> R:
        return self.storage().visit_mut[R, F](func)

    def try_visit[
        R: Movable, F: type_of(AnyType), *, Func: AnyType
    ](self, func: Func, message: StaticString) raises -> R:
        return self.storage().try_visit[R, F](func, message)

    def try_visit_mut[
        R: Movable, F: type_of(AnyType), *, Func: AnyType
    ](mut self, func: Func, message: StaticString) raises -> R:
        return self.storage().try_visit_mut[R, F](func, message)
