# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`Enum` and `EnumVariant`: the defaults a handle inherits over its storage,
under each policy, and a handle whose copies share one storage."""

from std.memory import ArcPointer
from std.sys import size_of
from std.testing import assert_equal, assert_false, assert_raises, assert_true

from ..enum import ArcMembers, Enum, Members, OwnedMembers, widen


trait Shape(Copyable, Equatable, Movable):
    def area(self) -> Float64:
        ...

    def scale(mut self, k: Float64):
        ...


trait Polygon(Shape):
    def sides(self) -> Int:
        ...


@fieldwise_init
struct Circle(Hashable, ImplicitlyCopyable, RegisterPassable, Shape, Writable):
    var r: Float64

    def area(self) -> Float64:
        return 3.0 * self.r * self.r

    def scale(mut self, k: Float64):
        self.r *= k


@fieldwise_init
struct Square(
    Hashable, ImplicitlyCopyable, Polygon, RegisterPassable, Writable
):
    var s: Float64

    def area(self) -> Float64:
        return self.s * self.s

    def scale(mut self, k: Float64):
        self.s *= k

    def sides(self) -> Int:
        return 4


struct DynShape(Copyable, Enum, Equatable, Hashable, Writable):
    comptime Storage = Members[Shape, Circle, Square]
    var _storage: Self.Storage

    @always_inline
    def storage(ref self) -> ref[self] Self.Storage:
        return widen[origin_of(self)](self._storage)

    @implicit
    def __init__[T: Movable & Deinitable](out self, var value: T):
        self._storage = Self.Storage(value^)

    def __init__(out self, var storage: Self.Storage, /):
        self._storage = storage^


def test_enum_implicit_init_and_isa() raises:
    var sq: DynShape = Square(2.0)
    var c: DynShape = Circle(1.0)
    assert_true(sq.isa[Square]())
    assert_false(sq.isa[Circle]())
    assert_true(c.isa[Circle]())


def test_enum_get_borrows() raises:
    var sq: DynShape = Square(2.0)
    assert_equal(sq.get[Square]().s, 2.0)
    sq.get[Square]().s = 3.0
    assert_equal(sq.get[Square]().s, 3.0)


def test_enum_equality() raises:
    var a: DynShape = Square(2.0)
    var b: DynShape = Square(2.0)
    var c: DynShape = Circle(2.0)
    assert_true(a == b)
    assert_false(a == c)


def test_enum_visit_reads() raises:
    var sq: DynShape = Square(2.0)
    var k = 10.0
    assert_equal(
        sq.try_visit[Float64, Shape](
            lambda [S: Shape](s: S) raises {imm} -> Float64: s.area() * k, ""
        ),
        40.0,
    )


def test_enum_visit_mutates() raises:
    var c: DynShape = Circle(1.0)

    var k = 2.0

    def double[S: Shape](mut s: S) raises {imm} -> Bool:
        s.scale(k)
        return True

    _ = c.try_visit_mut[Bool, Shape](double, "")
    assert_equal(c.get[Circle]().r, 2.0)


def test_enum_visit_narrows() raises:
    var sq: DynShape = Square(2.0)
    var c: DynShape = Circle(1.0)
    assert_equal(
        sq.try_visit[Int, Polygon](
            lambda [P: Polygon](p: P) raises {} -> Int: p.sides(),
            "not a polygon",
        ),
        4,
    )
    with assert_raises(contains="TypeError: not a polygon"):
        _ = c.try_visit[Int, Polygon](
            lambda [P: Polygon](p: P) raises {} -> Int: p.sides(),
            "not a polygon",
        )


def test_enum_hash() raises:
    var a: DynShape = Square(2.0)
    var b: DynShape = Square(2.0)
    var c: DynShape = Circle(2.0)
    assert_equal(hash(a), hash(b))
    assert_true(hash(a) != hash(c))


def test_enum_write() raises:
    var a: DynShape = Square(2.0)
    assert_equal(String(a), "DynShape(_storage=Square(s=2.0))")


def test_enum_copy() raises:
    var a: DynShape = Square(2.0)
    var b = a.copy()
    b.get[Square]().s = 3.0
    assert_equal(a.get[Square]().s, 2.0)
    assert_equal(b.get[Square]().s, 3.0)


@fieldwise_init
struct Plain(Copyable, Equatable, Movable):
    var v: Int


@fieldwise_init
struct Other(Copyable, Equatable, Movable):
    var v: Int


struct DynPlain(Copyable, Enum, Equatable):
    """Members that are neither `Hashable` nor `Writable`: the storage's
    hashing and writing would not build if used, and nothing uses them."""

    comptime Storage = Members[Equatable, Plain, Other]
    var _storage: Self.Storage

    @always_inline
    def storage(ref self) -> ref[self] Self.Storage:
        return widen[origin_of(self)](self._storage)

    @implicit
    def __init__[T: Movable & Deinitable](out self, var value: T):
        self._storage = Self.Storage(value^)

    def __init__(out self, var storage: Self.Storage, /):
        self._storage = storage^


def test_enum_unhashable_members() raises:
    var a: DynPlain = Plain(1)
    var b = a.copy()
    assert_true(a == b)
    assert_false(a == DynPlain(Other(1)))


struct SharedShape(Copyable, Enum):
    """Copies share one shape, as `DynBuilder`'s copies share one builder."""

    comptime Storage = Members[Shape, Circle, Square]
    var _storage: ArcPointer[Self.Storage]

    @always_inline
    def storage(ref self) -> ref[self] Self.Storage:
        return widen[origin_of(self)](self._storage[])

    @implicit
    def __init__[T: Movable & Deinitable](out self, var value: T):
        self._storage = ArcPointer(Self.Storage(value^))


def test_enum_shared_copies_share() raises:
    var a: SharedShape = Square(2.0)
    var b = a.copy()

    var k = 2.0

    def double[S: Shape](mut s: S) raises {imm} -> Bool:
        s.scale(k)
        return True

    _ = a.try_visit_mut[Bool, Shape](double, "")
    assert_true(b.isa[Square]())
    assert_equal(b.get[Square]().s, 4.0)
    assert_equal(
        b.try_visit[Float64, Shape](
            lambda [S: Shape](s: S) raises {} -> Float64: s.area(), ""
        ),
        16.0,
    )


def test_enum_shared_narrows() raises:
    var c: SharedShape = Circle(1.0)
    with assert_raises(contains="TypeError: not a polygon"):
        _ = c.try_visit[Int, Polygon](
            lambda [P: Polygon](p: P) raises {} -> Int: p.sides(),
            "not a polygon",
        )


def test_enum_visit_total() raises:
    var sq: DynShape = Square(2.0)
    var c: SharedShape = Circle(1.0)
    var k = 2.0
    assert_equal(
        sq.visit[Float64](
            lambda [S: Shape](s: S) raises {imm} -> Float64: s.area() * k
        ),
        8.0,
    )
    assert_equal(
        c.visit[Float64](
            lambda [S: Shape](s: S) raises {imm} -> Float64: s.area() * k
        ),
        6.0,
    )


struct DynPolygon(Enum):
    """A handle whose every member is also a `Polygon`."""

    comptime Storage = Members[Shape, Square]
    var _storage: Self.Storage

    @always_inline
    def storage(ref self) -> ref[self] Self.Storage:
        return widen[origin_of(self)](self._storage)

    @implicit
    def __init__[T: Movable & Deinitable](out self, var value: T):
        self._storage = Self.Storage(value^)


def test_enum_visit_defaults_to_the_storage_family() raises:
    var p: DynPolygon = Square(2.0)
    var k = 3.0
    # No family named: the storage's `Shape`.
    assert_equal(
        p.visit[Float64](
            lambda [S: Shape](s: S) raises {imm} -> Float64: s.area() * k
        ),
        12.0,
    )
    # A sub-family every member belongs to.
    assert_equal(
        p.visit[Int, Polygon](
            lambda [P: Polygon](q: P) raises {} -> Int: q.sides()
        ),
        4,
    )

    def double[S: Shape](mut s: S) raises {imm} -> Bool:
        s.scale(k)
        return True

    def sides[P: Polygon](mut q: P) raises {imm} -> Int:
        q.scale(k)
        return q.sides()

    _ = p.visit_mut[Bool](double)
    assert_equal(p.get[Square]().s, 6.0)
    assert_equal(p.visit_mut[Int, Polygon](sides), 4)
    assert_equal(p.get[Square]().s, 18.0)


@fieldwise_init
struct Labelled(Copyable, Hashable, Polygon, Writable):
    """A shape with a heap-allocated field."""

    var s: Float64
    var label: String

    def area(self) -> Float64:
        return self.s * self.s

    def scale(mut self, k: Float64):
        self.s *= k

    def sides(self) -> Int:
        return 4


struct Tracked(Copyable, Shape):
    """Counts its destructions in a counter every copy shares."""

    var drops: ArcPointer[Int]

    def __init__(out self, drops: ArcPointer[Int]):
        self.drops = drops

    def __init__(out self, *, copy: Self):
        self.drops = copy.drops

    def __deinit__(deinit self):
        self.drops[] += 1

    def __eq__(self, other: Self) -> Bool:
        return True

    def area(self) -> Float64:
        return 0.0

    def scale(mut self, k: Float64):
        pass


struct ArcShape(Copyable, Enum, Equatable):
    """Copies share one shape until one is changed."""

    comptime Storage = ArcMembers[Shape, Circle, Labelled, Tracked]
    var _storage: Self.Storage

    @always_inline
    def storage(ref self) -> ref[self] Self.Storage:
        return widen[origin_of(self)](self._storage)

    @implicit
    def __init__[T: Movable & Deinitable](out self, var value: T):
        self._storage = Self.Storage(value^)

    def __init__(out self, *, copy: Self):
        self._storage = copy._storage.copy()

    def holding(self, var shape: Labelled) -> Self:
        """A copy holding `shape`: a shared member is replaced, never
        changed in place."""
        var out = self.copy()
        out._storage = Self.Storage(shape^)
        return out^


def test_enum_arc_is_two_words() raises:
    assert_equal(size_of[ArcShape](), 2 * size_of[Int]())


def test_enum_arc_get_and_isa() raises:
    var c: ArcShape = Circle(1.5)
    var l: ArcShape = Labelled(2.0, "sq")
    assert_true(c.isa[Circle]())
    assert_false(c.isa[Labelled]())
    assert_false(c.isa[Square]())
    assert_equal(c.get[Circle]().r, 1.5)
    assert_true(l.isa[Labelled]())
    assert_equal(l.get[Labelled]().label, "sq")
    assert_equal(
        l.try_visit[Int, Polygon](
            lambda [P: Polygon](p: P) raises {} -> Int: p.sides(), ""
        ),
        4,
    )
    with assert_raises(contains="TypeError: not a polygon"):
        _ = c.try_visit[Int, Polygon](
            lambda [P: Polygon](p: P) raises {} -> Int: p.sides(),
            "not a polygon",
        )


def test_enum_arc_copies_share_and_replace() raises:
    var a: ArcShape = Labelled(2.0, "sq")
    var b = a.copy()
    assert_equal(_address(a.get[Labelled]()), _address(b.get[Labelled]()))
    var c = b.holding(Labelled(4.0, "c"))
    assert_equal(a.get[Labelled]().s, 2.0)
    assert_equal(b.get[Labelled]().s, 2.0)
    assert_equal(c.get[Labelled]().s, 4.0)
    assert_equal(c.get[Labelled]().label, "c")


def test_enum_arc_destroys_once() raises:
    var drops = ArcPointer[Int](0)
    var a: ArcShape = Tracked(drops)
    var b = a.copy()
    var c = b.copy()
    _ = a^
    _ = b^
    assert_equal(drops[], 0)
    _ = c^
    assert_equal(drops[], 1)


def test_enum_arc_equality_hash_write() raises:
    var a = ArcMembers[Shape, Circle, Labelled](Labelled(2.0, "x"))
    var b = ArcMembers[Shape, Circle, Labelled](Labelled(2.0, "x"))
    var c = ArcMembers[Shape, Circle, Labelled](Circle(2.0))
    assert_true(a == b)
    assert_false(a == c)
    assert_equal(hash(a), hash(b))
    assert_true(hash(a) != hash(c))
    assert_equal(String(c), String(Circle(2.0)))


struct Dot(Shape):
    """A shape with no fields."""

    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True

    def area(self) -> Float64:
        return 0.0

    def scale(mut self, k: Float64):
        pass


def _address[T: AnyType](ref value: T) -> Int:
    return Int(Pointer(to=value))


def test_enum_boxes_zero_sized_members() raises:
    comptime Dots = ArcMembers[Shape, Circle, Dot]
    var a = Dots(Dot())
    var b = a.copy()
    # Boxed like any other member: copies share it.
    assert_equal(_address(a.get[Dot]()), _address(b.get[Dot]()))
    assert_true(a == b)
    _ = b^
    assert_true(a.isa[Dot]())
    comptime OwnedDots = OwnedMembers[Shape, Circle, Dot]
    var o = OwnedDots(Dot())
    assert_true(o.copy().isa[Dot]())


struct OwnedShape(Copyable, Enum, Equatable):
    """Each copy owns its shape behind its own pointer."""

    comptime Storage = OwnedMembers[Shape, Circle, Labelled, Tracked]
    var _storage: Self.Storage

    @always_inline
    def storage(ref self) -> ref[self] Self.Storage:
        return widen[origin_of(self)](self._storage)

    @implicit
    def __init__[T: Movable & Deinitable](out self, var value: T):
        self._storage = Self.Storage(value^)

    def __init__(out self, *, copy: Self):
        self._storage = copy._storage.copy()


def test_enum_owned_is_two_words() raises:
    assert_equal(size_of[OwnedShape](), 2 * size_of[Int]())


def test_enum_owned_get_and_isa() raises:
    var c: OwnedShape = Circle(1.5)
    var l: OwnedShape = Labelled(2.0, "sq")
    assert_true(c.isa[Circle]())
    assert_false(c.isa[Labelled]())
    assert_false(c.isa[Square]())
    assert_equal(c.get[Circle]().r, 1.5)
    assert_true(l.isa[Labelled]())
    assert_equal(l.get[Labelled]().label, "sq")
    assert_equal(
        l.try_visit[Int, Polygon](
            lambda [P: Polygon](p: P) raises {} -> Int: p.sides(), ""
        ),
        4,
    )
    with assert_raises(contains="TypeError: not a polygon"):
        _ = c.try_visit[Int, Polygon](
            lambda [P: Polygon](p: P) raises {} -> Int: p.sides(),
            "not a polygon",
        )


def test_enum_owned_visit_defaults_to_the_storage_family() raises:
    var l: OwnedShape = Labelled(2.0, "sq")
    var k = 3.0
    assert_equal(
        l.visit[Float64](
            lambda [S: Shape](s: S) raises {imm} -> Float64: s.area() * k
        ),
        12.0,
    )


def test_enum_owned_copies_are_independent() raises:
    var a: OwnedShape = Labelled(2.0, "sq")
    var b = a.copy()
    var k = 2.0

    def double[S: Shape](mut s: S) raises {imm} -> Bool:
        s.scale(k)
        return True

    _ = b.try_visit_mut[Bool, Shape](double, "")
    assert_equal(a.get[Labelled]().s, 2.0)
    assert_equal(b.get[Labelled]().s, 4.0)
    _ = b.visit_mut[Bool](double)
    assert_equal(a.get[Labelled]().s, 2.0)
    assert_equal(b.get[Labelled]().s, 8.0)
    assert_equal(b.get[Labelled]().label, "sq")
    b.get[Labelled]().s = 5.0
    assert_equal(a.get[Labelled]().s, 2.0)
    assert_equal(b.get[Labelled]().s, 5.0)


def test_enum_owned_destroys_each_copy_once() raises:
    var drops = ArcPointer[Int](0)
    var a: OwnedShape = Tracked(drops)
    var b = a.copy()
    var c = b.copy()
    _ = a^
    assert_equal(drops[], 1)
    _ = b^
    assert_equal(drops[], 2)
    var d = c^
    assert_equal(drops[], 2)
    _ = d^
    assert_equal(drops[], 3)


def test_enum_owned_equality_hash_write() raises:
    var a = OwnedMembers[Shape, Circle, Labelled](Labelled(2.0, "x"))
    var b = OwnedMembers[Shape, Circle, Labelled](Labelled(2.0, "x"))
    var c = OwnedMembers[Shape, Circle, Labelled](Circle(2.0))
    assert_true(a == b)
    assert_false(a == c)
    assert_equal(hash(a), hash(b))
    assert_true(hash(a) != hash(c))
    assert_equal(String(c), String(Circle(2.0)))
    assert_equal(String(a.copy()), String(Labelled(2.0, "x")))
