# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`Enum`: the defaults a handle inherits over its `Variant` or
`SharedVariant` field, and the storages' visits."""

from std.testing import assert_equal, assert_false, assert_raises, assert_true

from ..enum import Enum, SharedVariant, UniqueVariant, widen


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
    comptime Storage = UniqueVariant[Shape, Circle, Square]
    var _storage: Self.Storage

    @always_inline
    def storage(ref self) -> ref[self] Self.Storage:
        return widen[origin_of(self)](self._storage)

    @implicit
    def __init__[T: Movable](out self, var value: T):
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
    """Members that are neither `Hashable` nor `Writable`, so neither is it."""

    comptime Storage = UniqueVariant[Equatable, Plain, Other]
    var _storage: Self.Storage

    @always_inline
    def storage(ref self) -> ref[self] Self.Storage:
        return widen[origin_of(self)](self._storage)

    @implicit
    def __init__[T: Movable](out self, var value: T):
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

    comptime Storage = SharedVariant[Shape, Circle, Square]
    var _storage: Self.Storage

    @always_inline
    def storage(ref self) -> ref[self] Self.Storage:
        return widen[origin_of(self)](self._storage)

    @implicit
    def __init__[T: Movable](out self, var value: T):
        self._storage = Self.Storage(value^)

    def __init__(out self, var storage: Self.Storage, /):
        self._storage = storage^


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
