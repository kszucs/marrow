"""`variant_dispatch` — runtime dispatch over a `Variant` with no vtable.

The point of this helper is that it resolves the *active* member and runs the
caller's closure on it, so these assert **which arm ran**, not merely that a
value came back. Marrow's four erased containers (`DynType`, `DynArray`,
`DynScalar`, `DynBuilder`) all sit on it, and a dispatch that silently picked
the wrong arm would be a wrong answer everywhere rather than an error.

Arms are identified by `size_of[T]()` rather than by comparing `T` to a concrete
type: neither marrow nor the standard library has a comptime type-identity
predicate, and the member sizes here are distinct, which is all the test needs.
"""

from std.testing import assert_equal, assert_true, assert_raises
from std.sys import size_of
from std.utils import Variant

from ..dispatch import variant_dispatch, variant_dispatch_raises


comptime _IntOrBool = Variant[Int, Bool]
comptime _Triple = Variant[Int, Bool, Float32]


def test_variant_dispatch_selects_the_active_member() raises:
    def arm_size[T: Movable](t: T) {imm} -> Int:
        return size_of[T]()

    var as_int = _IntOrBool(42)
    assert_equal(variant_dispatch(as_int, arm_size), size_of[Int]())

    var as_bool = _IntOrBool(True)
    assert_equal(variant_dispatch(as_bool, arm_size), size_of[Bool]())


def test_variant_dispatch_reaches_the_last_member() raises:
    """The loop walks members in order; the final one has to be reachable
    rather than falling through to the `abort`."""

    def arm_size[T: Movable](t: T) {imm} -> Int:
        return size_of[T]()

    var v = _Triple(Float32(1.5))
    assert_equal(variant_dispatch(v, arm_size), size_of[Float32]())


def test_variant_dispatch_runs_the_closure_exactly_once() raises:
    """One arm matches, so the closure runs once — not once per member, and not
    zero times with a default returned."""
    var calls = 0

    def count[T: Movable](t: T) {mut calls, imm} -> Int:
        calls += 1
        return size_of[T]()

    var v = _IntOrBool(7)
    var _ = variant_dispatch(v, count)
    assert_equal(calls, 1)


def test_variant_dispatch_raises_propagates_the_error() raises:
    """The raising form is a separate name, not an overload: a non-raising
    closure also satisfies `raises`, which would make one overload set
    ambiguous at every call site."""

    def boom[T: Movable](t: T) raises {imm} -> Int:
        raise Error("arm raised")

    var v = _IntOrBool(1)
    with assert_raises(contains="arm raised"):
        var _ = variant_dispatch_raises(v, boom)


def test_variant_dispatch_raises_returns_normally() raises:
    def arm_size[T: Movable](t: T) raises {imm} -> Int:
        return size_of[T]()

    var as_bool = _IntOrBool(False)
    assert_equal(variant_dispatch_raises(as_bool, arm_size), size_of[Bool]())


def test_variant_dispatch_mut_overload_dispatches() raises:
    """The `mut` overload hands the closure a mutable reference to the active
    member — `DynBuilder._dispatch_mut` is its only caller, and it is the one
    path that can append into the variant in place."""
    var calls = 0

    def arm_size[T: Movable](mut t: T) raises {mut calls, imm} -> Int:
        calls += 1
        return size_of[T]()

    var v = _IntOrBool(3)
    assert_equal(variant_dispatch_raises(v, arm_size), size_of[Int]())
    assert_equal(calls, 1)
