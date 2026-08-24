"""The compiler contracts the `t.amount` surface is built on.

These are **not** tests of marrow code. They pin four behaviours of the Mojo
compiler that a schema-carrying handle needs, each of which was probed for
Phase 0 and each of which a language upgrade could withdraw silently. Written
down as prose in a design document, a withdrawn contract surfaces as a baffling
error in whatever is built on top; written down here, it surfaces as a named
failing test.

One of the four contradicts what CLAUDE.md claimed until 2026-08-22 — a
conditional comptime type *does* reduce at a return site and *does* carry its
trait bound — so the cost of not having these as tests is already demonstrated.
"""

from std.testing import assert_equal, assert_true

from ...builders import col
from ....dtypes import Float64Type, Int64Type, int64
from ..leaves import Column


struct MiniSchema(Copyable, Movable):
    """The smallest thing that can stand in for a schema at comptime.

    Deliberately holds `List`s: the point of the first contract is that a
    **heap-allocated** field does not disqualify a struct from being a comptime
    parameter.
    """

    var names: List[String]
    var codes: List[Int]

    def __init__(out self, var names: List[String], var codes: List[Int]):
        self.names = names^
        self.codes = codes^

    def index_of(self, name: String) -> Int:
        """Non-raising, so it is comptime-eligible. A raising `def` is not —
        which is why a missing column returns `-1` rather than raising, and why
        the caller asserts on the result."""
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        return -1


comptime SCHEMA = MiniSchema(["a", "b"], [0, 1])
"""Contract 1: a struct holding heap-allocated fields is a valid comptime
parameter. This line failing to compile *is* the failure."""


struct Handle[s: MiniSchema](Copyable, Movable):
    """A schema-carrying handle — what `t` is in `t.amount`."""

    def __init__(out self):
        pass

    def __getattr_param__[
        name: String
    ](self) -> Column[
        Int64Type if Self.s.codes[Self.s.index_of(name)] == 0 else Float64Type
    ]:
        """Contracts 2-4, all in this signature.

        2. `__getattr_param__` fires for an attribute the struct does not have.
        3. The **return type depends on the parameter** — a conditional comptime
           type that reduces at a return site *and* satisfies `Column`'s
           `NumericType` bound. Both branches must always be well-formed;
           totality is what makes it reduce.
        4. `comptime assert` on `Self.s` turns an unknown column into a build
           failure carrying its own message.
        """
        comptime assert Self.s.index_of(name) >= 0, "unknown column: " + name
        comptime T = Int64Type if Self.s.codes[
            Self.s.index_of(name)
        ] == 0 else Float64Type
        return Column[T](name)


def test_handle_resolves_an_attribute_to_a_column() raises:
    var t = Handle[SCHEMA]()
    assert_equal(t.a.name(), "a")
    assert_equal(t.b.name(), "b")


def test_handle_infers_each_column_dtype_from_the_schema() raises:
    """The contract that makes `t.amount` worth having over `col("amount",
    int64)`: two attributes of one handle have genuinely different types."""
    var t = Handle[SCHEMA]()
    # `type_of(...)` alone is a comptime-only use, which reads to the compiler
    # as `t` never being used at all — hence a runtime assertion beside it.
    assert_equal(t.a.name(), "a")
    assert_true(type_of(t.a).Type == Int64Type)
    assert_true(type_of(t.b).Type == Float64Type)


def test_handle_column_is_a_normal_fused_leaf() raises:
    """The inferred type is a real `Column[T]`, not something merely shaped like
    one — it reports the shape the fusion driver keys off."""
    var t = Handle[SCHEMA]()
    assert_true(type_of(t.a).shape == type_of(col("a", int64)).shape)
    assert_equal(len(t.a.columns()), 1)
    assert_equal(t.a.columns()[0], "a")


# An unknown column is a **compile** error, so it cannot be asserted from a
# running test. Verified by hand 2026-08-22; uncomment to re-verify:
#
#     var t = Handle[SCHEMA]()
#     _ = t.nope
#
# => constraint failed: unknown column: nope
#
# It must be `pytest` that verifies it, never `precompile`: precompile
# elaborates signatures and not bodies, so even `comptime assert False` in this
# position reports nothing and the validation reads as absent when it is not.
