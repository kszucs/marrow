"""Choosing a backend at run time, from a URI.

The one file in `marrow/io` that names every provider, which is why it is
the one that cannot live in `core.mojo`: core states what a backend must
provide and depends on nothing, and a dispatcher there would make it
depend on all of them. (Mojo would accept the cycle -- it resolves
circular imports within a package -- so this is a layering choice, not a
language constraint.)

`read_table("s3://bucket/x.parquet")` cannot know its source type at compile
time, so `DynSource` and `DynSink` erase it. They are the *only* thing in
`marrow/io` that names every backend, which is what keeps the choice out of the
readers and writers -- and out of any binary that does not ask for it.

**These two conform to the traits they erase, unlike every `Dyn*` box in the
tree.** CLAUDE.md states that rule without exception, and the exception is
deliberate: it was written because `DynArray` had no generic consumer, whereas
`ParquetFile[S: ByteSource]`, `Index.from_parquet[S]` and `page_selections[S]`
are real ones and are the entire reason these boxes exist. The costs that
motivated the rule do not apply either -- `ByteSource` and `ByteSink` require
only `Deinitable, Movable`, which a `Variant` field satisfies for free. The exception is recorded in CLAUDE.md next to the
rule itself, which is the only place a reader would look for it.

**A `Variant`, not a trampoline.** `backlog.md` prescribed the trampoline and
flagged the risk that sank it: a `thin fn` slot has a concrete signature, so
`read_at`'s `Span[UInt8, origin_of(self)]` would have to name `ImmutAnyOrigin`,
which the buffer rules forbid outright. `Variant.__getitem__` hands back a
*tracked* reference, so the origin survives the same one-line widening
`BufferSource` already does. A variant also destroys its member at the true type,
so the `_drop`-trampoline hazard every erased box here otherwise needs does not
arise.

The ladders below are written out per method rather than routed through a
shared narrowing closure. That is not repetition for its own sake:
**+662,740 bytes** on one gate is what the shared helper cost the last time it
was tried.

The closed type set costs nothing here. An instrumenting or test-local source
is still supplied as the comptime `S` -- which is how
`marrow/parquet/tests/test_page_io.mojo` does it, and it is unaffected.
"""

from std.os import abort
from std.utils import Variant

from .opendal import OpenDalStore, OpenDalWriter, OpenDalSource
from .core import ByteSink, ByteSource, Fetched
from .local import BufferSource, FileSink, MemorySink
from ..utils.uri import StorageOptions, Uri


struct DynSource(ByteSource):
    """A `ByteSource` chosen at run time."""

    comptime VariantType = Variant[BufferSource, OpenDalSource]
    """Every member is pointer-, `Int`- or `String`-backed, so all are
    8-aligned. That matters: a `List` of a `Variant` whose largest member is not
    its most-aligned one silently loses every other element (see CLAUDE.md).
    Do not add a 32-aligned member without re-reading that note."""

    var _v: Self.VariantType

    @implicit
    def __init__(out self, var s: BufferSource):
        self._v = Self.VariantType(s^)

    @implicit
    def __init__(out self, var s: OpenDalSource):
        self._v = Self.VariantType(s^)

    def size(self) -> Int:
        comptime for i in range(len(Self.VariantType.Ts)):
            comptime T = Self.VariantType.Ts[i]
            comptime if conforms_to(T, ByteSource):
                if self._v.isa[T]():
                    return self._v[T].size()
        abort("DynSource.size: no arm matched")

    def read_at(
        ref self, offset: Int, length: Int
    ) raises -> Span[UInt8, origin_of(self)]:
        comptime for i in range(len(Self.VariantType.Ts)):
            comptime T = Self.VariantType.Ts[i]
            comptime if conforms_to(T, ByteSource):
                if self._v.isa[T]():
                    return rebind[Span[UInt8, origin_of(self)]](
                        self._v[T].read_at(offset, length)
                    )
        raise Error("DynSource.read_at: no arm matched")

    def read_ranges(ref self, ranges: List[Tuple[Int, Int]]) raises -> Fetched:
        comptime for i in range(len(Self.VariantType.Ts)):
            comptime T = Self.VariantType.Ts[i]
            comptime if conforms_to(T, ByteSource):
                if self._v.isa[T]():
                    return self._v[T].read_ranges(ranges)
        raise Error("DynSource.read_ranges: no arm matched")

    @staticmethod
    def open(
        uri: String, options: StorageOptions = StorageOptions()
    ) raises -> Self:
        """Open `uri` for reading, choosing a backend from its scheme.

        A bare path or `file://` takes the local memory map **without touching
        `dlopen`**, which is what makes marrow work when `libopendal_c` is not
        installed — enforced by this branch rather than by a comment.
        Everything else goes to OpenDAL, whose constructor raises naming the
        library if it is missing.
        """
        var u = Uri.parse(uri)
        if u.is_local():
            return Self(BufferSource(u.local_path()))
        return Self(OpenDalSource(_store_for(u, options), u.object_key()))


struct DynSink(ByteSink):
    """A `ByteSink` chosen at run time. See `DynSource`."""

    comptime VariantType = Variant[FileSink, MemorySink, OpenDalWriter]
    var _v: Self.VariantType

    @implicit
    def __init__(out self, var s: FileSink):
        self._v = Self.VariantType(s^)

    @implicit
    def __init__(out self, var s: MemorySink):
        self._v = Self.VariantType(s^)

    @implicit
    def __init__(out self, var s: OpenDalWriter):
        self._v = Self.VariantType(s^)

    def write[o: Origin[mut=False]](mut self, data: Span[UInt8, o]) raises:
        comptime for i in range(len(Self.VariantType.Ts)):
            comptime T = Self.VariantType.Ts[i]
            comptime if conforms_to(T, ByteSink):
                if self._v.isa[T]():
                    self._v[T].write(data)
                    return
        raise Error("DynSink.write: no arm matched")

    def close(mut self) raises:
        comptime for i in range(len(Self.VariantType.Ts)):
            comptime T = Self.VariantType.Ts[i]
            comptime if conforms_to(T, ByteSink):
                if self._v.isa[T]():
                    self._v[T].close()
                    return
        raise Error("DynSink.close: no arm matched")

    @staticmethod
    def open(
        uri: String, options: StorageOptions = StorageOptions()
    ) raises -> Self:
        """Open `uri` for writing. Nothing is published until `close()`."""
        var u = Uri.parse(uri)
        if u.is_local():
            return Self(FileSink(u.local_path()))
        return Self(_store_for(u, options).writer(u.object_key()))


def _store_for(u: Uri, options: StorageOptions) raises -> OpenDalStore:
    """The operator `u` names. One line, but both openers need it and getting
    the two out of step would build an operator for the right service with the
    wrong options."""
    return OpenDalStore(u.service(), options.resolve(u))
