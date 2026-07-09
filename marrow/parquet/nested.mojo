"""List (single-level) decoding via Dremel repetition/definition levels.

A repeated leaf yields, per page, a value for each *present* element plus full
rep/def levels for every slot. `LeveledColumnReader` drives a `PageReader`
through a growable `ChildBuilder` (one per element type) to build the element
child array, while accumulating the levels; `assemble_list` then turns the levels
into Arrow list offsets and list-level validity.

This is a separate path from the flat `ColumnReader`: list child arrays are
data-sized (not `num_rows`), so they use growable builders rather than the flat
memcpy fast path. Value reading reuses the shared `read_*` helpers.
"""

from std.sys import size_of

from ..arrays import AnyArray, ListArray
from ..builders import PrimitiveBuilder, BinaryLikeBuilder, BoolBuilder
from ..dtypes import (
    NumericType,
    BinaryLikeType,
    Int32Type,
    Int64Type,
    UInt32Type,
    UInt64Type,
    Float32Type,
    Float64Type,
    Int8Type,
    Int16Type,
    UInt8Type,
    UInt16Type,
    StringType,
    LargeStringType,
    BinaryType,
    LargeBinaryType,
    bool_,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
)

from .page import Page, PageReader, PAGEKIND_DICT, read_u32le, read_fixed_le
from .encoding import rle_decode
from .compression import Codecs
from .schema import LeafColumn
from .format import ColumnMetaData


struct DecodedLeaf(Movable):
    """One decoded leaf column. A flat leaf carries just its array; a repeated
    (list-element) leaf also carries the per-slot rep/def levels. All leaves
    under the same list share these levels, and the list geometry (which def
    levels mean present/empty/null) lives on the schema node, so this stays a
    plain data record."""

    var leveled: Bool
    var array: AnyArray  # flat column, or the list's element/child array
    var rep_levels: List[Int32]
    var def_levels: List[Int32]

    def __init__(
        out self,
        leveled: Bool,
        var array: AnyArray,
        var rep_levels: List[Int32],
        var def_levels: List[Int32],
    ):
        self.leveled = leveled
        self.array = array^
        self.rep_levels = rep_levels^
        self.def_levels = def_levels^

    @staticmethod
    def flat(var array: AnyArray) -> DecodedLeaf:
        return DecodedLeaf(False, array^, List[Int32](), List[Int32]())


# ---------------------------------------------------------------------------
# ChildBuilder — accumulates the element (child) array, growing as needed
# ---------------------------------------------------------------------------


trait ChildBuilder(ImplicitlyDeletable, Movable):
    def dict_page(mut self, var page: Page) raises:
        ...

    def present_plain(mut self, vspan: Span[UInt8, _], mut vi: Int) raises:
        ...

    def present_dict(mut self, idx: Int) raises:
        ...

    def null(mut self) raises:
        ...

    def finish(deinit self) raises -> AnyArray:
        ...


struct PrimChild[T: NumericType, phys: DType](ChildBuilder):
    var builder: PrimitiveBuilder[Self.T]
    var dict: List[Scalar[Self.T.native]]

    def __init__(out self, capacity: Int):
        self.builder = PrimitiveBuilder[Self.T](capacity)
        self.dict = List[Scalar[Self.T.native]]()

    def dict_page(mut self, var page: Page) raises:
        comptime PW = size_of[Scalar[Self.phys]]()
        var span = page.body
        for i in range(page.num_values):
            self.dict.append(
                read_fixed_le[Self.phys](span, i * PW).cast[Self.T.native]()
            )

    def present_plain(mut self, vspan: Span[UInt8, _], mut vi: Int) raises:
        comptime PW = size_of[Scalar[Self.phys]]()
        self.builder.append(
            read_fixed_le[Self.phys](vspan, vi).cast[Self.T.native]()
        )
        vi += PW

    def present_dict(mut self, idx: Int) raises:
        self.builder.append(self.dict[idx])

    def null(mut self) raises:
        self.builder.append_null()

    def finish(deinit self) raises -> AnyArray:
        var b = self.builder^
        var out: AnyArray = b.finish()
        return out^


struct BytesChild[BT: BinaryLikeType](ChildBuilder):
    var builder: BinaryLikeBuilder[Self.BT]
    var dict_body: List[UInt8]
    var dict_off: List[Int]
    var dict_len: List[Int]

    def __init__(out self, capacity: Int):
        self.builder = BinaryLikeBuilder[Self.BT](capacity)
        self.dict_body = List[UInt8]()
        self.dict_off = List[Int]()
        self.dict_len = List[Int]()

    def dict_page(mut self, var page: Page) raises:
        self.dict_body = List[UInt8]()
        self.dict_body.extend(page.body)
        var span = Span(self.dict_body)
        var off = 0
        for _ in range(page.num_values):
            var n = read_u32le(span, off)
            off += 4
            self.dict_off.append(off)
            self.dict_len.append(n)
            off += n

    def present_plain(mut self, vspan: Span[UInt8, _], mut vi: Int) raises:
        var n = read_u32le(vspan, vi)
        vi += 4
        self.builder.append(StringSlice(unsafe_from_utf8=vspan[vi : vi + n]))
        vi += n

    def present_dict(mut self, idx: Int) raises:
        var start = self.dict_off[idx]
        var entry = List[UInt8]()
        entry.extend(Span(self.dict_body)[start : start + self.dict_len[idx]])
        self.builder.append(StringSlice(unsafe_from_utf8=Span(entry)))

    def null(mut self) raises:
        self.builder.append_null()

    def finish(deinit self) raises -> AnyArray:
        var b = self.builder^
        var out: AnyArray = b.finish()
        return out^


# ---------------------------------------------------------------------------
# LeveledColumnReader — drive pages through a ChildBuilder
# ---------------------------------------------------------------------------


struct LeveledColumnReader[o: Origin[mut=False]](Movable):
    var pages: PageReader[Self.o]
    var num_rows: Int

    def __init__(
        out self,
        data: Span[UInt8, Self.o],
        var meta: ColumnMetaData,
        var leaf: LeafColumn,
        num_rows: Int,
    ):
        self.pages = PageReader(data, meta^, leaf^)
        self.num_rows = num_rows

    def _drive[
        B: ChildBuilder
    ](
        mut self,
        var child: B,
        mut codecs: Codecs,
        floor: Int,
        max_def: Int,
    ) raises -> DecodedLeaf:
        var rep_out = List[Int32]()
        var def_out = List[Int32]()
        while self.pages.has_next():
            var pg = self.pages.next(codecs)
            if pg.kind == PAGEKIND_DICT:
                child.dict_page(pg^)
                continue
            rep_out.extend(Span(pg.rep_levels))
            def_out.extend(Span(pg.def_levels))
            var vspan = pg.values()
            var use_dict = pg.is_dictionary()
            var indices = List[Int32]()
            if use_dict:
                indices = rle_decode(vspan[1:], Int(vspan[0]), pg.num_present)
            var vi = 0
            var di = 0
            for k in range(pg.num_values):
                var d = Int(pg.def_levels[k])
                if d < floor:
                    continue  # list null/empty — no element in this slot
                if d == max_def:
                    if use_dict:
                        child.present_dict(Int(indices[di]))
                        di += 1
                    else:
                        child.present_plain(vspan, vi)
                else:
                    child.null()
        var values = child^.finish()
        return DecodedLeaf(
            leveled=True,
            array=values^,
            rep_levels=rep_out^,
            def_levels=def_out^,
        )

    def read(mut self, mut codecs: Codecs) raises -> DecodedLeaf:
        ref leaf = self.pages.leaf
        ref dt = leaf.dtype
        # a value slot exists at/above the leaf's repetition floor (the innermost
        # enclosing list's element level); the value is present at max_def.
        var floor = leaf.rep_floor
        var md = leaf.max_def
        var cap = self.num_rows
        if dt == int32:
            return self._drive(
                PrimChild[Int32Type, DType.int32](cap), codecs, floor, md
            )
        elif dt == int64:
            return self._drive(
                PrimChild[Int64Type, DType.int64](cap), codecs, floor, md
            )
        elif dt == uint32:
            return self._drive(
                PrimChild[UInt32Type, DType.uint32](cap), codecs, floor, md
            )
        elif dt == uint64:
            return self._drive(
                PrimChild[UInt64Type, DType.uint64](cap), codecs, floor, md
            )
        elif dt == float32:
            return self._drive(
                PrimChild[Float32Type, DType.float32](cap), codecs, floor, md
            )
        elif dt == float64:
            return self._drive(
                PrimChild[Float64Type, DType.float64](cap), codecs, floor, md
            )
        elif dt == int8:
            return self._drive(
                PrimChild[Int8Type, DType.int32](cap), codecs, floor, md
            )
        elif dt == int16:
            return self._drive(
                PrimChild[Int16Type, DType.int32](cap), codecs, floor, md
            )
        elif dt == uint8:
            return self._drive(
                PrimChild[UInt8Type, DType.int32](cap), codecs, floor, md
            )
        elif dt == uint16:
            return self._drive(
                PrimChild[UInt16Type, DType.int32](cap), codecs, floor, md
            )
        elif dt.is_string():
            return self._drive(BytesChild[StringType](cap), codecs, floor, md)
        elif dt.is_large_string():
            return self._drive(
                BytesChild[LargeStringType](cap), codecs, floor, md
            )
        elif dt.is_binary():
            return self._drive(BytesChild[BinaryType](cap), codecs, floor, md)
        elif dt.is_large_binary():
            return self._drive(
                BytesChild[LargeBinaryType](cap), codecs, floor, md
            )
        else:
            raise Error("parquet: unsupported list element type " + String(dt))


# ---------------------------------------------------------------------------
# List assembly — rep/def levels → offsets + list validity
# ---------------------------------------------------------------------------


def assemble_list(
    var element: AnyArray,
    rep_levels: List[Int32],
    def_levels: List[Int32],
    list_def: Int,
    element_floor: Int,
    list_optional: Bool,
) raises -> AnyArray:
    """Fold a decoded element array + its Dremel levels into an Arrow
    `ListArray`.

    A new list starts at every `rep == 0` slot; a slot holds an element when its
    definition level reaches `element_floor`; the list itself is null when the
    definition level is below `list_def`. `element` is the list's child array
    (a leaf column or an assembled struct) — one entry per present element.

    This handles a single level of repetition (`list<T>`, `list<struct<...>>`).
    """
    var n = len(rep_levels)
    var offsets = PrimitiveBuilder[Int32Type](n + 1)
    var mask = BoolBuilder(n)
    var any_null = False
    var child_idx = 0
    offsets.append(Int32(0))
    for i in range(n):
        var r = Int(rep_levels[i])
        var d = Int(def_levels[i])
        if r == 0:
            if i > 0:
                offsets.append(Int32(child_idx))
            var is_null = list_optional and d < list_def
            mask.append(is_null)
            if is_null:
                any_null = True
        if d >= element_floor:
            child_idx += 1
    offsets.append(Int32(child_idx))

    var offsets_arr = offsets.finish()
    var out: AnyArray
    if any_null:
        var m = mask.finish()
        out = ListArray.from_arrays(offsets_arr, element^, m^)
    else:
        out = ListArray.from_arrays(offsets_arr, element^, None)
    return out^
