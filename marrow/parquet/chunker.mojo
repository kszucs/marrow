"""Content-defined chunking for the Parquet writer.

Data page boundaries derived from a rolling gear hash over the column's
`(def_level, rep_level, value)` stream, so an edit early in a column stops
shifting every page after it and a content-addressable store can deduplicate
the unchanged tail. Only the writer is aware of this; the reader is unaffected.

Ported from Arrow C++ `parquet/chunker_internal.{h,cc}` and arrow-rs
`column/chunker/cdc.rs`; `marrow/parquet/tests/test_chunker_parity.mojo` pins
the boundaries against Arrow C++'s own, through pyarrow.
"""

from std.bit import bit_width
from std.sys import size_of

from ..arrays import DynArray, PrimitiveArray, BinaryLikeArray
from .. import dtypes as dt
from .gearhash import gearhash_table, NUM_GEARHASH_TABLES


struct ContentDefinedChunking(Copyable, Movable):
    """How to chunk: the size envelope and the normalisation level.

    `min_chunk_size` is a skip window -- the rolling hash is not updated until
    a chunk reaches it, which is FastCDC's cut-point skipping and is also why
    the mask targets a smaller size than the average. `max_chunk_size` is a
    hard cut. Every byte fed to the hash counts toward the size, definition and
    repetition levels included.

    `norm_level` widens or narrows the mask. Raising it makes a match more
    likely, which tightens the size distribution around the average and
    improves the deduplication ratio at the cost of more small pages. Outside
    [-3, 3] is not useful.
    """

    var min_chunk_size: Int
    var max_chunk_size: Int
    var norm_level: Int
    var mask: UInt64
    """The rolling-hash mask a boundary candidate must clear, computed once
    here from the size envelope -- so a `ContentDefinedChunking` is valid by
    construction and `ContentDefinedChunker` just reads the field instead of
    recomputing (and re-validating) it once per leaf per row group."""

    def __init__(
        out self,
        min_chunk_size: Int = 256 * 1024,
        max_chunk_size: Int = 1024 * 1024,
        norm_level: Int = 0,
    ) raises:
        """Validates the size envelope and derives `mask` from it.

        A gear hash distributes uniformly, so a mask with the top N bits set
        matches with probability 1/2^N. Two adjustments: the first
        `min_chunk_size` bytes are skipped, so the mask targets the average
        *minus* that window; and a boundary needs `NUM_GEARHASH_TABLES`
        consecutive matches, so the per-match target is divided by that too.
        """
        if min_chunk_size < 0:
            raise Error("parquet: cdc min_chunk_size must be non-negative")
        if max_chunk_size <= min_chunk_size:
            raise Error(
                "parquet: cdc max_chunk_size must be greater than"
                " min_chunk_size"
            )
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.norm_level = norm_level
        var avg = (min_chunk_size + max_chunk_size) // 2
        var target = (avg - min_chunk_size) // NUM_GEARHASH_TABLES
        var tb = bit_width(UInt64(target))
        var mask_bits = 0 if tb == 0 else Int(tb) - 1
        var eff = mask_bits - norm_level
        if eff < 1 or eff > 63:
            raise Error(
                "parquet: cdc mask must be between 1 and 63 bits, got "
                + String(eff)
            )
        self.mask = UInt64.MAX << UInt64(64 - eff)


@fieldwise_init
struct Chunk(Copyable, Movable):
    """One page's worth of the column: where it starts in the levels, where it
    starts in the leaf values, and how many levels it spans.

    `value_offset` counts **leaf slots** -- it advances on
    `def >= slot_def`, including null elements inside a list -- which is what
    slices the values array. arrow-rs publishes a non-null count under this
    name; that is not this.
    """

    var level_offset: Int
    var value_offset: Int
    var num_levels: Int


struct ContentDefinedChunker(Movable):
    """The rolling state for one leaf column of one row group.

    The state is **never reset between pages** or between `chunks()` calls, and
    that continuity is what keeps a page boundary where it was after an edit
    earlier in the column. It **is** dropped at the end of a row group, because
    that is where Arrow C++ drops it: `content_defined_chunker_` is a member of
    `ColumnWriterImpl` (`column_writer.cc`) and `RowGroupSerializer` builds a
    fresh column writer per row group, so a row group always begins at
    `rolling_hash = 0`. Keeping one chunker per file instead is not a stronger
    guarantee, just a different chunking -- and one no other implementation
    would reproduce.

    The gearhash table is passed in on every call rather than stored: it is
    16 KiB, and one copy per file is right where one copy per column is not.
    """

    var min_chunk_size: Int
    var max_chunk_size: Int
    var mask: UInt64
    var max_def: Int
    var max_rep: Int
    var slot_def: Int

    var rolling_hash: UInt64
    var has_matched: Bool
    var nth_run: Int
    var chunk_size: Int

    def __init__(
        out self,
        options: ContentDefinedChunking,
        max_def: Int,
        max_rep: Int,
        slot_def: Int,
    ):
        self.min_chunk_size = options.min_chunk_size
        self.max_chunk_size = options.max_chunk_size
        self.mask = options.mask
        self.max_def = max_def
        self.max_rep = max_rep
        self.slot_def = slot_def
        self.rolling_hash = 0
        self.has_matched = False
        self.nth_run = 0
        self.chunk_size = 0

    def _roll_byte(mut self, table: List[UInt64], b: UInt8):
        """Feed one byte. The caller has already counted it toward
        `chunk_size` and checked the skip window."""
        self.rolling_hash = (self.rolling_hash << 1) + table[
            self.nth_run * 256 + Int(b)
        ]
        self.has_matched = self.has_matched or (
            (self.rolling_hash & self.mask) == 0
        )

    def _roll_scalar[T: DType](mut self, table: List[UInt64], v: Scalar[T]):
        """Feed a fixed-width value as its little-endian storage bytes.

        `as_bytes` reads the raw bit pattern for any `DType`, which is the
        same primitive `Plain.encode_primitive` and `ColumnWriter._hash_prim`
        already serialize a scalar with -- so floats need no special case and
        the 16- and 32-byte decimal widths come for free.
        """
        comptime W = size_of[Scalar[T]]()
        self.chunk_size += W
        if self.chunk_size < self.min_chunk_size:
            return
        var bytes = v.as_bytes[big_endian=False]()
        comptime for i in range(W):
            self._roll_byte(table, bytes[i])

    def _roll_bytes(mut self, table: List[UInt64], data: Span[UInt8, _]):
        """Feed a variable-length value (a string or binary element)."""
        self.chunk_size += len(data)
        if self.chunk_size < self.min_chunk_size:
            return
        for i in range(len(data)):
            self._roll_byte(table, data[i])

    def _roll_level(mut self, table: List[UInt64], level: Int32):
        """Feed a definition or repetition level as 2-byte little-endian.

        Marrow stores levels as `Int32`; the references use `int16_t`, and the
        hashed byte width changes every boundary downstream, so this narrows.
        Parquet caps levels at 32767 regardless.
        """
        self._roll_scalar[DType.int16](table, Int16(level))

    def _need_new_chunk(mut self) -> Bool:
        """Whether to cut here.

        Two conditions. A single gear hash gives geometrically distributed
        chunk sizes; requiring `NUM_GEARHASH_TABLES` consecutive matches, each
        against a different table, approximates a normal distribution by the
        central limit theorem. And a hard cut at `max_chunk_size` bounds the
        tail. Neither resets the rolling hash -- only the size counter -- which
        the references also do deliberately.
        """
        if self.has_matched:
            self.has_matched = False
            self.nth_run += 1
            if self.nth_run >= NUM_GEARHASH_TABLES:
                self.nth_run = 0
                self.chunk_size = 0
                return True
        if self.chunk_size >= self.max_chunk_size:
            self.chunk_size = 0
            return True
        return False

    def _calculate[
        R: def(mut ContentDefinedChunker, Int) raises -> None
    ](
        mut self,
        table: List[UInt64],
        defs: List[Int32],
        reps: List[Int32],
        num_levels: Int,
        var roll_value: R,
    ) raises -> List[Chunk]:
        """Walk the (def, rep, value) triplets and emit chunk boundaries.

        `roll_value` is called with the **leaf slot index**, not the level
        index and not a non-null count.

        The two paths are selected by `max_rep`, never by whether a level
        list happens to be non-empty: `ColumnWriter.write` synthesizes an
        all-zero `defs` array for required columns, and hashing it would feed
        bytes the reference never feeds -- so the flat path only rolls a def
        level at all when `max_def > 0`, a required column (`max_def == 0`)
        hashes values alone.
        """
        var chunks = List[Chunk]()
        var prev_offset = 0
        var prev_value_offset = 0

        if self.max_rep == 0:
            # Flat: one leaf slot per level. Required (`max_def == 0`) hashes
            # values only; nullable hashes the def level for every row and
            # the value only when present -- a loop-invariant condition, but
            # tested per level so both share the same boundary-cut-and-append
            # code below.
            for offset in range(num_levels):
                if self.max_def > 0:
                    var d = defs[offset]
                    self._roll_level(table, d)
                    if Int(d) == self.max_def:
                        roll_value(self, offset)
                else:
                    roll_value(self, offset)
                if self._need_new_chunk():
                    var levels_to_write = offset - prev_offset
                    if levels_to_write > 0:
                        # One leaf slot per level on this path, so
                        # `value_offset == level_offset` by construction.
                        chunks.append(
                            Chunk(prev_offset, prev_offset, levels_to_write)
                        )
                        prev_offset = offset
            prev_value_offset = prev_offset
        else:
            # Nested: leaf slots and levels diverge, and cuts land only on
            # record boundaries.
            var value_offset = 0
            for offset in range(num_levels):
                var d = defs[offset]
                var r = reps[offset]
                self._roll_level(table, d)
                self._roll_level(table, r)
                if Int(d) == self.max_def:
                    roll_value(self, value_offset)
                if Int(r) == 0 and self._need_new_chunk():
                    var levels_to_write = offset - prev_offset
                    if levels_to_write > 0:
                        chunks.append(
                            Chunk(
                                prev_offset, prev_value_offset, levels_to_write
                            )
                        )
                        prev_offset = offset
                        prev_value_offset = value_offset
                if Int(d) >= self.slot_def:
                    value_offset += 1

        if prev_offset < num_levels:
            chunks.append(
                Chunk(prev_offset, prev_value_offset, num_levels - prev_offset)
            )
        return chunks^

    def chunks(
        mut self,
        table: List[UInt64],
        values: DynArray,
        defs: List[Int32],
        reps: List[Int32],
    ) raises -> List[Chunk]:
        """Chunk one leaf column -- the single entry point `writer.mojo` calls,
        dispatching over every leaf type the writer emits.

        `values` is the leaf array, one slot per leaf level. `defs` / `reps`
        are the shredded levels; both are ignored when `max_def` / `max_rep`
        are zero, which is why the paths branch on those and not on
        emptiness -- see `_calculate`.
        """
        var num_levels = values.length() if (
            self.max_def == 0 and self.max_rep == 0
        ) else len(defs)
        ref vt = values.dtype()

        if vt.is_bool():
            var bools = values.as_bool().copy()

            def roll_bool(
                mut c: ContentDefinedChunker, i: Int
            ) raises {imm table, imm bools} -> None:
                c._roll_scalar[DType.uint8](
                    table, UInt8(1) if bools.values().test(i) else UInt8(0)
                )

            return self._calculate(table, defs, reps, num_levels, roll_bool)
        elif vt.is_binary_like():
            # Every binary-like leaf hashes its element bytes -- string,
            # binary and their large_ variants all share this path.
            def bytes_leaf[
                BT: dt.BinaryLikeType
            ](witness: BT) raises {
                mut self,
                imm table,
                imm values,
                imm defs,
                imm reps,
                imm num_levels,
            } -> List[Chunk]:
                return self._chunks_binary_like_typed(
                    table, values.as_binary_like[BT](), defs, reps, num_levels
                )

            return vt.dispatch_binarylike(bytes_leaf)
        elif vt.is_fixed_size_binary():
            var fsb = values.as_fixed_size_binary().copy()

            # `FixedSizeBinaryArray.__getitem__` builds a fresh `List[UInt8]`
            # one `unsafe_get` at a time and wraps it in a scalar -- wasted
            # allocation and a byte-at-a-time copy on this hot path, so this
            # reads straight out of the backing buffer instead.
            def roll_fixed_size_binary(
                mut c: ContentDefinedChunker, i: Int
            ) raises {imm table, imm fsb} -> None:
                var start = (fsb.offset + i) * fsb.byte_width
                c._roll_bytes(
                    table,
                    fsb.buffer.view[DType.uint8](
                        start, fsb.byte_width
                    ).as_span(),
                )

            return self._calculate(
                table, defs, reps, num_levels, roll_fixed_size_binary
            )
        elif vt.is_null():
            # No bytes to feed -- a null column chunks on `max_chunk_size`
            # alone (or on the levels, in the leveled paths).
            def roll_none(
                mut c: ContentDefinedChunker, i: Int
            ) raises {imm} -> None:
                pass

            return self._calculate(table, defs, reps, num_levels, roll_none)
        else:
            # One arm for every fixed-width family: numeric, temporal,
            # interval and decimal all satisfy `PrimitiveType`, so this single
            # `dispatch_primitive` call carries all of them -- the dispatch
            # `filter`/`take` narrowed per-family and silently lost decimal
            # and interval columns over.
            def primitive_leaf[
                T: dt.PrimitiveType
            ](witness: T) raises {
                mut self,
                imm table,
                imm values,
                imm defs,
                imm reps,
                imm num_levels,
            } -> List[Chunk]:
                return self._chunks_primitive_typed(
                    table, values.as_primitive[T](), defs, reps, num_levels
                )

            return vt.dispatch_primitive(primitive_leaf)

    def _chunks_binary_like_typed[
        BT: dt.BinaryLikeType
    ](
        mut self,
        table: List[UInt64],
        arr: BinaryLikeArray[BT],
        defs: List[Int32],
        reps: List[Int32],
        num_levels: Int,
    ) raises -> List[Chunk]:
        """Roll one binary-like leaf's element bytes through the rolling
        hash -- string, binary and their large_ variants all share this."""

        def roll(
            mut c: ContentDefinedChunker, i: Int
        ) raises {imm table, imm arr} -> None:
            c._roll_bytes(table, arr.unsafe_get(UInt(i)).as_bytes())

        return self._calculate(table, defs, reps, num_levels, roll)

    def _chunks_primitive_typed[
        T: dt.PrimitiveType
    ](
        mut self,
        table: List[UInt64],
        arr: PrimitiveArray[T],
        defs: List[Int32],
        reps: List[Int32],
        num_levels: Int,
    ) raises -> List[Chunk]:
        """Roll one fixed-width leaf's values through the rolling hash --
        numeric, temporal, interval and decimal all satisfy `PrimitiveType`
        and share this one path."""

        def roll(
            mut c: ContentDefinedChunker, i: Int
        ) raises {imm table, imm arr} -> None:
            c._roll_scalar[T.native](table, arr.values()[i])

        return self._calculate(table, defs, reps, num_levels, roll)
