"""Unit tests for content-defined chunking -- see `marrow/parquet/chunker.mojo`'s
module docstring."""

from std.testing import assert_true, assert_equal, assert_raises

from ..gearhash import gearhash_table, NUM_GEARHASH_TABLES
from ..chunker import ContentDefinedChunking, ContentDefinedChunker, Chunk
from ...arrays import DynArray, StructArray
from ... import builders as ar
from ... import dtypes as dt
from ...kernels.cast import cast


def test_gearhash_table_intact() raises:
    """The table is frozen with an order-sensitive FNV-1a digest: marrow's
    chunk boundaries deduplicate against Arrow C++ and arrow-rs only while all
    three use these exact values in this exact order. The digest and all spot
    checks are derived from `arrow/cpp/src/parquet/chunker_internal_generated.h`.

    FNV-1a catches reordering, transposition, and missing entries that XOR alone
    would not. Spot checks at byte 137 of each table (indices 137, 393, 649,
    905, 1161, 1417, 1673, 1929) provide per-table verification."""
    var t = gearhash_table()
    assert_equal(len(t), NUM_GEARHASH_TABLES * 256)

    # FNV-1a digest: order-sensitive hash over all 2048 entries
    var h: UInt64 = 1469598103934665603  # FNV offset basis
    for v in t:
        h = (h ^ v) * 1099511628211  # FNV prime, wrapping at 64 bits
    assert_equal(h, 0xCA04E5DCF0E98D6D)

    # Boundary spot checks: two outer edges + middle of last byte
    assert_equal(t[0], 0xF09F35A563783945)  # table 0, byte 0
    assert_equal(t[256], 0xECFCBA92FE5691A3)  # table 1, byte 0
    assert_equal(t[2047], 0x8DF9314914F0857E)  # table 7, byte 255

    # Per-table spot checks at byte 137 (middle, not edge)
    assert_equal(t[137], 0x775866CE7C83C762)  # table 0, byte 137
    assert_equal(t[393], 0xFC7DB8AA9ED67295)  # table 1, byte 137
    assert_equal(t[649], 0x7A87CD2A93276B54)  # table 2, byte 137
    assert_equal(t[905], 0x206F336FCCA11320)  # table 3, byte 137
    assert_equal(t[1161], 0x7232653DA72FC7F6)  # table 4, byte 137
    assert_equal(t[1417], 0x908A581D57113718)  # table 5, byte 137
    assert_equal(t[1673], 0x11CDAD9A76EE1DC4)  # table 6, byte 137
    assert_equal(t[1929], 0xBDB4A02A24D4DEE0)  # table 7, byte 137


def _mask(min_size: Int, max_size: Int, norm: Int) raises -> UInt64:
    return ContentDefinedChunking(min_size, max_size, norm).mask


def test_chunker_mask_vectors() raises:
    """Mask values transplanted from Arrow C++'s `RollingHashMaskCalculation`
    (`cpp/src/parquet/chunker_internal_test.cc`). These fix the chunk size
    distribution, so a change here is a change to every boundary marrow writes.
    """
    var lo = 256 * 1024
    var hi = 1024 * 1024
    assert_equal(_mask(lo, hi, 0), 0xFFFE000000000000)
    assert_equal(_mask(lo, hi, 1), 0xFFFC000000000000)
    assert_equal(_mask(lo, hi, 2), 0xFFF8000000000000)
    assert_equal(_mask(lo, hi, 3), 0xFFF0000000000000)
    assert_equal(_mask(lo, hi, -1), 0xFFFF000000000000)
    assert_equal(_mask(0, 32, 0), 0x8000000000000000)
    assert_equal(_mask(0, 64, 0), 0xC000000000000000)
    assert_equal(_mask(0, 16, -1), 0x8000000000000000)


def test_chunker_mask_rejects_bad_sizes() raises:
    """The effective mask width must land in [1, 63]; outside that the chunker
    would either never cut or cut on every byte."""
    with assert_raises():
        _ = _mask(0, 16, 0)  # 0 effective bits
    with assert_raises():
        _ = _mask(0, 32, 1)  # normalised down to 0
    with assert_raises():
        _ = _mask(128, 384, -60)  # normalised past 63
    with assert_raises():
        _ = _mask(10, 10, 0)  # max must exceed min
    with assert_raises():
        _ = _mask(-1, 10, 0)  # min must be non-negative


def _required_int64_chunks(
    values: List[Int64], opts: ContentDefinedChunking
) raises -> List[Chunk]:
    """Chunk a required (max_def == 0) int64 column -- the values-only path."""
    var table = gearhash_table()
    var c = ContentDefinedChunker(opts, max_def=0, max_rep=0, slot_def=0)
    return c.chunks(table, _int64_array(values), List[Int32](), List[Int32]())


def _tiny() raises -> ContentDefinedChunking:
    """A size envelope small enough that a few thousand values chunk several
    times, so the tests run in milliseconds rather than megabytes."""
    return ContentDefinedChunking(512, 4096, 0)


def _seq(n: Int) -> List[Int64]:
    """A deterministic, non-repeating sequence -- a constant column would hash
    the same byte forever and chunk only at `max_chunk_size`."""
    var out = List[Int64](capacity=n)
    var x: Int64 = 0
    for i in range(n):
        x = (x * 6364136223846793005 + 1442695040888963407) ^ Int64(i)
        out.append(x)
    return out^


def _assert_covers(chunks: List[Chunk], num_levels: Int) raises:
    """Every invariant C++'s `ValidateChunks` asserts: contiguous, monotonic,
    starting at zero, covering everything."""
    assert_true(len(chunks) > 0)
    assert_equal(chunks[0].level_offset, 0)
    assert_equal(chunks[0].value_offset, 0)
    var total = chunks[0].num_levels
    for i in range(1, len(chunks)):
        assert_true(chunks[i].num_levels > 0)
        assert_equal(
            chunks[i].level_offset,
            chunks[i - 1].level_offset + chunks[i - 1].num_levels,
        )
        assert_true(chunks[i].value_offset >= chunks[i - 1].value_offset)
        total += chunks[i].num_levels
    assert_equal(total, num_levels)
    assert_equal(
        chunks[len(chunks) - 1].level_offset
        + chunks[len(chunks) - 1].num_levels,
        num_levels,
    )


def test_chunker_flat_required_invariants() raises:
    var vals = _seq(20_000)
    var chunks = _required_int64_chunks(vals, _tiny())
    assert_true(
        len(chunks) > 3, "expected several chunks, got " + String(len(chunks))
    )
    _assert_covers(chunks, len(vals))


def test_chunker_is_deterministic() raises:
    var vals = _seq(20_000)
    var a = _required_int64_chunks(vals, _tiny())
    var b = _required_int64_chunks(vals, _tiny())
    assert_equal(len(a), len(b))
    for i in range(len(a)):
        assert_equal(a[i].level_offset, b[i].level_offset)
        assert_equal(a[i].num_levels, b[i].num_levels)


def test_chunker_resyncs_after_an_insertion() raises:
    """The property the whole feature exists for. Insert one value near the
    front; boundaries must re-synchronise rather than all shift by one."""
    var base = _seq(20_000)
    var edited = List[Int64](capacity=len(base) + 1)
    for i in range(len(base)):
        if i == 50:
            edited.append(Int64(-999_999))
        edited.append(base[i])

    var a = _required_int64_chunks(base, _tiny())
    var b = _required_int64_chunks(edited, _tiny())

    # Boundaries as absolute level offsets. Past the edit, `b`'s should be
    # `a`'s shifted by exactly one value, for the large majority of chunks.
    var shifted = 0
    var total = 0
    for i in range(1, len(a)):
        var want = a[i].level_offset + 1
        if a[i].level_offset <= 50:
            continue
        total += 1
        for j in range(1, len(b)):
            if b[j].level_offset == want:
                shifted += 1
                break
    assert_true(total > 2, "not enough boundaries after the edit to judge")
    assert_true(
        shifted * 2 > total,
        "boundaries did not resync: "
        + String(shifted)
        + " of "
        + String(total)
        + " realigned",
    )


def test_chunker_empty_column() raises:
    """C++ returns no chunks here and its own validation then asserts the list
    is non-empty -- a latent bug. Marrow returns the empty list and the writer
    is responsible for still emitting one page."""
    var chunks = _required_int64_chunks(List[Int64](), _tiny())
    assert_equal(len(chunks), 0)


def test_chunker_respects_max_chunk_size() raises:
    """A constant column never matches the mask, so only the hard cut fires.
    8 bytes per value, so a 4096-byte cap is 512 values -- except the very
    first chunk, which is one short. Both reference implementations record a
    cut as `[prev_offset, offset)`: the value whose bytes push `chunk_size`
    to the cap is excluded from the chunk it completed and becomes the first
    value of the next one, and `chunk_size` resets to 0 without crediting it.
    That is a one-time effect at `prev_offset == 0`; every later cut resets
    from a chunk that already "lost" that value, so it accumulates a full
    512 of its own before the next cut. See `cpp/src/parquet/chunker_internal.cc`
    `Impl::Calculate` and arrow-rs `column/chunker/cdc.rs` `calculate`."""
    var vals = List[Int64](capacity=5_000)
    for _ in range(5_000):
        vals.append(Int64(7))
    var chunks = _required_int64_chunks(vals, _tiny())
    _assert_covers(chunks, len(vals))
    assert_equal(chunks[0].num_levels, 511)
    for i in range(1, len(chunks) - 1):
        assert_equal(chunks[i].num_levels, 512)


def test_chunker_flat_required_no_zero_length_first_chunk() raises:
    """A hard cut that fires on the very first value must not emit a
    zero-length `Chunk`. Arrow C++ never emits one -- `AddDataPage()` in
    `column_writer.cc` is gated on `num_buffered_values_ > 0` -- so an empty
    first chunk here would break byte-identity with the reference (see
    `test_chunker_parity.mojo`). Every string below is far longer than
    `max_chunk_size`, so the hard cut fires while hashing the very first
    element, before any level has been "written" to a chunk."""
    var n = 200
    var vals = List[String](capacity=n)
    for i in range(n):
        vals.append(
            "this-value-is-much-longer-than-the-cdc-max-chunk-size-" + String(i)
        )
    var arr = _binary_like(vals, dt.string)
    var chunks = _chunks_of(arr, ContentDefinedChunking(0, 8, -1))
    _assert_covers(chunks, n)
    for c in chunks:
        assert_true(c.num_levels > 0, "a chunk had zero levels")


def test_chunker_flat_nullable_no_zero_length_first_chunk() raises:
    """The same hazard on the flat-*nullable* branch -- `chunker.mojo`'s other
    flat path, and the second place `_calculate` builds a `Chunk` unconditionally
    on a cut. A `max_chunk_size` of 2 is smaller than a single def level's own
    2 hashed bytes, so the hard cut fires at level 0 regardless of whether
    that level is null or present."""
    var n = 200
    var vals = _seq(n)
    for first_def in [Int32(0), Int32(1)]:
        var defs = List[Int32](capacity=n)
        defs.append(first_def)
        for i in range(1, n):
            defs.append(Int32(1) if i % 2 == 0 else Int32(0))
        var chunks = _chunks_with_levels(
            vals, defs, List[Int32](), 1, 0, 0, ContentDefinedChunking(0, 2, -1)
        )
        _assert_covers(chunks, n)
        for c in chunks:
            assert_true(c.num_levels > 0, "a chunk had zero levels")


def _chunks_with_levels(
    values: List[Int64],
    defs: List[Int32],
    reps: List[Int32],
    max_def: Int,
    max_rep: Int,
    slot_def: Int,
    opts: ContentDefinedChunking,
) raises -> List[Chunk]:
    var table = gearhash_table()
    var c = ContentDefinedChunker(opts, max_def, max_rep, slot_def)
    return c.chunks(table, _int64_array(values), defs, reps)


def test_chunker_required_hashes_no_levels() raises:
    """`max_def == 0` must hash values only. Marrow synthesizes an all-zero
    `defs` array for required columns that C++ never hashes, so feeding it
    would change every boundary. Same values as a nullable all-present column
    must therefore chunk *differently*."""
    var vals = _seq(20_000)
    var zero_defs = List[Int32](capacity=len(vals))
    var ones = List[Int32](capacity=len(vals))
    for _ in range(len(vals)):
        zero_defs.append(Int32(0))
        ones.append(Int32(1))

    var required = _chunks_with_levels(
        vals, zero_defs, List[Int32](), 0, 0, 0, _tiny()
    )
    var optional = _chunks_with_levels(
        vals, ones, List[Int32](), 1, 0, 0, _tiny()
    )
    _assert_covers(required, len(vals))
    _assert_covers(optional, len(vals))
    var same = len(required) == len(optional)
    if same:
        for i in range(len(required)):
            if required[i].level_offset != optional[i].level_offset:
                same = False
                break
    assert_true(
        not same,
        (
            "required and optional chunked identically -- levels are being"
            " hashed for a required column, or not hashed for an optional one"
        ),
    )


def test_chunker_nested_cuts_only_at_record_boundaries() raises:
    """A record must never split across pages, so every boundary past the
    first must land on a `rep_level == 0` level."""
    var n = 30_000
    var vals = _seq(n)
    var defs = List[Int32](capacity=n)
    var reps = List[Int32](capacity=n)
    for i in range(n):
        defs.append(Int32(3))  # present leaf under a present list
        reps.append(Int32(0) if i % 3 == 0 else Int32(1))
    var chunks = _chunks_with_levels(vals, defs, reps, 3, 1, 2, _tiny())
    _assert_covers(chunks, n)
    assert_true(len(chunks) > 3)
    for i in range(1, len(chunks)):
        assert_equal(
            reps[chunks[i].level_offset],
            Int32(0),
            "chunk " + String(i) + " split a record",
        )


def test_chunker_nested_value_offset_counts_leaf_slots() raises:
    """The worked example from arrow-rs's `cdc.rs`: `List<Int32?>` with
    `slot_def == 2` and `max_def == 3`. A null *element* inside a present list
    still occupies a leaf slot, so `value_offset` counts it; a non-null count
    would index the wrong slot."""
    var n = 30_000
    var vals = _seq(n)
    var defs = List[Int32](capacity=n)
    var reps = List[Int32](capacity=n)
    for i in range(n):
        # Every third leaf is a null element (def 2), still a leaf slot.
        defs.append(Int32(2) if i % 3 == 1 else Int32(3))
        reps.append(Int32(0) if i % 3 == 0 else Int32(1))
    var chunks = _chunks_with_levels(vals, defs, reps, 3, 1, 2, _tiny())
    _assert_covers(chunks, n)
    # Every level here has def >= slot_def (2), so leaf slots and levels are
    # one-to-one and the two cursors must agree.
    for i in range(len(chunks)):
        assert_equal(chunks[i].value_offset, chunks[i].level_offset)


def _chunks_of(
    values: DynArray, opts: ContentDefinedChunking
) raises -> List[Chunk]:
    var table = gearhash_table()
    var c = ContentDefinedChunker(opts, max_def=0, max_rep=0, slot_def=0)
    return c.chunks(table, values, List[Int32](), List[Int32]())


def _int64_array(vals: List[Int64]) raises -> DynArray:
    """An int64 column from a pre-built `List[Int64]` -- the free `array()`
    factory only infers its type parameter from a list *literal* (whose
    elements are polymorphic), not from an already-typed `List[Int64]`, so
    large pre-built sequences (`_seq`) go through the builder directly."""
    var b = ar.Int64Builder(capacity=len(vals))
    for v in vals:
        b.append(v)
    return b.finish()


def _float64_array(vals: List[Float64]) raises -> DynArray:
    var b = ar.Float64Builder(capacity=len(vals))
    for v in vals:
        b.append(v)
    return b.finish()


def _as(vals: List[Int64], to: dt.DynType) raises -> DynArray:
    """Cast an int64 column to `to` -- the idiom `kernels/tests/test_cast.mojo`
    uses to build temporal and decimal columns, not a reinterpret of the
    int64 bytes."""
    return cast(_int64_array(vals), to)


def _bools(vals: List[Bool]) raises -> DynArray:
    var b = ar.BoolBuilder(capacity=len(vals))
    for v in vals:
        b.append(v)
    return b.finish()


def _binary_like(vals: List[String], to: dt.DynType) raises -> DynArray:
    """A binary-like column of dtype `to`. String, large_string, binary and
    large_binary all share `BinaryLikeBuilder.append(String)`; only the
    accessor that reaches the right variant differs."""
    var b = ar.DynBuilder(to, capacity=len(vals))
    if to.is_string():
        for ref v in vals:
            b.as_string().append(v)
    elif to.is_large_string():
        for ref v in vals:
            b.as_large_string().append(v)
    elif to.is_binary():
        for ref v in vals:
            b.as_binary().append(v)
    else:
        for ref v in vals:
            b.as_large_binary().append(v)
    return b.finish()


def _fixed_size_binary(vals: List[Int64], width: Int) raises -> DynArray:
    """A fixed-size-binary column: each element is the low `width` bytes of
    `vals[i]`, little-endian -- non-repeating, like `_seq`, so it does not
    chunk only at `max_chunk_size`."""
    var b = ar.FixedSizeBinaryBuilder(width, capacity=len(vals))
    for v in vals:
        var value_bytes = List[UInt8](capacity=width)
        for k in range(width):
            value_bytes.append(((v >> Int64(k * 8)) & 0xFF).cast[DType.uint8]())
        b.append(Span(value_bytes))
    return b.finish()


def test_chunker_dispatches_every_leaf_type() raises:
    """Each supported leaf type must chunk without raising and must cover its
    input. A type that reaches the fallback raises, which is the point."""
    var n = 8_000
    var i64 = List[Int64](capacity=n)
    var f64 = List[Float64](capacity=n)
    var s = List[String](capacity=n)
    var b = List[Bool](capacity=n)
    var seq = _seq(n)
    for i in range(n):
        i64.append(seq[i])
        f64.append(Float64(seq[i]) * 0.5)
        s.append(String(seq[i]))
        b.append(seq[i] % 3 == 0)

    # Temporal, decimal and interval are the families a per-family dispatch
    # ladder silently drops -- `filter`/`take` shipped exactly that defect.
    # They all satisfy `PrimitiveType`, so a single `dispatch_primitive` arm
    # has to carry them, and this list is what proves it did.
    var cases: List[DynArray] = [
        _int64_array(i64),
        _float64_array(f64),
        _binary_like(s, dt.string),
        _bools(b),
        _as(i64, dt.timestamp(dt.microsecond).to_dyn()),  # temporal
        _as(i64, dt.duration(dt.microsecond).to_dyn()),  # temporal
        _as(i64, dt.decimal128(20, 4).to_dyn()),  # 16-byte decimal
        _as(i64, dt.decimal256(40, 4).to_dyn()),  # 32-byte decimal
        _binary_like(s, dt.large_string),
        _binary_like(s, dt.binary),
        _binary_like(s, dt.large_binary),
        _fixed_size_binary(i64, 4),  # the fixed_size_binary arm
        ar.nulls(n, dt.null),  # the null arm
    ]
    for ref leaf in cases:
        var chunks = _chunks_of(leaf, _tiny())
        _assert_covers(chunks, leaf.length())


def test_chunker_rejects_an_unsupported_type() raises:
    """A leaf type with no byte representation here must raise rather than
    silently hash nothing and chunk only at max_chunk_size."""
    var nested: DynArray = _int64_array(List[Int64]())
    var st = StructArray(
        dtype=dt.struct_(dt.Field("a", nested.dtype().copy())),
        length=0,
        nulls=0,
        offset=0,
        bitmap=None,
        children=[nested^],
    )
    var table = gearhash_table()
    var c = ContentDefinedChunker(_tiny(), max_def=0, max_rep=0, slot_def=0)
    # A struct array has no leaf byte stream; the writer never hands one here.
    with assert_raises():
        _ = c.chunks(table, st^, List[Int32](), List[Int32]())
