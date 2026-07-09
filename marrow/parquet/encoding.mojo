"""Parquet value and level encodings.

`Encoding` is a Parquet `Encoding` enum value that also *owns* the decode of a
data page's present values: `decode_primitive` / `decode_bytes` / `decode_bool`
each dispatch on the encoding and turn a value byte-span into decoded values, so
the flat and nested reader paths share one decoder per physical layout. The
free-standing primitives below (RLE/bit-packed hybrid, DELTA_BINARY_PACKED,
byte helpers) do the actual bit twiddling; the decoders compose them.

RLE / bit-packed *hybrid* wire format (per the Parquet spec): a sequence of
runs, each introduced by a ULEB128 header. `header & 1` selects the run kind:
    - RLE run:        `header >> 1` = repeat count, followed by the value in
                      `ceil(bit_width/8)` little-endian bytes.
    - bit-packed run: `header >> 1` = number of 8-value groups; the values follow
                      LSB-first, `bit_width` bits each.
"""

from std.sys import size_of

from ..views import load_word_le


# ---------------------------------------------------------------------------
# Little-endian byte primitives (shared by the encoders and the page reader)
# ---------------------------------------------------------------------------


def read_u32le(body: Span[UInt8, _], off: Int) -> Int:
    return (
        Int(body[off])
        | (Int(body[off + 1]) << 8)
        | (Int(body[off + 2]) << 16)
        | (Int(body[off + 3]) << 24)
    )


def read_fixed_le[dt: DType](body: Span[UInt8, _], off: Int) -> Scalar[dt]:
    comptime W = size_of[Scalar[dt]]()
    var arr = InlineArray[UInt8, W](fill=0)
    for i in range(W):
        arr[i] = body[off + i]
    return SIMD[dt, 1].from_bytes[big_endian=False](arr)


# ---------------------------------------------------------------------------
# Encoding — a Parquet Encoding value plus its present-value decoders
# ---------------------------------------------------------------------------


struct Encoding(Equatable, ImplicitlyCopyable, Movable):
    """A Parquet `Encoding` enum value and the decode of a data page's *present*
    values under it. The decoders append `num_present` values from a value
    byte-span (nulls are placed later by the caller from the definition levels),
    so the same logic serves the flat and nested reader paths.
    """

    var code: Int

    comptime PLAIN = Self(0)
    comptime PLAIN_DICTIONARY = Self(2)
    comptime RLE = Self(3)
    comptime BIT_PACKED = Self(4)
    comptime DELTA_BINARY_PACKED = Self(5)
    comptime DELTA_LENGTH_BYTE_ARRAY = Self(6)
    comptime DELTA_BYTE_ARRAY = Self(7)
    comptime RLE_DICTIONARY = Self(8)
    comptime BYTE_STREAM_SPLIT = Self(9)

    def __init__(out self, code: Int):
        self.code = code

    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        return self.code != other.code

    def is_plain(self) -> Bool:
        return self == Self.PLAIN

    def is_dictionary(self) -> Bool:
        return self == Self.RLE_DICTIONARY or self == Self.PLAIN_DICTIONARY

    def decode_primitive[
        store: DType, phys: DType
    ](
        self,
        values: Span[UInt8, _],
        num_present: Int,
        dict: List[Scalar[store]],
        mut out: List[Scalar[store]],
    ) raises:
        """Append the present fixed-width values (widened `phys` -> `store`)."""
        comptime PW = size_of[Scalar[phys]]()
        var np = num_present
        if self.is_plain():
            for i in range(np):
                out.append(read_fixed_le[phys](values, i * PW).cast[store]())
        elif self.is_dictionary():
            var base = len(out)
            out.resize(unsafe_uninit_length=base + np)
            rle_gather[store](
                values[1:],
                Int(values[0]),
                np,
                dict.unsafe_ptr(),
                out.unsafe_ptr() + base,
            )
        elif self == Self.DELTA_BINARY_PACKED:
            var decoded = delta_binary_packed_decode(values, np)
            for i in range(np):
                out.append(decoded[i].cast[store]())
        elif self == Self.BYTE_STREAM_SPLIT:
            # byte k of value i lives at values[k*np + i]
            for i in range(np):
                var raw = InlineArray[UInt8, PW](fill=0)

                comptime for k in range(PW):
                    raw[k] = values[k * np + i]
                out.append(
                    SIMD[phys, 1]
                    .from_bytes[big_endian=False](raw)
                    .cast[store]()
                )
        else:
            raise Error(
                "parquet: unsupported data page encoding " + String(self.code)
            )

    def decode_bytes(
        self,
        values: Span[UInt8, _],
        num_present: Int,
        dict_body: List[UInt8],
        dict_off: List[Int],
        dict_len: List[Int],
    ) raises -> List[List[UInt8]]:
        """Return the present variable-length byte values."""
        var np = num_present
        var out = List[List[UInt8]]()
        if self.is_plain():
            var vi = 0
            for _ in range(np):
                var n = read_u32le(values, vi)
                vi += 4
                var v = List[UInt8]()
                v.extend(values[vi : vi + n])
                out.append(v^)
                vi += n
        elif self.is_dictionary():
            var indices = rle_decode(values[1:], Int(values[0]), np)
            for i in range(np):
                var idx = Int(indices[i])
                var start = dict_off[idx]
                var v = List[UInt8]()
                v.extend(Span(dict_body)[start : start + dict_len[idx]])
                out.append(v^)
        elif self == Self.DELTA_LENGTH_BYTE_ARRAY:
            var lengths = List[Int64]()
            var pos = delta_decode(values, 0, np, lengths)
            for i in range(np):
                var n = Int(lengths[i])
                var v = List[UInt8]()
                v.extend(values[pos : pos + n])
                out.append(v^)
                pos += n
        elif self == Self.DELTA_BYTE_ARRAY:
            var prefixes = List[Int64]()
            var pos = delta_decode(values, 0, np, prefixes)
            var suffix_lens = List[Int64]()
            pos = delta_decode(values, pos, np, suffix_lens)
            var prev = List[UInt8]()
            for i in range(np):
                var v = List[UInt8]()
                v.extend(Span(prev)[0 : Int(prefixes[i])])
                var sl = Int(suffix_lens[i])
                v.extend(values[pos : pos + sl])
                pos += sl
                out.append(v.copy())
                prev = v^
        else:
            raise Error(
                "parquet: unsupported byte-array encoding " + String(self.code)
            )
        return out^

    def decode_bool(
        self, values: Span[UInt8, _], num_present: Int
    ) raises -> List[Bool]:
        """Return the present PLAIN bit-packed booleans."""
        if not self.is_plain():
            raise Error("parquet: non-plain bool encoding not supported")
        var out = List[Bool]()
        for i in range(num_present):
            var byte = values[i >> 3]
            out.append(((byte >> UInt8(i & 7)) & 1) == 1)
        return out^

    @staticmethod
    def decode_dict_primitive[
        store: DType, phys: DType
    ](
        body: Span[UInt8, _], num_values: Int, mut dict: List[Scalar[store]]
    ) raises:
        """Decode a primitive dictionary page (PLAIN fixed-width) into `dict`.
        """
        comptime PW = size_of[Scalar[phys]]()
        for i in range(num_values):
            dict.append(read_fixed_le[phys](body, i * PW).cast[store]())

    @staticmethod
    def decode_dict_bytes(
        body: Span[UInt8, _],
        num_values: Int,
        mut dict_body: List[UInt8],
        mut dict_off: List[Int],
        mut dict_len: List[Int],
    ) raises:
        """Decode a byte-array dictionary page (length-prefixed values)."""
        dict_body.clear()
        dict_body.extend(body)
        var span = Span(dict_body)
        var off = 0
        for _ in range(num_values):
            var n = read_u32le(span, off)
            off += 4
            dict_off.append(off)
            dict_len.append(n)
            off += n


def bit_width(max_value: Int) -> Int:
    """Number of bits needed to represent values in `[0, max_value]`."""
    var w = 0
    var v = max_value
    while v > 0:
        w += 1
        v >>= 1
    return w


def _read_varint(data: Span[UInt8, _], pos: Int) raises -> Tuple[UInt64, Int]:
    var result: UInt64 = 0
    var shift: Int = 0
    var p = pos
    while True:
        if p >= len(data):
            raise Error("rle: varint out of bounds")
        var b = data[p]
        p += 1
        result |= UInt64(b & 0x7F) << UInt64(shift)
        if b & 0x80 == 0:
            break
        shift += 7
    return (result, p)


@always_inline
def _unpack8(
    data: Span[UInt8, _],
    byte_base: Int,
    base_bit: Int,
    width: Int,
    maskv: SIMD[DType.uint64, 8],
) -> SIMD[DType.uint64, 8]:
    """Unpack 8 bit-packed values in one shot: one unaligned 64-bit load per
    lane, then a *vector* shift + mask over all 8 lanes. ~3x faster than the
    scalar word-at-a-time loop.

    Requires `width <= 32` so the top read bit (`7 + width`) stays inside the
    64-bit word and a single load per lane covers every value — always true for
    dictionary indices (`bit_width(dict_size)`) and definition/repetition
    levels."""
    debug_assert(
        width <= 32, "_unpack8 requires width <= 32 (one 64-bit load per lane)"
    )
    var words = SIMD[DType.uint64, 8](0)
    var shifts = SIMD[DType.uint64, 8](0)
    comptime for j in range(8):
        var ab = base_bit + j * width
        words[j] = load_word_le(data, byte_base + (ab >> 3))
        shifts[j] = UInt64(ab & 7)
    return (words >> shifts) & maskv


def _read_bits(data: Span[UInt8, _], bit_offset: Int, nbits: Int) -> UInt64:
    """Read `nbits` starting at absolute `bit_offset`, least-significant first.
    """
    var result: UInt64 = 0
    for i in range(nbits):
        var abs_bit = bit_offset + i
        var byte_idx = abs_bit >> 3
        var bit_idx = abs_bit & 7
        var bit = (UInt64(data[byte_idx]) >> UInt64(bit_idx)) & 1
        result |= bit << UInt64(i)
    return result


def rle_decode(
    data: Span[UInt8, _], width: Int, count: Int
) raises -> List[Int32]:
    """Decode `count` values from an RLE/bit-packed hybrid stream."""
    var out = List[Int32]()
    if count == 0:
        return out^
    out.reserve(count)
    if width == 0:
        for _ in range(count):
            out.append(0)
        return out^

    var byte_width = (width + 7) // 8
    var pos = 0
    while len(out) < count:
        var header: UInt64
        header, pos = _read_varint(data, pos)
        if (header & 1) == 1:
            # bit-packed run — SIMD-unpack 8 values at a time (width <= 32).
            var num_groups = Int(header >> 1)
            var num_vals = num_groups * 8
            var maskv = SIMD[DType.uint64, 8]((UInt64(1) << UInt64(width)) - 1)
            var g = 0
            while g < num_vals:
                var v = _unpack8(data, pos, g * width, width, maskv)
                if len(out) + 8 <= count:
                    # comptime-unrolled extraction: lane index must be a
                    # compile-time constant or the SIMD lane-select is runtime.
                    comptime for j in range(8):
                        out.append(Int32(v[j]))
                else:
                    var take = count - len(out)
                    for j in range(take):
                        out.append(Int32(v[j]))
                g += 8
            pos += num_groups * width
        else:
            # RLE run
            var run_len = Int(header >> 1)
            var val: Int32 = 0
            for b in range(byte_width):
                val |= Int32(Int(data[pos + b]) << (8 * b))
            pos += byte_width
            for _ in range(run_len):
                if len(out) < count:
                    out.append(val)
    return out^


def rle_gather[
    dt: DType, do: Origin[mut=True]
](
    data: Span[UInt8, _],
    width: Int,
    count: Int,
    dict: UnsafePointer[Scalar[dt], _],
    dest: UnsafePointer[Scalar[dt], do],
) raises:
    """Decode `count` RLE/bit-packed dictionary indices and write `dict[idx]`
    straight to `out` — fuses index decode and gather (no index buffer)."""
    if count == 0:
        return
    if width == 0:
        for i in range(count):
            dest[i] = dict[0]
        return
    var byte_width = (width + 7) // 8
    var pos = 0
    var produced = 0
    while produced < count:
        var header: UInt64
        header, pos = _read_varint(data, pos)
        if (header & 1) == 1:
            # SIMD-unpack 8 indices at a time (width <= 32 for dict indices),
            # then gather dict values — the gather stays scalar (the dictionary
            # is small and cache-resident, so it is effectively free).
            var num_groups = Int(header >> 1)
            var num_vals = num_groups * 8
            var maskv = SIMD[DType.uint64, 8]((UInt64(1) << UInt64(width)) - 1)
            var g = 0
            while g < num_vals:
                var idxv = _unpack8(data, pos, g * width, width, maskv)
                if produced + 8 <= count:
                    # comptime-unrolled extraction: lane index must be a
                    # compile-time constant or the SIMD lane-select is runtime.
                    comptime for j in range(8):
                        dest[produced + j] = dict[Int(idxv[j])]
                    produced += 8
                else:
                    var take = count - produced
                    for j in range(take):
                        dest[produced] = dict[Int(idxv[j])]
                        produced += 1
                g += 8
            pos += num_groups * width
        else:
            var run_len = Int(header >> 1)
            var val: Int32 = 0
            for b in range(byte_width):
                val |= Int32(Int(data[pos + b]) << (8 * b))
            pos += byte_width
            var idx = Int(val)
            var take = min(run_len, count - produced)
            for _ in range(take):
                dest[produced] = dict[idx]
                produced += 1


@always_inline
def _zigzag(u: UInt64) -> Int64:
    """ULEB128 zigzag -> signed."""
    return Int64(u >> 1) ^ -Int64(u & 1)


def delta_decode(
    data: Span[UInt8, _], start: Int, count: Int, mut out: List[Int64]
) raises -> Int:
    """Decode `count` DELTA_BINARY_PACKED integers starting at byte `start`,
    appending to `out`; return the byte position just past the stream (so
    callers like DELTA_LENGTH_BYTE_ARRAY can find the data that follows).

    Layout (Parquet spec): a header of `block_size`, `miniblocks_per_block`,
    `total_value_count`, and the zigzag `first_value`; then blocks, each a
    zigzag `min_delta`, one bit-width byte per miniblock, and the bit-packed
    miniblock deltas. Each value is `prev + min_delta + unpacked_delta`.
    """
    if count == 0:
        return start

    var pos = start
    var block_size: UInt64
    block_size, pos = _read_varint(data, pos)
    var miniblocks: UInt64
    miniblocks, pos = _read_varint(data, pos)
    var _total: UInt64
    _total, pos = _read_varint(data, pos)
    var first_z: UInt64
    first_z, pos = _read_varint(data, pos)

    var num_miniblocks = Int(miniblocks)
    var vals_per_mb = Int(block_size) // num_miniblocks

    var value = _zigzag(first_z)
    out.append(value)
    var produced = 1

    while produced < count:
        var min_delta_z: UInt64
        min_delta_z, pos = _read_varint(data, pos)
        var min_delta = _zigzag(min_delta_z)
        var widths_at = pos
        pos += num_miniblocks  # one bit-width byte per miniblock
        for mb in range(num_miniblocks):
            var w = Int(data[widths_at + mb])
            var base_bit = pos * 8
            pos += (vals_per_mb * w) // 8  # miniblock is full even if padded
            for j in range(vals_per_mb):
                var delta: UInt64 = 0
                if w > 0:
                    delta = _read_bits(data, base_bit + j * w, w)
                value += min_delta + Int64(delta)
                out.append(value)
                produced += 1
                if produced == count:
                    return pos
    return pos


def delta_binary_packed_decode(
    data: Span[UInt8, _], count: Int
) raises -> List[Int64]:
    """Decode `count` DELTA_BINARY_PACKED integers (whole value stream)."""
    var out = List[Int64]()
    out.reserve(count)
    _ = delta_decode(data, 0, count, out)
    return out^


def rle_count_matches(
    data: Span[UInt8, _], width: Int, count: Int, target: Int32
) raises -> Int:
    """Count how many of the first `count` values equal `target`, without
    materializing them. An all-equal column is a single run, so the common
    no-null case (every definition level == max_def) is O(1)."""
    if width == 0:
        return count if target == 0 else 0
    var byte_width = (width + 7) // 8
    var pos = 0
    var produced = 0
    var matches = 0
    while produced < count:
        var header: UInt64
        header, pos = _read_varint(data, pos)
        if (header & 1) == 1:
            var num_groups = Int(header >> 1)
            var num_vals = num_groups * 8
            var base_bit = pos * 8
            for k in range(num_vals):
                if produced < count:
                    var v = Int32(_read_bits(data, base_bit + k * width, width))
                    if v == target:
                        matches += 1
                    produced += 1
            pos += num_groups * width
        else:
            var run_len = Int(header >> 1)
            var val: Int32 = 0
            for b in range(byte_width):
                val |= Int32(Int(data[pos + b]) << (8 * b))
            pos += byte_width
            var take = min(run_len, count - produced)
            if val == target:
                matches += take
            produced += take
    return matches


def _write_varint(mut out: List[UInt8], var v: UInt64):
    while True:
        var b = UInt8(v & 0x7F)
        v >>= 7
        if v != 0:
            out.append(b | 0x80)
        else:
            out.append(b)
            break


def rle_encode(values: List[Int32], width: Int) -> List[UInt8]:
    """Encode `values` as an all-RLE-run hybrid stream (runs of equal values).

    Levels are highly repetitive (often all-1s), so run-length runs compress
    them well and keep the encoder trivial.
    """
    var out = List[UInt8]()
    if width == 0:
        return out^
    var byte_width = (width + 7) // 8
    var i = 0
    var n = len(values)
    while i < n:
        var v = values[i]
        var j = i + 1
        while j < n and values[j] == v:
            j += 1
        var run_len = j - i
        _write_varint(out, UInt64(run_len) << 1)  # RLE run (low bit 0)
        for b in range(byte_width):
            out.append(UInt8((Int(v) >> (8 * b)) & 0xFF))
        i = j
    return out^
