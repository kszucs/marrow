"""Parquet value/level encodings.

The RLE / bit-packed *hybrid* encoding carries both the definition/repetition
levels and the dictionary index streams, so it is the workhorse here. PLAIN
fixed-width and BYTE_ARRAY decoding is done directly in `reader.mojo` (a straight
memcpy / length-prefixed walk), so this module only needs the hybrid codec plus
the bit-width helper.

Wire format (per the Parquet spec): a sequence of runs, each introduced by a
ULEB128 header. `header & 1` selects the run kind:
    - RLE run:        `header >> 1` = repeat count, followed by the value in
                      `ceil(bit_width/8)` little-endian bytes.
    - bit-packed run: `header >> 1` = number of 8-value groups; the values follow
                      LSB-first, `bit_width` bits each.
"""

from ..views import load_word_le


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
