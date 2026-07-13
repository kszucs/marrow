"""Parquet codecs — one small type per coding scheme.

Each encoding is its own stateless codec (`encode`/`decode` static methods);
`Encoding` is the enum that dispatches a data page's *present* values to the
right one, so the flat and nested reader paths share one decoder per layout:

- `LittleEndian` — the byte/bit/varint reads and writes the codecs are built on.
- `Rle` — the RLE / bit-packed *hybrid* codec carrying definition/repetition
  levels and dictionary indices (the workhorse; SIMD bit-unpack).
- `DeltaBinaryPacked` — the block/miniblock zigzag-delta integer codec.
- `Plain` · `Dictionary` · `ByteStreamSplit` · `DeltaLengthByteArray` ·
  `DeltaByteArray` — the data-page value codecs `Encoding` dispatches to.
- `Compression` — the page compression codec (dispatches onto `CompressionLibs`
  in `utils.mojo`).

RLE / bit-packed hybrid wire format (per the Parquet spec): a sequence of runs,
each introduced by a ULEB128 header. `header & 1` selects the run kind:
    - RLE run:        `header >> 1` = repeat count, followed by the value in
                      `ceil(bit_width/8)` little-endian bytes.
    - bit-packed run: `header >> 1` = number of 8-value groups; the values follow
                      LSB-first, `bit_width` bits each.
"""

from std.sys import size_of
from std.memory import memcpy

from ..arrays import (
    AnyArray,
    PrimitiveArray,
    StringArray,
    BinaryLikeArray,
    BoolArray,
    FixedSizeBinaryArray,
)
from .. import dtypes as dt
from ..utils import LittleEndian
from ..views import load_word_le
from .utils import CompressionLibs


struct Zigzag:
    """Signed <-> unsigned mapping so small-magnitude signed integers stay small
    as varints — shared by the delta codecs and the Thrift Compact Protocol.
    Stateless; a namespace of static methods."""

    @staticmethod
    @always_inline
    def encode(v: Int64) -> UInt64:
        return UInt64((v << 1) ^ (v >> 63))

    @staticmethod
    @always_inline
    def decode(u: UInt64) -> Int64:
        return Int64(u >> 1) ^ -Int64(u & 1)


struct Rle:
    """The RLE / bit-packed hybrid codec — definition/repetition levels and
    dictionary indices. Stateless static methods; the bit-packed runs are
    SIMD-unpacked eight values at a time."""

    @staticmethod
    def bit_width(max_value: Int) -> Int:
        """Number of bits needed to represent values in `[0, max_value]`."""
        var w = 0
        var v = max_value
        while v > 0:
            w += 1
            v >>= 1
        return w

    @staticmethod
    @always_inline
    def _run_value(data: Span[UInt8, _], pos: Int, byte_width: Int) -> Int32:
        """The little-endian value introducing an RLE run: `byte_width` bytes at
        `pos`. The caller advances its cursor by `byte_width`."""
        var val: Int32 = 0
        for b in range(byte_width):
            val |= Int32(Int(data[pos + b]) << (8 * b))
        return val

    @staticmethod
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
        64-bit word and a single load per lane covers every value — always true
        for dictionary indices (`bit_width(dict_size)`) and levels."""
        debug_assert(
            width <= 32,
            "Rle._unpack8 requires width <= 32 (one 64-bit load per lane)",
        )
        var words = SIMD[DType.uint64, 8](0)
        var shifts = SIMD[DType.uint64, 8](0)
        comptime for j in range(8):
            var ab = base_bit + j * width
            words[j] = load_word_le(data, byte_base + (ab >> 3))
            shifts[j] = UInt64(ab & 7)
        return (words >> shifts) & maskv

    @staticmethod
    def decode(
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
            header, pos = LittleEndian.varint(data, pos)
            if (header & 1) == 1:
                # bit-packed run — SIMD-unpack 8 values at a time (width <= 32).
                var num_groups = Int(header >> 1)
                var num_vals = num_groups * 8
                var maskv = SIMD[DType.uint64, 8](
                    (UInt64(1) << UInt64(width)) - 1
                )
                var g = 0
                while g < num_vals:
                    var v = Self._unpack8(data, pos, g * width, width, maskv)
                    if len(out) + 8 <= count:
                        # comptime-unrolled extraction: lane index must be a
                        # compile-time constant or the SIMD lane-select is
                        # runtime.
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
                var val = Self._run_value(data, pos, byte_width)
                pos += byte_width
                for _ in range(run_len):
                    if len(out) < count:
                        out.append(val)
        return out^

    @staticmethod
    def gather[
        dt: DType, do: Origin[mut=True]
    ](
        data: Span[UInt8, _],
        width: Int,
        count: Int,
        dict: Span[Scalar[dt], _],
        dest: Span[Scalar[dt], do],
        dest_offset: Int = 0,
    ) raises:
        """Decode `count` RLE/bit-packed dictionary indices and write `dict[idx]`
        straight into `dest` at `dest_offset` — fuses index decode and gather (no
        index buffer). Callers pass `Span`s (a `List` or a `BufferView.as_span()`)
        so no pointer crosses the boundary; the fused gather loop then works
        through raw pointers, since `Span` subscripting does not lower to a plain
        store on this path (a ~3x slowdown measured)."""
        if count == 0:
            return
        var dp = dest.unsafe_ptr() + dest_offset
        var kp = dict.unsafe_ptr()
        if width == 0:
            for i in range(count):
                dp[i] = kp[0]
            return
        var byte_width = (width + 7) // 8
        var pos = 0
        var produced = 0
        while produced < count:
            var header: UInt64
            header, pos = LittleEndian.varint(data, pos)
            if (header & 1) == 1:
                # SIMD-unpack 8 indices at a time (width <= 32 for dict indices),
                # then gather dict values — the gather stays scalar (the
                # dictionary is small and cache-resident, so it is ~free).
                var num_groups = Int(header >> 1)
                var num_vals = num_groups * 8
                var maskv = SIMD[DType.uint64, 8](
                    (UInt64(1) << UInt64(width)) - 1
                )
                var g = 0
                while g < num_vals:
                    var idxv = Self._unpack8(data, pos, g * width, width, maskv)
                    if produced + 8 <= count:
                        # comptime-unrolled extraction: lane index must be a
                        # compile-time constant or the SIMD lane-select is
                        # runtime.
                        comptime for j in range(8):
                            dp[produced + j] = kp[Int(idxv[j])]
                        produced += 8
                    else:
                        var take = count - produced
                        for j in range(take):
                            dp[produced] = kp[Int(idxv[j])]
                            produced += 1
                    g += 8
                pos += num_groups * width
            else:
                var run_len = Int(header >> 1)
                var val = Self._run_value(data, pos, byte_width)
                pos += byte_width
                var idx = Int(val)
                var take = min(run_len, count - produced)
                for _ in range(take):
                    dp[produced] = kp[idx]
                    produced += 1

    @staticmethod
    def count_matches(
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
            header, pos = LittleEndian.varint(data, pos)
            if (header & 1) == 1:
                var num_groups = Int(header >> 1)
                var num_vals = num_groups * 8
                var base_bit = pos * 8
                for k in range(num_vals):
                    if produced < count:
                        var v = Int32(
                            LittleEndian.bits(data, base_bit + k * width, width)
                        )
                        if v == target:
                            matches += 1
                        produced += 1
                pos += num_groups * width
            else:
                var run_len = Int(header >> 1)
                var val = Self._run_value(data, pos, byte_width)
                pos += byte_width
                var take = min(run_len, count - produced)
                if val == target:
                    matches += take
                produced += take
        return matches

    @staticmethod
    def encode_bitpacked(values: List[Int32], width: Int) -> List[UInt8]:
        """Encode all of `values` as a single bit-packed hybrid run (groups of 8,
        LSB-first) at `width` bits each — compact for dictionary indices, which
        rarely form the equal-value runs `encode` exploits. A `width` of 0 (a
        single-entry dictionary) emits nothing; the reader treats every index as
        0. Mirrors the reader's `_unpack8` bit order."""
        var out = List[UInt8]()
        var n = len(values)
        if width == 0 or n == 0:
            return out^
        var num_groups = (n + 7) // 8
        LittleEndian.put_varint(out, (UInt64(num_groups) << 1) | 1)
        var mask = (UInt64(1) << UInt64(width)) - 1
        var acc: UInt64 = 0
        var acc_bits = 0
        for i in range(num_groups * 8):
            var v = (UInt64(Int(values[i])) & mask) if i < n else UInt64(0)
            acc |= v << UInt64(acc_bits)
            acc_bits += width
            while acc_bits >= 8:
                out.append(UInt8(acc & 0xFF))
                acc >>= 8
                acc_bits -= 8
        if acc_bits > 0:
            out.append(UInt8(acc & 0xFF))
        return out^

    @staticmethod
    def encode(values: List[Int32], width: Int) -> List[UInt8]:
        """Encode `values` as an all-RLE-run hybrid stream (runs of equal
        values). Levels are highly repetitive (often all-1s), so run-length runs
        compress them well and keep the encoder trivial."""
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
            LittleEndian.put_varint(out, UInt64(run_len) << 1)  # RLE run
            for b in range(byte_width):
                out.append(UInt8((Int(v) >> (8 * b)) & 0xFF))
            i = j
        return out^


struct DeltaBinaryPacked:
    """The DELTA_BINARY_PACKED integer codec (block / miniblock zigzag deltas).

    Layout (Parquet spec): a header of `block_size`, `miniblocks_per_block`,
    `total_value_count`, and the zigzag `first_value`; then blocks, each a zigzag
    `min_delta`, one bit-width byte per miniblock, and the bit-packed miniblock
    deltas. Each value is `prev + min_delta + unpacked_delta`."""

    @staticmethod
    def decode_into(
        data: Span[UInt8, _], start: Int, count: Int, mut out: List[Int64]
    ) raises -> Int:
        """Decode `count` integers starting at byte `start`, appending to `out`;
        return the byte position just past the stream (so callers like
        DELTA_LENGTH_BYTE_ARRAY can find the data that follows)."""
        if count == 0:
            return start

        var pos = start
        var block_size: UInt64
        block_size, pos = LittleEndian.varint(data, pos)
        var miniblocks: UInt64
        miniblocks, pos = LittleEndian.varint(data, pos)
        var _total: UInt64
        _total, pos = LittleEndian.varint(data, pos)
        var first_z: UInt64
        first_z, pos = LittleEndian.varint(data, pos)

        var num_miniblocks = Int(miniblocks)
        var vals_per_mb = Int(block_size) // num_miniblocks

        var value = Zigzag.decode(first_z)
        out.append(value)
        var produced = 1

        while produced < count:
            var min_delta_z: UInt64
            min_delta_z, pos = LittleEndian.varint(data, pos)
            var min_delta = Zigzag.decode(min_delta_z)
            var widths_at = pos
            pos += num_miniblocks  # one bit-width byte per miniblock
            for mb in range(num_miniblocks):
                var w = Int(data[widths_at + mb])
                var base_bit = pos * 8
                pos += (
                    vals_per_mb * w
                ) // 8  # miniblock is full even if padded
                for j in range(vals_per_mb):
                    var delta: UInt64 = 0
                    if w > 0:
                        delta = LittleEndian.bits(data, base_bit + j * w, w)
                    value += min_delta + Int64(delta)
                    out.append(value)
                    produced += 1
                    if produced == count:
                        return pos
        return pos

    @staticmethod
    def decode(data: Span[UInt8, _], count: Int) raises -> List[Int64]:
        """Decode `count` integers from the whole value stream."""
        var out = List[Int64]()
        out.reserve(count)
        _ = Self.decode_into(data, 0, count, out)
        return out^

    @staticmethod
    def _put_bits(
        mut out: List[UInt8],
        mut acc: UInt64,
        mut acc_bits: Int,
        v: UInt64,
        w: Int,
    ):
        """Append `w` low bits of `v` LSB-first, flushing whole bytes and keeping
        `acc_bits < 8`, so any width up to 64 packs without overflow."""
        var rem = w
        var val = v
        while rem > 0:
            var take = min(8 - acc_bits, rem)
            acc |= (val & ((UInt64(1) << UInt64(take)) - 1)) << UInt64(acc_bits)
            acc_bits += take
            val >>= UInt64(take)
            rem -= take
            if acc_bits == 8:
                out.append(UInt8(acc & 0xFF))
                acc = 0
                acc_bits = 0

    @staticmethod
    def encode(values: List[Int64]) -> List[UInt8]:
        """Encode integers as DELTA_BINARY_PACKED (block 128, 4 miniblocks of 32).
        Each block stores its `min_delta` and a per-miniblock bit width, then the
        bit-packed `delta - min_delta`; a short final block is zero-padded (the
        reader stops at the value count). Also encodes the length/prefix streams
        of the byte-array delta codecs."""
        comptime BLOCK = 128
        comptime NMB = 4
        comptime VPM = BLOCK // NMB  # 32
        var out = List[UInt8]()
        var n = len(values)
        LittleEndian.put_varint(out, UInt64(BLOCK))
        LittleEndian.put_varint(out, UInt64(NMB))
        LittleEndian.put_varint(out, UInt64(n))
        var first = values[0] if n > 0 else Int64(0)
        LittleEndian.put_varint(out, Zigzag.encode(first))
        if n <= 1:
            return out^

        var i = 1
        while i < n:
            var end = min(i + BLOCK, n)
            var min_d = values[i] - values[i - 1]
            for k in range(i + 1, end):
                var d = values[k] - values[k - 1]
                if d < min_d:
                    min_d = d
            LittleEndian.put_varint(out, Zigzag.encode(min_d))

            # (delta - min_delta) for this block, zero-padded to a full BLOCK.
            var rel = List[UInt64](capacity=BLOCK)
            for k in range(i, end):
                rel.append(UInt64((values[k] - values[k - 1]) - min_d))
            while len(rel) < BLOCK:
                rel.append(0)

            var widths = List[Int](capacity=NMB)
            for mb in range(NMB):
                var maxv: UInt64 = 0
                for j in range(VPM):
                    if rel[mb * VPM + j] > maxv:
                        maxv = rel[mb * VPM + j]
                var w = 0
                while (maxv >> UInt64(w)) > 0:
                    w += 1
                widths.append(w)
                out.append(UInt8(w))
            for mb in range(NMB):
                var w = widths[mb]
                if w == 0:
                    continue
                var acc: UInt64 = 0
                var acc_bits = 0
                for j in range(VPM):
                    Self._put_bits(out, acc, acc_bits, rel[mb * VPM + j], w)
                # VPM*w is a multiple of 8, so the miniblock is byte-aligned.
            i = end
        return out^


struct Plain:
    """The PLAIN codec — values laid out in order: fixed-width little-endian for
    primitives, bit-packed for booleans, 4-byte-length-prefixed for byte arrays.
    Encode takes a present-value Arrow array; decode returns the present values.
    """

    @staticmethod
    def encode_primitive[
        store: dt.PrimitiveType, phys: DType, big_endian: Bool = False
    ](arr: PrimitiveArray[store], mut out: List[UInt8]) raises:
        """PLAIN fixed-width encode of the present values, `phys`-wide. `store`
        may be numeric, temporal, decimal, or interval; `big_endian` (for DECIMAL
        FIXED_LEN_BYTE_ARRAY) writes the two's-complement value most-significant
        byte first."""
        comptime W = size_of[Scalar[phys]]()
        for i in range(arr.length):
            if arr.is_valid(i):
                var bytes = (
                    arr[i]
                    .value()
                    .cast[phys]()
                    .as_bytes[big_endian=big_endian]()
                )
                for b in range(W):
                    out.append(bytes[b])

    @staticmethod
    def encode_bool(arr: BoolArray, mut out: List[UInt8]) raises:
        var acc: UInt8 = 0
        var nbits = 0
        for i in range(arr.length):
            if arr.is_valid(i):
                if arr[i].value():
                    acc |= UInt8(1) << UInt8(nbits)
                nbits += 1
                if nbits == 8:
                    out.append(acc)
                    acc = 0
                    nbits = 0
        if nbits > 0:
            out.append(acc)

    @staticmethod
    def encode_bytes[
        BT: dt.BinaryLikeType
    ](arr: BinaryLikeArray[BT], mut out: List[UInt8]) raises:
        """PLAIN byte arrays: each present value's 4-byte LE length then its raw
        bytes. Serves string/binary and their large_ variants alike."""
        for i in range(arr.length):
            if arr.is_valid(i):
                var b = arr.unsafe_get(UInt(i)).as_bytes()
                LittleEndian.put_u32(out, len(b))
                out.extend(b)

    @staticmethod
    def encode_fixed_size_binary(
        arr: FixedSizeBinaryArray, mut out: List[UInt8]
    ) raises:
        """FIXED_LEN_BYTE_ARRAY: the present values' `byte_width` bytes, no
        length prefix (the width is in the schema)."""
        for i in range(arr.length):
            if arr.is_valid(i):
                out.extend(Span(arr[i].value()))

    @staticmethod
    def decode_primitive[
        store: DType, phys: DType
    ](values: Span[UInt8, _], np: Int, mut out: List[Scalar[store]]) raises:
        comptime PW = size_of[Scalar[phys]]()
        for i in range(np):
            out.append(LittleEndian.fixed[phys](values, i * PW).cast[store]())

    @staticmethod
    def decode_bytes(
        values: Span[UInt8, _], np: Int
    ) raises -> List[List[UInt8]]:
        var out = List[List[UInt8]]()
        var vi = 0
        for _ in range(np):
            var n = LittleEndian.u32(values, vi)
            vi += 4
            var v = List[UInt8]()
            v.extend(values[vi : vi + n])
            out.append(v^)
            vi += n
        return out^

    @staticmethod
    def decode_bool(values: Span[UInt8, _], np: Int) raises -> List[Bool]:
        var out = List[Bool]()
        for i in range(np):
            var byte = values[i >> 3]
            out.append(((byte >> UInt8(i & 7)) & 1) == 1)
        return out^


struct Dictionary:
    """The dictionary codec — a dictionary page of distinct values, then data
    pages of RLE/bit-packed indices into it. `encode` builds both from a column;
    `decode_page_*` reads the dictionary page and `decode_*` reads a data page's
    indices and gathers the values."""

    # --- encode: column -> dictionary-page bytes + per-present-value indices ---

    @staticmethod
    def _encode_prim[
        store: dt.NumericType, phys: DType
    ](
        arr: PrimitiveArray[store],
        mut dict_body: List[UInt8],
        mut indices: List[Int32],
    ) raises -> Int:
        """Dictionary-encode a primitive column: PLAIN-encode each distinct value
        (widened to `phys`) into `dict_body`, collect the per-value index."""
        comptime W = size_of[Scalar[phys]]()
        var seen = Dict[Scalar[store.native], Int]()
        var num_dict = 0
        for i in range(arr.length):
            if arr.is_valid(i):
                var v = arr[i].value()
                if v in seen:
                    indices.append(Int32(seen[v]))
                else:
                    seen[v] = num_dict
                    indices.append(Int32(num_dict))
                    num_dict += 1
                    var bytes = v.cast[phys]().as_bytes[big_endian=False]()
                    for b in range(W):
                        dict_body.append(bytes[b])
        return num_dict

    @staticmethod
    def _encode_bytes[
        BT: dt.BinaryLikeType
    ](
        arr: BinaryLikeArray[BT],
        mut dict_body: List[UInt8],
        mut indices: List[Int32],
    ) raises -> Int:
        """Dictionary-encode a byte-array column (string/binary and their large_
        variants): length-prefixed distinct values in the dictionary page, one
        index per present value."""
        var seen = Dict[String, Int]()
        var num_dict = 0
        for i in range(arr.length):
            if arr.is_valid(i):
                var v = String(arr.unsafe_get(UInt(i)))
                if v in seen:
                    indices.append(Int32(seen[v]))
                else:
                    seen[v] = num_dict
                    indices.append(Int32(num_dict))
                    num_dict += 1
                    var b = v.as_bytes()
                    LittleEndian.put_u32(dict_body, len(b))
                    dict_body.extend(b)
        return num_dict

    @staticmethod
    def encode(
        dtype: dt.AnyDataType,
        col: AnyArray,
        mut dict_body: List[UInt8],
        mut indices: List[Int32],
    ) raises -> Int:
        """Build the dictionary page bytes + per-value indices for `col`; returns
        the dictionary size. Dispatches on the Arrow type like the PLAIN encoders
        (bool is never dictionary-encoded)."""
        if dtype == dt.int32:
            return Self._encode_prim[dt.Int32Type, DType.int32](
                col.as_int32(), dict_body, indices
            )
        elif dtype == dt.int64:
            return Self._encode_prim[dt.Int64Type, DType.int64](
                col.as_int64(), dict_body, indices
            )
        elif dtype == dt.uint32:
            return Self._encode_prim[dt.UInt32Type, DType.uint32](
                col.as_uint32(), dict_body, indices
            )
        elif dtype == dt.uint64:
            return Self._encode_prim[dt.UInt64Type, DType.uint64](
                col.as_uint64(), dict_body, indices
            )
        elif dtype == dt.float32:
            return Self._encode_prim[dt.Float32Type, DType.float32](
                col.as_float32(), dict_body, indices
            )
        elif dtype == dt.float64:
            return Self._encode_prim[dt.Float64Type, DType.float64](
                col.as_float64(), dict_body, indices
            )
        elif dtype == dt.float16:
            return Self._encode_prim[dt.Float16Type, DType.float16](
                col.as_float16(), dict_body, indices
            )
        elif dtype == dt.int8:
            return Self._encode_prim[dt.Int8Type, DType.int32](
                col.as_int8(), dict_body, indices
            )
        elif dtype == dt.int16:
            return Self._encode_prim[dt.Int16Type, DType.int32](
                col.as_int16(), dict_body, indices
            )
        elif dtype == dt.uint8:
            return Self._encode_prim[dt.UInt8Type, DType.int32](
                col.as_uint8(), dict_body, indices
            )
        elif dtype == dt.uint16:
            return Self._encode_prim[dt.UInt16Type, DType.int32](
                col.as_uint16(), dict_body, indices
            )
        elif dtype.is_string():
            return Self._encode_bytes(col.as_string(), dict_body, indices)
        elif dtype.is_large_string():
            return Self._encode_bytes(col.as_large_string(), dict_body, indices)
        elif dtype.is_binary():
            return Self._encode_bytes(col.as_binary(), dict_body, indices)
        elif dtype.is_large_binary():
            return Self._encode_bytes(col.as_large_binary(), dict_body, indices)
        else:
            raise Error(
                "parquet: cannot dictionary-encode column type " + String(dtype)
            )

    # --- decode ---

    @staticmethod
    def decode_page_primitive[
        store: DType, phys: DType
    ](
        body: Span[UInt8, _], num_values: Int, mut dict: List[Scalar[store]]
    ) raises:
        """Read a primitive dictionary page (PLAIN fixed-width) into `dict`."""
        comptime PW = size_of[Scalar[phys]]()
        for i in range(num_values):
            dict.append(LittleEndian.fixed[phys](body, i * PW).cast[store]())

    @staticmethod
    def decode_page_bytes(
        body: Span[UInt8, _],
        num_values: Int,
        mut dict_body: List[UInt8],
        mut dict_off: List[Int],
        mut dict_len: List[Int],
    ) raises:
        """Read a byte-array dictionary page (length-prefixed values)."""
        dict_body.clear()
        dict_body.extend(body)
        var span = Span(dict_body)
        var off = 0
        for _ in range(num_values):
            var n = LittleEndian.u32(span, off)
            off += 4
            dict_off.append(off)
            dict_len.append(n)
            off += n

    @staticmethod
    def decode_primitive[
        store: DType
    ](
        values: Span[UInt8, _],
        np: Int,
        dict: List[Scalar[store]],
        mut out: List[Scalar[store]],
    ) raises:
        var base = len(out)
        out.resize(unsafe_uninit_length=base + np)
        Rle.gather[store](
            values[1:],
            Int(values[0]),
            np,
            Span(dict),
            Span(out),
            base,
        )

    @staticmethod
    def decode_bytes(
        values: Span[UInt8, _],
        np: Int,
        dict_body: List[UInt8],
        dict_off: List[Int],
        dict_len: List[Int],
    ) raises -> List[List[UInt8]]:
        var out = List[List[UInt8]]()
        var indices = Rle.decode(values[1:], Int(values[0]), np)
        for i in range(np):
            var idx = Int(indices[i])
            var start = dict_off[idx]
            var v = List[UInt8]()
            v.extend(Span(dict_body)[start : start + dict_len[idx]])
            out.append(v^)
        return out^


struct ByteStreamSplit:
    """The BYTE_STREAM_SPLIT codec — each value's bytes are transposed into
    per-byte planes (byte `k` of value `i` at `values[k*np + i]`)."""

    @staticmethod
    def decode_primitive[
        store: DType, phys: DType
    ](values: Span[UInt8, _], np: Int, mut out: List[Scalar[store]]) raises:
        comptime PW = size_of[Scalar[phys]]()
        for i in range(np):
            var raw = InlineArray[UInt8, PW](fill=0)

            comptime for k in range(PW):
                raw[k] = values[k * np + i]
            out.append(
                SIMD[phys, 1].from_bytes[big_endian=False](raw).cast[store]()
            )

    @staticmethod
    def decode_flba(
        values: Span[UInt8, _], np: Int, width: Int
    ) raises -> List[UInt8]:
        """Runtime-width transpose for FIXED_LEN_BYTE_ARRAY: value `i`'s byte `k`
        is at `values[k*np + i]`. Returns the `np * width` value-major bytes (the
        `decode_primitive` counterpart when the width is only known at runtime).
        """
        var out = List[UInt8](length=np * width, fill=0)
        for i in range(np):
            for k in range(width):
                out[i * width + k] = values[k * np + i]
        return out^

    @staticmethod
    def encode[
        store: dt.NumericType, phys: DType
    ](arr: PrimitiveArray[store], mut out: List[UInt8]) raises:
        """Transpose the present values into per-byte planes: byte `k` of value
        `i` at `out[base + k*np + i]` (inverse of `decode_primitive`)."""
        comptime PW = size_of[Scalar[phys]]()
        # value-major little-endian bytes of the present values
        var raw = List[UInt8]()
        for i in range(arr.length):
            if arr.is_valid(i):
                var b = arr[i].value().cast[phys]().as_bytes[big_endian=False]()
                for k in range(PW):
                    raw.append(b[k])
        var np = len(raw) // PW
        for k in range(PW):
            for i in range(np):
                out.append(raw[i * PW + k])


struct DeltaLengthByteArray:
    """The DELTA_LENGTH_BYTE_ARRAY codec — a delta-packed length stream followed
    by the concatenated value bytes."""

    @staticmethod
    def decode_bytes(
        values: Span[UInt8, _], np: Int
    ) raises -> List[List[UInt8]]:
        var out = List[List[UInt8]]()
        var lengths = List[Int64]()
        var pos = DeltaBinaryPacked.decode_into(values, 0, np, lengths)
        for i in range(np):
            var n = Int(lengths[i])
            var v = List[UInt8]()
            v.extend(values[pos : pos + n])
            out.append(v^)
            pos += n
        return out^

    @staticmethod
    def encode[
        BT: dt.BinaryLikeType
    ](arr: BinaryLikeArray[BT], mut out: List[UInt8]) raises:
        """DELTA_LENGTH_BYTE_ARRAY: a delta-packed length stream then the
        concatenated present-value bytes."""
        var lengths = List[Int64]()
        var data = List[UInt8]()
        for i in range(arr.length):
            if arr.is_valid(i):
                var b = arr.unsafe_get(UInt(i)).as_bytes()
                lengths.append(Int64(len(b)))
                data.extend(b)
        out.extend(Span(DeltaBinaryPacked.encode(lengths)))
        out.extend(Span(data))


struct DeltaByteArray:
    """The DELTA_BYTE_ARRAY codec — incremental prefix reconstruction: a
    delta-packed shared-prefix-length stream, then a delta-packed suffix-length
    stream, then the suffix bytes; each value is `prev[:prefix] + suffix`."""

    @staticmethod
    def decode_bytes(
        values: Span[UInt8, _], np: Int
    ) raises -> List[List[UInt8]]:
        var out = List[List[UInt8]]()
        var prefixes = List[Int64]()
        var pos = DeltaBinaryPacked.decode_into(values, 0, np, prefixes)
        var suffix_lens = List[Int64]()
        pos = DeltaBinaryPacked.decode_into(values, pos, np, suffix_lens)
        var prev = List[UInt8]()
        for i in range(np):
            var v = List[UInt8]()
            v.extend(Span(prev)[0 : Int(prefixes[i])])
            var sl = Int(suffix_lens[i])
            v.extend(values[pos : pos + sl])
            pos += sl
            out.append(v.copy())
            prev = v^
        return out^

    @staticmethod
    def encode[
        BT: dt.BinaryLikeType
    ](arr: BinaryLikeArray[BT], mut out: List[UInt8]) raises:
        """DELTA_BYTE_ARRAY: a delta-packed shared-prefix-length stream, then a
        delta-packed suffix-length stream, then the suffix bytes; each value is
        `prev[:prefix] + suffix`."""
        var prefixes = List[Int64]()
        var suffix_lens = List[Int64]()
        var suffix_data = List[UInt8]()
        var prev = List[UInt8]()
        for i in range(arr.length):
            if arr.is_valid(i):
                var v = List[UInt8](arr.unsafe_get(UInt(i)).as_bytes())
                var m = min(len(prev), len(v))
                var p = 0
                while p < m and prev[p] == v[p]:
                    p += 1
                prefixes.append(Int64(p))
                suffix_lens.append(Int64(len(v) - p))
                for k in range(p, len(v)):
                    suffix_data.append(v[k])
                prev = v^
        out.extend(Span(DeltaBinaryPacked.encode(prefixes)))
        out.extend(Span(DeltaBinaryPacked.encode(suffix_lens)))
        out.extend(Span(suffix_data))


@fieldwise_init
struct Encoding(Equatable, ImplicitlyCopyable, Movable):
    """A Parquet `Encoding` enum value. The decode methods dispatch a data page's
    *present* values to the per-encoding codec above; each appends `num_present`
    values from a value byte-span (nulls are placed later by the caller from the
    definition levels), so the same logic serves the flat and nested reader
    paths."""

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
        if self.is_plain():
            Plain.decode_primitive[store, phys](values, num_present, out)
        elif self.is_dictionary():
            Dictionary.decode_primitive[store](values, num_present, dict, out)
        elif self == Self.DELTA_BINARY_PACKED:
            var decoded = DeltaBinaryPacked.decode(values, num_present)
            for i in range(num_present):
                out.append(decoded[i].cast[store]())
        elif self == Self.BYTE_STREAM_SPLIT:
            ByteStreamSplit.decode_primitive[store, phys](
                values, num_present, out
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
        if self.is_plain():
            return Plain.decode_bytes(values, num_present)
        elif self.is_dictionary():
            return Dictionary.decode_bytes(
                values, num_present, dict_body, dict_off, dict_len
            )
        elif self == Self.DELTA_LENGTH_BYTE_ARRAY:
            return DeltaLengthByteArray.decode_bytes(values, num_present)
        elif self == Self.DELTA_BYTE_ARRAY:
            return DeltaByteArray.decode_bytes(values, num_present)
        else:
            raise Error(
                "parquet: unsupported byte-array encoding " + String(self.code)
            )

    def decode_flba(
        self, values: Span[UInt8, _], num_present: Int, width: Int
    ) raises -> List[UInt8]:
        """Decode the present FIXED_LEN_BYTE_ARRAY values (each `width` bytes) of
        a non-PLAIN/dict page into a contiguous `num_present * width` buffer — the
        DELTA_BYTE_ARRAY and BYTE_STREAM_SPLIT encodings PyArrow emits for decimal
        / fixed_size_binary. PLAIN and dictionary pages are read in place by the
        leaf builders, so they never reach here."""
        if self == Self.DELTA_BYTE_ARRAY:
            var vals = DeltaByteArray.decode_bytes(values, num_present)
            var out = List[UInt8](capacity=num_present * width)
            for i in range(num_present):
                out.extend(Span(vals[i]))
            return out^
        elif self == Self.BYTE_STREAM_SPLIT:
            return ByteStreamSplit.decode_flba(values, num_present, width)
        else:
            raise Error(
                "parquet: unsupported FIXED_LEN_BYTE_ARRAY encoding "
                + String(self.code)
            )

    def decode_bool(
        self, values: Span[UInt8, _], num_present: Int
    ) raises -> List[Bool]:
        """Return the present booleans — PLAIN bit-packed, or RLE (the encoding
        arrow/PyArrow use for boolean values in DataPage v2). An RLE boolean
        stream is a 4-byte little-endian length then a width-1 RLE/bit-packed
        hybrid run, exactly like a level stream."""
        if self.is_plain():
            return Plain.decode_bool(values, num_present)
        elif self == Self.RLE:
            var length = LittleEndian.u32(values, 0)
            var decoded = Rle.decode(values[4 : 4 + length], 1, num_present)
            var out = List[Bool](capacity=num_present)
            for i in range(num_present):
                out.append(Int(decoded[i]) == 1)
            return out^
        else:
            raise Error("parquet: non-plain bool encoding not supported")


@fieldwise_init
struct Compression(Equatable, ImplicitlyCopyable, Movable):
    """A Parquet `CompressionCodec` value: the codec identity plus the
    `compress` / `decompress` operations, dispatched onto a `CompressionLibs`
    handle pool (the `dlopen` bindings in `utils.mojo`). Enum values:
        0 UNCOMPRESSED  1 SNAPPY  2 GZIP  4 BROTLI  5 LZ4  6 ZSTD  7 LZ4_RAW
    """

    var code: Int

    comptime UNCOMPRESSED = Self(0)
    comptime SNAPPY = Self(1)
    comptime GZIP = Self(2)
    comptime BROTLI = Self(4)
    comptime LZ4 = Self(5)
    comptime ZSTD = Self(6)
    comptime LZ4_RAW = Self(7)

    def decompress_into(
        self,
        mut libs: CompressionLibs,
        src: Span[UInt8, _],
        out_size: Int,
        mut scratch: List[UInt8],
    ) raises:
        """Decompress `src` into `scratch` (resized, reused across pages).

        8 trailing bytes of slack let the bit-unpackers do unaligned 64-bit
        loads past the last value without overrunning the buffer."""
        scratch.resize(unsafe_uninit_length=out_size + 8)
        var ptr = scratch.unsafe_ptr()
        if self == Self.UNCOMPRESSED:
            memcpy(dest=ptr, src=src.unsafe_ptr(), count=out_size)
        elif self == Self.ZSTD:
            libs.zstd_decompress(src, ptr, out_size)
        elif self == Self.SNAPPY:
            libs.snappy_decompress(src, ptr, out_size)
        elif self == Self.LZ4_RAW:
            libs.lz4_raw_decompress(src, ptr, out_size)
        elif self == Self.LZ4:
            # Deprecated LZ4 (code 5): modern writers (PyArrow) emit a plain LZ4
            # block, but tolerate the legacy Hadoop frame ([be u32 decompressed
            # size][be u32 compressed size] prefix) by stripping it when present.
            libs.lz4_raw_decompress(
                Self._strip_lz4_frame(src, out_size), ptr, out_size
            )
        elif self == Self.GZIP:
            libs.gzip_decompress(src, ptr, out_size)
        elif self == Self.BROTLI:
            libs.brotli_decompress(src, ptr, out_size)
        else:
            raise Error(
                "parquet: unsupported compression codec " + String(self.code)
            )

    @staticmethod
    def _strip_lz4_frame[
        o: Origin[mut=False]
    ](src: Span[UInt8, o], out_size: Int) -> Span[UInt8, o]:
        """Strip the legacy Hadoop LZ4 8-byte frame header when present: a
        big-endian u32 decompressed size (== `out_size`) then a big-endian u32
        compressed size (== the remaining bytes). A plain LZ4 block is returned
        unchanged."""
        if len(src) >= 8:
            var dlen = (
                (Int(src[0]) << 24)
                | (Int(src[1]) << 16)
                | (Int(src[2]) << 8)
                | Int(src[3])
            )
            var clen = (
                (Int(src[4]) << 24)
                | (Int(src[5]) << 16)
                | (Int(src[6]) << 8)
                | Int(src[7])
            )
            if dlen == out_size and clen == len(src) - 8:
                return src[8:]
        return src

    def decompress(
        self, mut libs: CompressionLibs, src: Span[UInt8, _], out_size: Int
    ) raises -> List[UInt8]:
        """Decompress `src` into a fresh `out_size`-byte list."""
        var dst = List[UInt8]()
        self.decompress_into(libs, src, out_size, dst)
        dst.resize(unsafe_uninit_length=out_size)  # drop the scratch slack
        return dst^

    def compress(
        self, mut libs: CompressionLibs, src: Span[UInt8, _]
    ) raises -> List[UInt8]:
        """Compress `src`, returning the codec's output bytes. Writers emit
        UNCOMPRESSED, SNAPPY, ZSTD, GZIP, BROTLI, LZ4, or LZ4_RAW."""
        if self == Self.UNCOMPRESSED:
            var out = List[UInt8]()
            out.extend(src)
            return out^
        elif self == Self.ZSTD:
            return libs.zstd_compress(src)
        elif self == Self.SNAPPY:
            return libs.snappy_compress(src)
        elif self == Self.LZ4_RAW or self == Self.LZ4:
            # Both emit a plain LZ4 block; code 5 readers accept it (see the
            # Hadoop-frame tolerance in `_strip_lz4_frame`).
            return libs.lz4_compress(src)
        elif self == Self.GZIP:
            return libs.gzip_compress(src)
        elif self == Self.BROTLI:
            return libs.brotli_compress(src)
        else:
            raise Error(
                "parquet: unsupported compression codec " + String(self.code)
            )
