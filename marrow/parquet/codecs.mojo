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

from ..arrays import PrimitiveArray, StringArray, BoolArray
from .. import dtypes as dt
from ..views import load_word_le
from .utils import CompressionLibs


struct LittleEndian:
    """Little-endian byte, bit, and LEB128-varint reads over a byte span — the
    low-level primitives the codecs below share. Stateless; a namespace of
    static methods rather than free functions."""

    @staticmethod
    def u32(body: Span[UInt8, _], off: Int) -> Int:
        return (
            Int(body[off])
            | (Int(body[off + 1]) << 8)
            | (Int(body[off + 2]) << 16)
            | (Int(body[off + 3]) << 24)
        )

    @staticmethod
    def fixed[dt: DType](body: Span[UInt8, _], off: Int) -> Scalar[dt]:
        comptime W = size_of[Scalar[dt]]()
        var arr = InlineArray[UInt8, W](fill=0)
        for i in range(W):
            arr[i] = body[off + i]
        return SIMD[dt, 1].from_bytes[big_endian=False](arr)

    @staticmethod
    def varint(data: Span[UInt8, _], pos: Int) raises -> Tuple[UInt64, Int]:
        """Read an unsigned LEB128 varint at `pos`; return `(value, next_pos)`.
        """
        var result: UInt64 = 0
        var shift: Int = 0
        var p = pos
        while True:
            if p >= len(data):
                raise Error("parquet: varint out of bounds")
            var b = data[p]
            p += 1
            result |= UInt64(b & 0x7F) << UInt64(shift)
            if b & 0x80 == 0:
                break
            shift += 7
        return (result, p)

    @staticmethod
    def put_u32(mut out: List[UInt8], v: Int):
        out.append(UInt8(v & 0xFF))
        out.append(UInt8((v >> 8) & 0xFF))
        out.append(UInt8((v >> 16) & 0xFF))
        out.append(UInt8((v >> 24) & 0xFF))

    @staticmethod
    def put_le(mut out: List[UInt8], bits: UInt64, width: Int):
        """Append the low `width` bytes of `bits`, least-significant first."""
        for i in range(width):
            out.append(UInt8((bits >> UInt64(i * 8)) & 0xFF))

    @staticmethod
    def bytes_less(a: Span[UInt8, _], b: Span[UInt8, _]) -> Bool:
        """Unsigned byte-wise lexicographic `a < b` (BYTE_ARRAY ordering)."""
        var n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return a[i] < b[i]
        return len(a) < len(b)

    @staticmethod
    def put_varint(mut out: List[UInt8], var v: UInt64):
        """Append `v` as an unsigned LEB128 varint."""
        while True:
            var b = UInt8(v & 0x7F)
            v >>= 7
            if v != 0:
                out.append(b | 0x80)
            else:
                out.append(b)
                break

    @staticmethod
    def bits(data: Span[UInt8, _], bit_offset: Int, nbits: Int) -> UInt64:
        """Read `nbits` starting at absolute `bit_offset`, least-significant
        first."""
        var result: UInt64 = 0
        for i in range(nbits):
            var abs_bit = bit_offset + i
            var byte_idx = abs_bit >> 3
            var bit_idx = abs_bit & 7
            var bit = (UInt64(data[byte_idx]) >> UInt64(bit_idx)) & 1
            result |= bit << UInt64(i)
        return result


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
                var val: Int32 = 0
                for b in range(byte_width):
                    val |= Int32(Int(data[pos + b]) << (8 * b))
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
        dict: UnsafePointer[Scalar[dt], _],
        dest: UnsafePointer[Scalar[dt], do],
    ) raises:
        """Decode `count` RLE/bit-packed dictionary indices and write `dict[idx]`
        straight to `dest` — fuses index decode and gather (no index buffer)."""
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
                var val: Int32 = 0
                for b in range(byte_width):
                    val |= Int32(Int(data[pos + b]) << (8 * b))
                pos += byte_width
                var take = min(run_len, count - produced)
                if val == target:
                    matches += take
                produced += take
        return matches

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
    @always_inline
    def _zigzag(u: UInt64) -> Int64:
        """ULEB128 zigzag -> signed."""
        return Int64(u >> 1) ^ -Int64(u & 1)

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

        var value = Self._zigzag(first_z)
        out.append(value)
        var produced = 1

        while produced < count:
            var min_delta_z: UInt64
            min_delta_z, pos = LittleEndian.varint(data, pos)
            var min_delta = Self._zigzag(min_delta_z)
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


struct Plain:
    """The PLAIN codec — values laid out in order: fixed-width little-endian for
    primitives, bit-packed for booleans, 4-byte-length-prefixed for byte arrays.
    Encode takes a present-value Arrow array; decode returns the present values.
    """

    @staticmethod
    def encode_primitive[
        store: dt.NumericType, phys: DType
    ](arr: PrimitiveArray[store], mut out: List[UInt8]) raises:
        comptime W = size_of[Scalar[phys]]()
        for i in range(arr.length):
            if arr.is_valid(i):
                var bytes = (
                    arr[i].value().cast[phys]().as_bytes[big_endian=False]()
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
    def encode_bytes(arr: StringArray, mut out: List[UInt8]) raises:
        for i in range(arr.length):
            if arr.is_valid(i):
                var b = String(arr[i]).as_bytes()
                LittleEndian.put_u32(out, len(b))
                out.extend(b)

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
    pages of RLE/bit-packed indices into it. `decode_page_*` reads the dictionary
    page; `decode_*` reads a data page's indices and gathers the values."""

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
            dict.unsafe_ptr(),
            out.unsafe_ptr() + base,
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

    def decode_bool(
        self, values: Span[UInt8, _], num_present: Int
    ) raises -> List[Bool]:
        """Return the present PLAIN bit-packed booleans."""
        if not self.is_plain():
            raise Error("parquet: non-plain bool encoding not supported")
        return Plain.decode_bool(values, num_present)


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

    def __init__(out self, code: Int):
        self.code = code

    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        return self.code != other.code

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
        elif self == Self.GZIP:
            libs.gzip_decompress(src, ptr, out_size)
        else:
            raise Error(
                "parquet: unsupported compression codec " + String(self.code)
            )

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
        """Compress `src`, returning the codec's output bytes. Writers currently
        emit UNCOMPRESSED, SNAPPY, ZSTD, or LZ4_RAW."""
        if self == Self.UNCOMPRESSED:
            var out = List[UInt8]()
            out.extend(src)
            return out^
        elif self == Self.ZSTD:
            return libs.zstd_compress(src)
        elif self == Self.SNAPPY:
            return libs.snappy_compress(src)
        elif self == Self.LZ4_RAW:
            return libs.lz4_compress(src)
        else:
            raise Error(
                "parquet: unsupported compression codec " + String(self.code)
            )
