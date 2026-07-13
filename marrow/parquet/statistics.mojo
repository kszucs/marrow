"""Column statistics — the min/max bounds shared by the reader and writer.

`Statistics.min_max` computes a column chunk's PLAIN-encoded min/max bytes from
an Arrow array (write side); `Statistics.decode` turns one such min/max byte
buffer back into a typed scalar (read side). Keeping both directions here means
the per-type byte layout is defined once, so the two sides cannot drift.
"""

from std.math import isnan
from std.sys import size_of

from .. import dtypes as dt
from ..dtypes import PrimitiveType
from ..arrays import AnyArray, PrimitiveArray, BinaryLikeArray
from ..utils import LittleEndian
from ..scalars import (
    AnyScalar,
    BoolScalar,
    StringScalar,
    Int8Scalar,
    Int16Scalar,
    Int32Scalar,
    Int64Scalar,
    UInt8Scalar,
    UInt16Scalar,
    UInt32Scalar,
    UInt64Scalar,
    Float16Scalar,
    Float32Scalar,
    Float64Scalar,
    Date32Scalar,
    Time32Scalar,
    Time64Scalar,
    TimestampScalar,
    Decimal32Scalar,
    Decimal64Scalar,
    Decimal128Scalar,
    Decimal256Scalar,
    FixedSizeBinaryScalar,
)
from .codecs import Plain


struct Statistics:
    """Bidirectional column-chunk statistics: min/max byte encode + decode."""

    @staticmethod
    def _mm[
        T: PrimitiveType, skip_nan: Bool = False
    ](arr: PrimitiveArray[T]) raises -> Tuple[
        Scalar[T.native], Scalar[T.native], Bool
    ]:
        """min/max over the valid values in T's native (signed/unsigned) order.
        `skip_nan` excludes NaN — floats only, per the spec (NaN bounds nothing);
        the check is comptime-elided for integer columns."""
        var seen = False
        var mn = Scalar[T.native](0)
        var mx = Scalar[T.native](0)
        for i in range(len(arr)):
            if arr.is_valid(i):
                var v = arr[i].value()

                @parameter
                if skip_nan:
                    if isnan(v):
                        continue
                if not seen:
                    mn = v
                    mx = v
                    seen = True
                else:
                    if v < mn:
                        mn = v
                    if v > mx:
                        mx = v
        return (mn, mx, seen)

    @staticmethod
    def _int_stats[
        T: PrimitiveType, width: Int
    ](
        arr: PrimitiveArray[T],
        mut min_out: List[UInt8],
        mut max_out: List[UInt8],
    ) raises -> Bool:
        """Signed/unsigned integer bounds, widened to `width` little-endian bytes
        (the physical INT32 / INT64 width). False when there are no values."""
        var r = Self._mm(arr)
        if not r[2]:
            return False
        LittleEndian.put_le(min_out, r[0].cast[DType.uint64](), width)
        LittleEndian.put_le(max_out, r[1].cast[DType.uint64](), width)
        return True

    @staticmethod
    def _float_stats[
        T: PrimitiveType, width: Int
    ](
        arr: PrimitiveArray[T],
        mut min_out: List[UInt8],
        mut max_out: List[UInt8],
    ) raises -> Bool:
        """IEEE float bounds (NaN skipped), stored as their bit pattern with the
        signed zero normalised so the bound brackets both +0.0 and -0.0."""
        var r = Self._mm[skip_nan=True](arr)
        if not r[2]:
            return False
        var zero = Scalar[T.native](0)
        var mn = -zero if r[0] == zero else r[0]
        var mx = zero if r[1] == zero else r[1]
        LittleEndian.put_le(min_out, UInt64(mn.to_bits()), width)
        LittleEndian.put_le(max_out, UInt64(mx.to_bits()), width)
        return True

    @staticmethod
    def _decimal_flba_stats[
        T: PrimitiveType, width: Int
    ](
        arr: PrimitiveArray[T],
        mut min_out: List[UInt8],
        mut max_out: List[UInt8],
    ) raises -> Bool:
        """DECIMAL FIXED_LEN_BYTE_ARRAY bounds: signed numeric min/max over the
        int128/int256 values, big-endian two's complement of `width` bytes (the
        full storage width, matching the value encoding)."""
        var r = Self._mm(arr)
        if not r[2]:
            return False
        var lo = r[0].as_bytes[big_endian=True]()
        var hi = r[1].as_bytes[big_endian=True]()
        for b in range(width):
            min_out.append(lo[b])
            max_out.append(hi[b])
        return True

    @staticmethod
    def _update_minmax(
        v: Span[UInt8, _],
        mut lo: List[UInt8],
        mut hi: List[UInt8],
        mut seen: Bool,
    ):
        """Fold one present value `v` into the running byte-wise `lo`/`hi`
        bounds (unsigned lexicographic, the BYTE_ARRAY / FIXED_LEN ordering)."""
        if not seen:
            lo = List[UInt8](v)
            hi = List[UInt8](v)
            seen = True
        else:
            if LittleEndian.bytes_less(v, Span(lo)):
                lo = List[UInt8](v)
            if LittleEndian.bytes_less(Span(hi), v):
                hi = List[UInt8](v)

    @staticmethod
    def _bytes_stats[
        BT: dt.BinaryLikeType
    ](
        arr: BinaryLikeArray[BT],
        mut min_out: List[UInt8],
        mut max_out: List[UInt8],
    ) raises -> Bool:
        """Unsigned byte-wise lexicographic min/max over the present values —
        the BYTE_ARRAY ordering shared by string/binary and their large_
        variants. Skips very long bounds rather than truncate (a missing bound is
        always valid)."""
        var seen = False
        var lo = List[UInt8]()
        var hi = List[UInt8]()
        for i in range(arr.length):
            if arr.is_valid(i):
                Self._update_minmax(
                    arr.unsafe_get(UInt(i)).as_bytes(), lo, hi, seen
                )
        if not seen or len(lo) > 4096 or len(hi) > 4096:
            return False
        min_out = lo^
        max_out = hi^
        return True

    @staticmethod
    def min_max(
        dtype: dt.AnyDataType,
        col: AnyArray,
        mut min_out: List[UInt8],
        mut max_out: List[UInt8],
    ) raises -> Bool:
        """Fill `min_out`/`max_out` with the PLAIN-encoded bounds; return False
        when there is nothing to summarise (all-null / empty) or the type carries
        no statistics. Byte arrays store raw bytes with no length prefix; the
        numeric widen/normalise rules live in `_int_stats` / `_float_stats`."""
        ref vt = dtype
        if vt == dt.int8:
            return Self._int_stats[width=4](col.as_int8(), min_out, max_out)
        elif vt == dt.int16:
            return Self._int_stats[width=4](col.as_int16(), min_out, max_out)
        elif vt == dt.int32:
            return Self._int_stats[width=4](col.as_int32(), min_out, max_out)
        elif vt == dt.uint8:
            return Self._int_stats[width=4](col.as_uint8(), min_out, max_out)
        elif vt == dt.uint16:
            return Self._int_stats[width=4](col.as_uint16(), min_out, max_out)
        elif vt == dt.uint32:
            return Self._int_stats[width=4](col.as_uint32(), min_out, max_out)
        elif vt == dt.int64:
            return Self._int_stats[width=8](col.as_int64(), min_out, max_out)
        elif vt == dt.uint64:
            return Self._int_stats[width=8](col.as_uint64(), min_out, max_out)
        elif vt == dt.float32:
            return Self._float_stats[width=4](
                col.as_float32(), min_out, max_out
            )
        elif vt == dt.float64:
            return Self._float_stats[width=8](
                col.as_float64(), min_out, max_out
            )
        elif vt == dt.float16:
            # FLOAT16 bounds follow IEEE float ordering, stored as the 2-byte
            # little-endian half bit pattern (the FLBA(2) value encoding).
            return Self._float_stats[width=2](
                col.as_float16(), min_out, max_out
            )
        elif vt == dt.bool_:
            ref b = col.as_bool()
            var seen = False
            var any_true = False
            var any_false = False
            for i in range(len(b)):
                if b.is_valid(i):
                    seen = True
                    if b[i].value():
                        any_true = True
                    else:
                        any_false = True
            if not seen:
                return False
            min_out.append(UInt8(0) if any_false else UInt8(1))
            max_out.append(UInt8(1) if any_true else UInt8(0))
            return True
        elif vt.is_string():
            return Self._bytes_stats(col.as_string(), min_out, max_out)
        elif vt.is_large_string():
            return Self._bytes_stats(col.as_large_string(), min_out, max_out)
        elif vt.is_binary():
            return Self._bytes_stats(col.as_binary(), min_out, max_out)
        elif vt.is_large_binary():
            return Self._bytes_stats(col.as_large_binary(), min_out, max_out)
        elif vt.is_date32():
            # temporal stored as INT32/INT64 — signed integer ordering
            return Self._int_stats[width=4](col.as_date32(), min_out, max_out)
        elif vt.is_time32():
            return Self._int_stats[width=4](col.as_time32(), min_out, max_out)
        elif vt.is_timestamp():
            return Self._int_stats[width=8](
                col.as_timestamp(), min_out, max_out
            )
        elif vt.is_time64():
            return Self._int_stats[width=8](col.as_time64(), min_out, max_out)
        elif vt.is_fixed_size_binary():
            ref fsb = col.as_fixed_size_binary()
            var seen = False
            var lo = List[UInt8]()
            var hi = List[UInt8]()
            for i in range(len(fsb)):
                if fsb.is_valid(i):
                    Self._update_minmax(Span(fsb[i].value()), lo, hi, seen)
            if not seen:
                return False
            min_out = lo^
            max_out = hi^
            return True
        elif vt.is_decimal32():
            return Self._int_stats[width=4](
                col.as_decimal32(), min_out, max_out
            )
        elif vt.is_decimal64():
            return Self._int_stats[width=8](
                col.as_decimal64(), min_out, max_out
            )
        elif vt.is_decimal128():
            return Self._decimal_flba_stats[width=16](
                col.as_decimal128(), min_out, max_out
            )
        elif vt.is_decimal256():
            return Self._decimal_flba_stats[width=32](
                col.as_decimal256(), min_out, max_out
            )
        else:
            return False

    @staticmethod
    def decode(
        dtype: dt.AnyDataType, b: List[UInt8]
    ) raises -> Optional[AnyScalar]:
        """Decode one PLAIN-encoded min/max value to a typed scalar, mirroring the
        writer's encoding (`LittleEndian.fixed` reads `size_of[dt]` LE bytes and
        reinterprets — the inverse of the writer's byte emission). Returns None for
        types this reader does not yet decode (raw bytes stay in `read_metadata`).
        """
        var s = Span(b)
        if dtype == dt.int8:
            return Int8Scalar(
                LittleEndian.fixed[DType.int32](s, 0).cast[DType.int8]()
            ).to_any()
        elif dtype == dt.int16:
            return Int16Scalar(
                LittleEndian.fixed[DType.int32](s, 0).cast[DType.int16]()
            ).to_any()
        elif dtype == dt.int32:
            return Int32Scalar(LittleEndian.fixed[DType.int32](s, 0)).to_any()
        elif dtype == dt.uint8:
            return UInt8Scalar(
                LittleEndian.fixed[DType.uint32](s, 0).cast[DType.uint8]()
            ).to_any()
        elif dtype == dt.uint16:
            return UInt16Scalar(
                LittleEndian.fixed[DType.uint32](s, 0).cast[DType.uint16]()
            ).to_any()
        elif dtype == dt.uint32:
            return UInt32Scalar(LittleEndian.fixed[DType.uint32](s, 0)).to_any()
        elif dtype == dt.int64:
            return Int64Scalar(LittleEndian.fixed[DType.int64](s, 0)).to_any()
        elif dtype == dt.uint64:
            return UInt64Scalar(LittleEndian.fixed[DType.uint64](s, 0)).to_any()
        elif dtype == dt.float32:
            return Float32Scalar(
                LittleEndian.fixed[DType.float32](s, 0)
            ).to_any()
        elif dtype == dt.float64:
            return Float64Scalar(
                LittleEndian.fixed[DType.float64](s, 0)
            ).to_any()
        elif dtype == dt.float16:
            return Float16Scalar(
                LittleEndian.fixed[DType.float16](s, 0)
            ).to_any()
        elif dtype == dt.bool_:
            return BoolScalar(len(b) > 0 and b[0] != 0).to_any()
        elif dtype.is_string():
            return StringScalar(
                String(StringSlice(unsafe_from_utf8=Span(b)))
            ).to_any()
        # Temporal / small-decimal: physical INT32 / INT64, carrying the leaf's
        # unit / precision-scale so the scalar retags to the Arrow type.
        elif dtype.is_date32():
            return Date32Scalar(
                LittleEndian.fixed[DType.int32](s, 0), dt.date32()
            ).to_any()
        elif dtype.is_time32():
            return Time32Scalar(
                LittleEndian.fixed[DType.int32](s, 0), dtype.as_time32()
            ).to_any()
        elif dtype.is_time64():
            return Time64Scalar(
                LittleEndian.fixed[DType.int64](s, 0), dtype.as_time64()
            ).to_any()
        elif dtype.is_timestamp():
            return TimestampScalar(
                LittleEndian.fixed[DType.int64](s, 0), dtype.as_timestamp()
            ).to_any()
        elif dtype.is_decimal32():
            return Decimal32Scalar(
                LittleEndian.fixed[DType.int32](s, 0), dtype.as_decimal32()
            ).to_any()
        elif dtype.is_decimal64():
            return Decimal64Scalar(
                LittleEndian.fixed[DType.int64](s, 0), dtype.as_decimal64()
            ).to_any()
        # decimal128/256: big-endian two's-complement FIXED_LEN_BYTE_ARRAY.
        elif dtype.is_decimal128():
            return Decimal128Scalar(
                Plain.decode_be_flba[DType.int128](s, 0, len(b)),
                dtype.as_decimal128(),
            ).to_any()
        elif dtype.is_decimal256():
            return Decimal256Scalar(
                Plain.decode_be_flba[DType.int256](s, 0, len(b)),
                dtype.as_decimal256(),
            ).to_any()
        elif dtype.is_fixed_size_binary():
            return FixedSizeBinaryScalar(
                List[UInt8](s), dtype.as_fixed_size_binary().byte_width
            ).to_any()
        else:
            # binary / large_binary have no scalar type; the raw min/max bytes are
            # still available via `read_metadata`.
            return None
