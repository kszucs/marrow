"""Per-encoding decoding of a data page's *present* values.

Both the flat column path (`column.mojo`) and the nested list-element path
(`nested.mojo`) decode the same value encodings; keeping that logic here — one
primitive decoder, one byte-array decoder, one boolean decoder — means every
encoding (PLAIN, dictionary, DELTA_*, BYTE_STREAM_SPLIT) works identically in
both, instead of the two paths drifting apart.

Each decoder appends the `page.num_present` present values (nulls are placed
later by the caller from the definition levels).
"""

from std.sys import size_of

from .page import Page, read_fixed_le, read_u32le
from .encoding import (
    rle_decode,
    rle_gather,
    delta_binary_packed_decode,
    delta_decode,
)
from .format import (
    ENC_DELTA_BINARY_PACKED,
    ENC_DELTA_LENGTH_BYTE_ARRAY,
    ENC_DELTA_BYTE_ARRAY,
    ENC_BYTE_STREAM_SPLIT,
)


def decode_primitive_present[
    store: DType, phys: DType
](
    page: Page,
    dict: List[Scalar[store]],
    mut out: List[Scalar[store]],
) raises:
    """Append the present fixed-width values (widened `phys` -> `store`)."""
    comptime PW = size_of[Scalar[phys]]()
    var vspan = page.values()
    var np = page.num_present
    if page.is_plain():
        for i in range(np):
            out.append(read_fixed_le[phys](vspan, i * PW).cast[store]())
    elif page.is_dictionary():
        var base = len(out)
        out.resize(unsafe_uninit_length=base + np)
        rle_gather[store](
            vspan[1:],
            Int(vspan[0]),
            np,
            dict.unsafe_ptr(),
            out.unsafe_ptr() + base,
        )
    elif page.encoding == ENC_DELTA_BINARY_PACKED:
        var decoded = delta_binary_packed_decode(vspan, np)
        for i in range(np):
            out.append(decoded[i].cast[store]())
    elif page.encoding == ENC_BYTE_STREAM_SPLIT:
        # byte k of value i lives at vspan[k*np + i]
        for i in range(np):
            var raw = InlineArray[UInt8, PW](fill=0)

            comptime for k in range(PW):
                raw[k] = vspan[k * np + i]
            out.append(
                SIMD[phys, 1].from_bytes[big_endian=False](raw).cast[store]()
            )
    else:
        raise Error(
            "parquet: unsupported data page encoding " + String(page.encoding)
        )


def decode_bytes_present(
    page: Page,
    dict_body: List[UInt8],
    dict_off: List[Int],
    dict_len: List[Int],
) raises -> List[List[UInt8]]:
    """Return the present variable-length byte values."""
    var vspan = page.values()
    var np = page.num_present
    var out = List[List[UInt8]]()
    if page.is_plain():
        var vi = 0
        for _ in range(np):
            var n = read_u32le(vspan, vi)
            vi += 4
            var v = List[UInt8]()
            v.extend(vspan[vi : vi + n])
            out.append(v^)
            vi += n
    elif page.is_dictionary():
        var indices = rle_decode(vspan[1:], Int(vspan[0]), np)
        for i in range(np):
            var idx = Int(indices[i])
            var start = dict_off[idx]
            var v = List[UInt8]()
            v.extend(Span(dict_body)[start : start + dict_len[idx]])
            out.append(v^)
    elif page.encoding == ENC_DELTA_LENGTH_BYTE_ARRAY:
        var lengths = List[Int64]()
        var pos = delta_decode(vspan, 0, np, lengths)
        for i in range(np):
            var n = Int(lengths[i])
            var v = List[UInt8]()
            v.extend(vspan[pos : pos + n])
            out.append(v^)
            pos += n
    elif page.encoding == ENC_DELTA_BYTE_ARRAY:
        var prefixes = List[Int64]()
        var pos = delta_decode(vspan, 0, np, prefixes)
        var suffix_lens = List[Int64]()
        pos = delta_decode(vspan, pos, np, suffix_lens)
        var prev = List[UInt8]()
        for i in range(np):
            var v = List[UInt8]()
            v.extend(Span(prev)[0 : Int(prefixes[i])])
            var sl = Int(suffix_lens[i])
            v.extend(vspan[pos : pos + sl])
            pos += sl
            out.append(v.copy())
            prev = v^
    else:
        raise Error(
            "parquet: unsupported byte-array encoding " + String(page.encoding)
        )
    return out^


def decode_dict_primitive[
    store: DType, phys: DType
](page: Page, mut dict: List[Scalar[store]]) raises:
    """Decode a primitive dictionary page (PLAIN fixed-width) into `dict`."""
    comptime PW = size_of[Scalar[phys]]()
    var span = page.body
    for i in range(page.num_values):
        dict.append(read_fixed_le[phys](span, i * PW).cast[store]())


def decode_dict_bytes(
    page: Page,
    mut dict_body: List[UInt8],
    mut dict_off: List[Int],
    mut dict_len: List[Int],
) raises:
    """Decode a byte-array dictionary page (length-prefixed values)."""
    dict_body.clear()
    dict_body.extend(page.body)
    var span = Span(dict_body)
    var off = 0
    for _ in range(page.num_values):
        var n = read_u32le(span, off)
        off += 4
        dict_off.append(off)
        dict_len.append(n)
        off += n


def decode_bool_present(page: Page) raises -> List[Bool]:
    """Return the present PLAIN bit-packed booleans."""
    if not page.is_plain():
        raise Error("parquet: non-plain bool encoding not supported")
    var vspan = page.values()
    var out = List[Bool]()
    for i in range(page.num_present):
        var byte = vspan[i >> 3]
        out.append(((byte >> UInt8(i & 7)) & 1) == 1)
    return out^
