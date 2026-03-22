"""Hashing kernels for Arrow arrays.

Provides column-wise hash computation for use in groupby, joins, and
partitioning. Follows the DuckDB/DataFusion approach of hashing each
column independently and combining hashes across columns.

Public API:
  - ``hash_``: hash any array → PrimitiveArray[uint64]
    - PrimitiveArray[T]: per-element hash (AHasher)
    - StringArray: per-element hash
    - StructArray: per-column hash with combining (multi-key)
    - AnyArray: runtime-typed dispatch
  - ``hash_identity``: identity hash for small integer types (bool, uint8, int8)
    — returns values cast to uint64, zero hash overhead
"""

from std.hashlib import hash as _hash

from ..arrays import PrimitiveArray, StringArray, StructArray, AnyArray
from ..builders import PrimitiveBuilder
from ..dtypes import (
    DataType,
    uint8,
    int8,
    uint64,
    bool_,
    numeric_dtypes,
    primitive_dtypes,
)

comptime _h = Scalar[uint64.native]

comptime NULL_HASH_SENTINEL = UInt64(0x517CC1B727220A95)
"""Fixed hash value used for null elements."""


@always_inline
def _combine(existing: UInt64, new: UInt64) -> UInt64:
    """Combine two hash values (DuckDB/DataFusion polynomial combine)."""
    return (UInt64(17) * 37 + existing) * 37 + new


# ---------------------------------------------------------------------------
# hash_ — public API
# ---------------------------------------------------------------------------


def hash_[
    T: DataType
](keys: PrimitiveArray[T]) raises -> PrimitiveArray[uint64]:
    """Hash each element of a primitive array.

    Null elements hash to ``NULL_HASH_SENTINEL``.

    Args:
        keys: Input primitive array.

    Returns:
        A uint64 array of per-element hashes.
    """
    var n = len(keys)
    var builder = PrimitiveBuilder[uint64](capacity=n)
    var buf = keys.buffer
    var off = keys.offset
    var has_bitmap = Bool(keys.bitmap)

    for i in range(n):
        if has_bitmap and not keys.bitmap.value().is_valid(off + i):
            builder.append(_h(NULL_HASH_SENTINEL))
        else:
            builder.append(_h(_hash(buf.unsafe_get[T.native](off + i))))

    return builder.finish_typed()


def hash_(keys: StringArray) raises -> PrimitiveArray[uint64]:
    """Hash each element of a string array.

    Null elements hash to ``NULL_HASH_SENTINEL``.
    """
    var n = len(keys)
    var builder = PrimitiveBuilder[uint64](capacity=n)
    var has_bitmap = Bool(keys.bitmap)

    for i in range(n):
        if has_bitmap and not keys.bitmap.value().is_valid(keys.offset + i):
            builder.append(_h(NULL_HASH_SENTINEL))
        else:
            builder.append(_h(_hash(String(keys.unsafe_get(UInt(i))))))

    return builder.finish_typed()


def hash_(keys: StructArray) raises -> PrimitiveArray[uint64]:
    """Hash a struct array by combining per-field hashes column-wise.

    Each field (child array) is hashed independently via ``hash_(AnyArray)``
    and the results are combined element-wise. This is the natural
    representation for multi-key hashing — multiple groupby key columns
    are a struct array.

    Args:
        keys: Input struct array.

    Returns:
        A uint64 array where result[i] is the combined hash for row i.
    """
    var n = len(keys)
    var num_fields = len(keys.children)
    if num_fields == 0:
        raise Error("hash_: empty struct array")

    # Hash the first field to initialize.
    var result = hash_(keys.children[0])

    # TODO: it could be made more efficient be using vectorized
    # operations and probably have fixed width combine kernels
    # and use them to compose larger structs instead of doing
    # combining a single array at a time

    # Hash remaining fields and combine element-wise.
    for k in range(1, num_fields):
        var field_hashes = hash_(keys.children[k])
        var builder = PrimitiveBuilder[uint64](capacity=n)
        for i in range(n):
            builder.append(
                _h(
                    _combine(
                        UInt64(result.unsafe_get(i)),
                        UInt64(field_hashes.unsafe_get(i)),
                    )
                )
            )
        result = builder.finish_typed()

    return result^


def hash_(keys: AnyArray) raises -> PrimitiveArray[uint64]:
    """Runtime-typed hash: dispatches to the correct typed overload.

    Supports primitive, string, and struct arrays.

    Args:
        keys: Input array (runtime-typed).

    Returns:
        A uint64 array of per-element hashes.
    """
    if keys.dtype == bool_:
        return hash_[bool_](PrimitiveArray[bool_](data=keys))

    comptime for dtype in numeric_dtypes:
        if keys.dtype == dtype:
            return hash_[dtype](keys.as_primitive[dtype]())

    if keys.dtype.is_string():
        return hash_(keys.as_string())

    if keys.dtype.is_struct():
        return hash_(keys.as_struct())

    raise Error("hash_: unsupported dtype ", keys.dtype)


# ---------------------------------------------------------------------------
# hash_identity — identity hash for small integer types
# ---------------------------------------------------------------------------


def hash_identity[
    T: DataType
](keys: PrimitiveArray[T]) raises -> PrimitiveArray[uint64]:
    """Identity hash: returns values cast to uint64 with no hash overhead.

    For int8, values are offset by +128 to produce non-negative indices.
    Null elements map to ``NULL_HASH_SENTINEL``.

    Only valid for bool, uint8, and int8 — produces dense values in [0, 255].
    """
    comptime _OFFSET = 128 if T == int8 else 0

    var n = len(keys)
    var builder = PrimitiveBuilder[uint64](capacity=n)
    var has_bitmap = Bool(keys.bitmap)

    for i in range(n):
        if has_bitmap and not keys.bitmap.value().is_valid(keys.offset + i):
            builder.append(_h(NULL_HASH_SENTINEL))
        else:
            builder.append(_h(Int(keys.unsafe_get(i)) + _OFFSET))

    return builder.finish_typed()


def hash_identity(keys: AnyArray) raises -> PrimitiveArray[uint64]:
    """Runtime-typed identity hash dispatch."""
    if keys.dtype == bool_:
        return hash_identity[bool_](PrimitiveArray[bool_](data=keys))
    if keys.dtype == uint8:
        return hash_identity[uint8](keys.as_primitive[uint8]())
    if keys.dtype == int8:
        return hash_identity[int8](keys.as_primitive[int8]())
    raise Error("hash_identity: only supports bool, uint8, int8")
