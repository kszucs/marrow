# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Key identity — when two keys are the same key, and their hash.

Two kernels, kept together because they must agree:

- ``HashKernel`` — one 64-bit hash per row of a key column (or columns).
- ``KeyCompare`` — whether two rows hold the same key.

"The same key" is SQL's grouping identity, ``IS NOT DISTINCT FROM``: NULL
equals NULL, every NaN equals every NaN, ``-0.0`` equals ``0.0``, a
dictionary-encoded value equals its decoded value, and nested values compare
structurally. ``HashKernel`` canonicalises by the same rules, so equal keys
always hash equally; the converse is what ``KeyCompare`` is for. The join,
the group-by, ``is_in`` and window partitioning all read keys through these
two.

Each column is hashed independently and the hashes are combined across
columns.

Public API — ``HashKernel[H]``, generic over the ``Hasher``:
  - ``apply``: typed leaves
    - BoolArray: vectorized via precomputed hash + SIMD select
    - PrimitiveArray[T]: vectorized ``H.hash_lanes`` over each value widened
      to a 64-bit word; values wider than 64 bits (decimal128/256) fold their
      64-bit limbs
    - BytesArray (either string layout): ``H.hash`` of each value's bytes
    - StructArray: per-column hash with combining (multi-key)
    - ListLikeArray[T] / FixedSizeListArray: fold the child hashes per row
  - ``dispatch``: runtime-typed dispatch, routed through the
    ``DynType.dispatch_*`` family rather than a hand-written dtype ladder.
    Temporal, interval, and decimal32/64 columns are hashed through their
    typed leaf (bound on ``PrimitiveType``); dictionary columns through their
    decoded values,
    so a dictionary-encoded key hashes identically to the plain column.

**The engine instantiates it once, at ``Fmix64``** — a bijection of each word,
rapidhash for bytes: the key hash (``KeyHash``) of the group-by, ``is_in``, the
join and ``count_distinct``, and ``approx_count_distinct``'s sketch hash.
"""


from std.math import isnan
from std.sys import size_of
from std.sys.info import simd_byte_width
from std.utils.numerics import nan

from ..arrays import (
    BoolArray,
    BinaryLikeArray,
    BinaryViewLikeArray,
    BytesArray,
    FixedSizeBinaryArray,
    PrimitiveArray,
    StructArray,
    ListLikeArray,
    FixedSizeListArray,
    DictionaryArray,
    DynArray,
    UInt64Array,
    Int32Array,
)
from ..builders import UInt64Builder, Int32Builder, arange
from ..buffers import Bitmap, Buffer
from ..views import BufferView, apply
from .cast import decode_dictionary
from .numeric import EqKernel
from .core import Kernel
from ..utils import Hasher, RapidHash64
from ..execution import ExecContext, GPU_ENABLED
from ..errors import InvalidError, TypeError
from ..dtypes import (
    BinaryLikeType,
    BinaryViewLikeType,
    Int32Type,
    IntegerType,
    PrimitiveType,
    ListLikeType,
    bool_,
    uint64,
)


comptime NULL_HASH_SENTINEL = UInt64(0x517CC1B727220A95)
"""Fixed hash value used for null elements."""


# ---------------------------------------------------------------------------
# HashKernel — one hash per row, vectorized for primitive arrays
struct HashKernel[H: Hasher](Kernel):
    """Column hashing kernel — one ``UInt64`` per row, under the hash `H`.

    `H` is comptime, so every call below resolves to a direct `H.hash_lanes` /
    `H.hash` and the generated code is what a hand-written kernel for that one
    algorithm would be. The engine instantiates it at `Fmix64` (see the module
    docstring); benchmarks pass the other hashers in `utils/hashing.mojo`, so
    the choice is a parameter rather than a rewrite.

    The typed leaves are the ``apply`` overloads; ``dispatch`` resolves a
    runtime-typed array to the matching leaf via the ``DynType.dispatch_*`` family
    rather than a per-dtype ladder, so adding a dtype to a family covers it
    without touching this file. Null elements hash to ``NULL_HASH_SENTINEL`` so
    that "null == null" holds for grouping and joining (Arrow's `hash_*`
    semantics).

    Not hashable: `null` and `fixed_size_binary` (no leaf yet). The 16-byte
    month-day-nano interval hashes like decimal128, by folding its limbs.
    """

    comptime name = Self.H.name

    # --- lane helpers ---------------------------------------------------
    #
    # Static methods rather than free functions: they are meaningless without
    # `H`, and `apply` accepts `Self._x[...]` as its comptime function exactly
    # as `BinaryKernel.apply` passes `Self.core[native, _]`. The two that merely
    # forwarded to the trait (`_hash_bits`, `_combine_hashes`) are gone — the
    # call sites say `Self.H.hash_lanes` / `Self.H.combine_lanes` directly.

    @staticmethod
    @always_inline
    def _bool_lanes[
        W: Int
    ](bits: SIMD[DType.bool, W]) -> SIMD[uint64.native, W]:
        """Select between the two precomputed bool digests."""
        comptime bw = size_of[Scalar[bool_.native]]()
        comptime h_false = Self.H.hash_lanes[bw, 1](SIMD[uint64.native, 1](0))[
            0
        ]
        comptime h_true = Self.H.hash_lanes[bw, 1](SIMD[uint64.native, 1](1))[0]
        return bits.select(
            SIMD[uint64.native, W](h_true), SIMD[uint64.native, W](h_false)
        )

    @staticmethod
    @always_inline
    def _bool_lanes_masked[
        W: Int
    ](bits: SIMD[DType.bool, W], valid: SIMD[DType.bool, W]) -> SIMD[
        uint64.native, W
    ]:
        return valid.select(
            Self._bool_lanes[W](bits),
            SIMD[uint64.native, W](NULL_HASH_SENTINEL),
        )

    @staticmethod
    @always_inline
    def _floating_lanes[
        T: PrimitiveType, W: Int
    ](vals: SIMD[T.native, W]) -> SIMD[uint64.native, W]:
        """Hash a float lane by its **bit pattern**, not by its value.

        `cast[uint64.native]()` is a numeric conversion: every float in
        (-1, 1) truncates to 0, so `-1.25` and `0.5` produced one hash — and
        grouping then bucketed on the hash alone, so they were one group
        (`golden/cases/group_by_float_key.mojo`). Grouping compares keys now,
        but a hash that collides on a whole interval would still make every
        probe a collision.

        Two values are canonicalised first, because "same number" and "same
        bits" disagree about them. `-0.0` equals `0.0` and must group with it;
        `+ 0.0` says that in one instruction, since adding zero is exact for
        every float and `-0.0 + 0.0` is `+0.0`. And NaN carries a payload
        chosen by whatever arithmetic produced it, so every NaN folds onto one
        pattern.

        `to_bits` bitcasts to the same-width unsigned type and widens, so the
        zero-extension `_integer_lanes` masks for is already done.
        """
        comptime byte_width = size_of[Scalar[T.native]]()
        var zeroed = vals + SIMD[T.native, W](0)
        var canonical = isnan(vals).select(
            SIMD[T.native, W](nan[T.native]()), zeroed
        )
        return Self.H.hash_lanes[byte_width, W](
            canonical.to_bits[uint64.native]()
        )

    @staticmethod
    @always_inline
    def _integer_lanes[
        T: PrimitiveType, W: Int
    ](vals: SIMD[T.native, W]) -> SIMD[uint64.native, W]:
        """Widen an integral lane to uint64 and hash it.

        Two widths, because one of them does not fit: decimal128/256 and the
        month-day-nano interval are wider than a uint64 lane, so their 64-bit
        limbs are folded instead. A truncating cast would collide every pair
        differing only above bit 63 — every such probe a collision to resolve.
        """
        comptime byte_width = size_of[Scalar[T.native]]()

        comptime if byte_width <= 8:
            # Zero-extend (matches C's rapid_read32/rapid_read64); mask to
            # byte_width bits so a narrow signed type does not sign-extend.
            comptime mask = ~UInt64(0) if byte_width >= 8 else (
                UInt64(1) << UInt64(byte_width * 8)
            ) - 1
            return Self.H.hash_lanes[byte_width, W](
                vals.cast[uint64.native]() & SIMD[uint64.native, W](mask)
            )
        else:
            comptime limbs = byte_width // 8
            var h = SIMD[uint64.native, W](0)
            comptime for i in range(limbs):
                var limb = (vals >> SIMD[T.native, W](i * 64)).cast[
                    uint64.native
                ]()
                h = Self.H.combine_lanes[W](h, Self.H.hash_lanes[8, W](limb))
            return h

    @staticmethod
    @always_inline
    def _primitive_lanes[
        T: PrimitiveType, W: Int
    ](vals: SIMD[T.native, W]) -> SIMD[uint64.native, W]:
        """Whichever of the two above `T` calls for.

        **A dispatcher rather than two overloads**, because both were written
        and the pair does not compile. `where T.native.is_floating_point()`
        parses, and then every call site reports *"ambiguous call ... cannot
        prove constraint for candidate"* — the constraint solver will not
        evaluate a non-builtin function, and `DType.is_floating_point` is a
        stdlib method.

        **And a dispatcher rather than branching at the call sites**, because
        two of the three take this as a comptime function *reference*
        (`Self._primitive_lanes[T, ...]`, handed to `apply`), and a reference
        cannot name an overload set or be chosen by a branch without
        duplicating the call it is passed to. The branch belongs in one place,
        and it costs nothing: each instantiation keeps one arm.
        """
        comptime if T.native.is_floating_point():
            return Self._floating_lanes[T, W](vals)
        else:
            return Self._integer_lanes[T, W](vals)

    @staticmethod
    @always_inline
    def _primitive_lanes_masked[
        T: PrimitiveType, W: Int
    ](vals: SIMD[T.native, W], valid: SIMD[DType.bool, W]) -> SIMD[
        uint64.native, W
    ]:
        return valid.select(
            Self._primitive_lanes[T, W](vals),
            SIMD[uint64.native, W](NULL_HASH_SENTINEL),
        )

    @staticmethod
    def dispatch(
        keys: DynArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> UInt64Array:
        """Resolve `keys`'s runtime dtype and hash it."""
        var dt = keys.dtype()
        if dt == bool_:
            return Self.apply(keys.as_bool(), ctx)
        elif dt.is_binary_like() or dt.is_string_view() or dt.is_binary_view():

            def bytes_leaf[A: BytesArray](arr: A) raises {imm} -> UInt64Array:
                return Self.apply(arr, ctx)

            return keys.dispatch_bytes(bytes_leaf)
        elif dt.is_list_like():

            def listlike[T: ListLikeType](d: T) raises {imm} -> UInt64Array:
                return Self.apply(keys.as_list_like[T](), ctx)

            return dt.dispatch_listlike(listlike)
        elif dt.is_struct():
            return Self.apply(keys.as_struct(), ctx)
        elif dt.is_fixed_size_list():
            return Self.apply(keys.as_fixed_size_list(), ctx)
        elif dt.is_dictionary():
            # Hash the decoded values: a dictionary-encoded key must hash the
            # same as the equivalent plain column, otherwise two batches with
            # different dictionaries would never group together.
            return Self.dispatch(
                decode_dictionary(keys.as_dictionary(), ctx), ctx
            )
        elif dt.is_primitive():
            # One arm for every fixed-width type. `apply` is bound on
            # `PrimitiveType`, so numeric, temporal, interval and *all four*
            # decimal widths hash through the typed leaf directly — the hash
            # only reads the value bytes via `T.native`, never the logical
            # dtype. No reinterpret to an integer backing is needed.
            #
            # `Decimal128Array` is `PrimitiveArray[Decimal128Type]`, so the
            # separate numeric/decimal128/decimal256 arms this replaces were
            # three more spellings of this one call.
            def primitive[T: PrimitiveType](d: T) raises {imm} -> UInt64Array:
                return Self.apply(keys.as_primitive[T](), ctx)

            return dt.dispatch_primitive(primitive)
        else:
            raise Self.error[TypeError](String("unsupported dtype ", dt))

    @staticmethod
    def apply(
        keys: BoolArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> UInt64Array:
        """Vectorized ``H`` for bool arrays.

        Precomputes hash(false) and hash(true), loads data bits via the
        bitmap-mask pattern, and uses ``SIMD.select()`` for branchless dispatch.
        Null elements are replaced with ``NULL_HASH_SENTINEL`` inline.

        Parallelism is delegated to ``apply`` via the ``ExecContext`` —
        no per-kernel stripe logic here.
        """
        var n = len(keys)
        var buf: Buffer[mut=True]
        comptime if GPU_ENABLED:
            if ctx.is_gpu():
                buf = Buffer.alloc_device[uint64.native](ctx.device.value(), n)
            else:
                buf = Buffer.alloc_uninit[uint64.native](n)
        else:
            buf = Buffer.alloc_uninit[uint64.native](n)

        var dst = buf.view[uint64.native](0, n)
        var validity = keys.validity()
        if validity:
            apply[uint64.native, Self._bool_lanes_masked[...]](
                keys.values(),
                validity.value(),
                dst,
                ctx,
            )
        else:
            apply[uint64.native, Self._bool_lanes[...]](keys.values(), dst, ctx)

        return UInt64Array(
            length=n,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=buf.to_immutable(),
        )

    # FIXME: use the seeding from the Rust implementation
    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        keys: PrimitiveArray[T],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> UInt64Array:
        """Vectorized ``H`` for primitive arrays.

        Each SIMD lane independently computes the hash of one element.
        Null elements are replaced with ``NULL_HASH_SENTINEL`` inline.

        Parallelism is handled uniformly by ``apply`` using the
        ``ExecContext`` — CPU vs GPU is picked from ``ctx.device``, and
        CPU stripe-parallelism is driven by ``ctx.num_threads``.

        Values wider than a uint64 lane (decimal128/decimal256) have no SIMD
        path; they run the same limb-folding core one element at a time.
        """
        var n = len(keys)

        comptime if size_of[Scalar[T.native]]() > 8:
            var values = keys.values()
            var buf = Buffer.alloc_uninit[uint64.native](max(n, 1))
            for i in range(n):
                if keys.is_valid(i):
                    buf.unsafe_set[uint64.native](
                        i, Self._primitive_lanes[T, 1](values.unsafe_get(i))
                    )
                else:
                    buf.unsafe_set[uint64.native](i, NULL_HASH_SENTINEL)
            return UInt64Array(
                length=n,
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=buf^.to_immutable(),
            )
        else:
            var buf: Buffer[mut=True]
            comptime if GPU_ENABLED:
                if ctx.is_gpu():
                    buf = Buffer.alloc_device[uint64.native](
                        ctx.device.value(), n
                    )
                else:
                    buf = Buffer.alloc_uninit[uint64.native](n)
            else:
                buf = Buffer.alloc_uninit[uint64.native](n)

            var dst = buf.view[uint64.native](0, n)
            var validity = keys.validity()
            if validity:
                apply[
                    T.native,
                    uint64.native,
                    Self._primitive_lanes_masked[T, ...],
                ](
                    keys.values(),
                    validity.value(),
                    dst,
                    ctx,
                )
            else:
                apply[T.native, uint64.native, Self._primitive_lanes[T, ...]](
                    keys.values(),
                    dst,
                    ctx,
                )

            return UInt64Array(
                length=n,
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=buf.to_immutable(),
            )

    @staticmethod
    def apply[
        A: BytesArray
    ](keys: A, ctx: ExecContext = ExecContext.serial()) raises -> UInt64Array:
        """Hash each element of a byte-string array, in either layout (string,
        large_string, binary, large_binary, string_view, binary_view).

        The bytes are hashed, not the layout, so a value hashes the same in
        every one of them. The key columns of a join or group-by still have
        to agree on a dtype: `DictionaryEncoder` checks that before encoding.
        Hashed with `H` like every other leaf. This used
        to call `std.hashlib.hash` (aHash) regardless of `H`, because the
        multi-branch byte-string path did not exist; a string column and a
        numeric column were therefore hashed by different algorithms. `H.hash`
        is that path.

        Currently scalar-serial; parallelising variable-length hashing is future
        work — the ``ctx`` parameter exists for API consistency.
        """
        _ = ctx  # TODO: SIMD + parallel string hashing
        var n = len(keys)
        var buf = Buffer.alloc_uninit[uint64.native](max(n, 1))
        var has_nulls = keys.null_count() > 0

        for i in range(n):
            if has_nulls and not keys.is_valid(i):
                buf.unsafe_set[uint64.native](i, NULL_HASH_SENTINEL)
            else:
                buf.unsafe_set[uint64.native](
                    i, Self.H.hash(keys.unsafe_get(UInt(i)).as_bytes())
                )

        return UInt64Array(
            length=n,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=buf^.to_immutable(),
        )

    @staticmethod
    def apply(
        keys: StructArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> UInt64Array:
        """Hash a struct array by combining per-field hashes column-wise.

        Each field is hashed independently via ``dispatch`` and the results are
        combined element-wise using ``H.combine_lanes``; ``ctx`` stripes both.
        A NULL struct hashes to ``NULL_HASH_SENTINEL`` whatever its fields
        hold, since it is one key whatever they hold.
        """
        var n = len(keys)
        var num_fields = len(keys.children)
        if num_fields == 0:
            raise Self.error[InvalidError]("empty struct array")
        var result = Self.dispatch(keys.field(0), ctx)
        for k in range(1, num_fields):
            var field_hashes = Self.dispatch(keys.field(k), ctx)
            var buf: Buffer[mut=True]
            comptime if GPU_ENABLED:
                if ctx.is_gpu():
                    buf = Buffer.alloc_device[uint64.native](
                        ctx.device.value(), n
                    )
                else:
                    buf = Buffer.alloc_uninit[uint64.native](n)
            else:
                buf = Buffer.alloc_uninit[uint64.native](n)
            apply[uint64.native, uint64.native, Self.H.combine_lanes[...]](
                result.values(),
                field_hashes.values(),
                buf.view[uint64.native](0, n),
                ctx,
            )
            result = UInt64Array(
                length=n,
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=buf.to_immutable(),
            )
        if keys.null_count() != 0:
            var buf = Buffer.alloc_uninit[uint64.native](n)
            var out = buf.view[uint64.native](0, n)
            for i in range(n):
                out.store[1](
                    i,
                    result.unsafe_get(i) if keys.is_valid(
                        i
                    ) else NULL_HASH_SENTINEL,
                )
            result = UInt64Array(
                length=n,
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=buf^.to_immutable(),
            )
        return result^

    @staticmethod
    def apply[
        T: ListLikeType
    ](
        keys: ListLikeArray[T],
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> UInt64Array:
        """Hash a list/large-list/map array row-wise: hash the whole child once,
        then fold the child hashes over each row's element range (column-wise,
        no row-encoding). Null rows hash to ``NULL_HASH_SENTINEL``."""
        var n = len(keys)
        var child_hashes = Self.dispatch(keys.values().copy(), ctx)
        var builder = UInt64Builder(n)
        for i in range(n):
            if not keys.is_valid(i):
                builder.append(NULL_HASH_SENTINEL)
            else:
                var h = UInt64(0)
                var rng = keys.child_range(i)
                for j in range(rng[0], rng[1]):
                    h = Self.H.combine_lanes[1](h, child_hashes.unsafe_get(j))
                builder.append(h)
        return builder.finish()

    @staticmethod
    def apply(
        keys: FixedSizeListArray,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> UInt64Array:
        """Hash a fixed-size-list array row-wise: hash the child once, fold each
        row's ``list_size`` child hashes. Null rows hash to
        ``NULL_HASH_SENTINEL``."""
        var n = len(keys)
        var size = keys.dtype.as_fixed_size_list().size
        var child_hashes = Self.dispatch(keys.values().copy(), ctx)
        var builder = UInt64Builder(n)
        for i in range(n):
            if not keys.is_valid(i):
                builder.append(NULL_HASH_SENTINEL)
            else:
                var h = UInt64(0)
                var base = (keys.offset + i) * size
                for j in range(size):
                    h = Self.H.combine_lanes[1](
                        h, child_hashes.unsafe_get(base + j)
                    )
                builder.append(h)
        return builder.finish()


comptime RapidHashKernel = HashKernel[RapidHash64]
"""Rapidhash v3 lanes, for benchmarks and tests that name the algorithm — the
engine hashes with `HashKernel[Fmix64]`."""


# ---------------------------------------------------------------------------
# KeyCompare — key equality, the other half of key identity
# ---------------------------------------------------------------------------


@always_inline
def _bytes_equal[
    LO: DType, RO: DType
](
    left_offsets: BufferView[LO, _],
    left_bytes: BufferView[DType.uint8, _],
    a: Int,
    right_offsets: BufferView[RO, _],
    right_bytes: BufferView[DType.uint8, _],
    b: Int,
) -> Bool:
    """Whether byte string ``a`` on the left is byte string ``b`` on the
    right, each read off an offsets view and a bytes view: lengths first, then
    the bytes, widest loads first and never past a value's last byte.
    """
    var x = Int(left_offsets.load[1](a))
    var y = Int(right_offsets.load[1](b))
    var length = Int(left_offsets.load[1](a + 1)) - x
    if length != Int(right_offsets.load[1](b + 1)) - y:
        return False
    var k = 0
    while k + 16 <= length:
        if left_bytes.load[16](x + k) != right_bytes.load[16](y + k):
            return False
        k += 16
    if k + 8 <= length:
        if left_bytes.load[8](x + k) != right_bytes.load[8](y + k):
            return False
        k += 8
    if k + 4 <= length:
        if left_bytes.load[4](x + k) != right_bytes.load[4](y + k):
            return False
        k += 4
    while k < length:
        if left_bytes.load[1](x + k) != right_bytes.load[1](y + k):
            return False
        k += 1
    return True


struct KeyCompare(Kernel):
    """Whether rows hold the same key — ``IS NOT DISTINCT FROM`` (see the
    module docstring), the equality ``HashKernel`` hashes by.

    Compares ``left[left_rows[i]]`` with ``right[right_rows[i]]`` and clears
    ``equal[i]`` where they differ. **Rows whose bit is already clear are
    skipped**, so a multi-column key is one call per column into one bitmap,
    each column looking only at the rows every earlier one kept. **Both sides
    are indexed**, so comparing a batch
    against stored keys by id copies neither.

    It compares every type ``HashKernel`` hashes, plus null and fixed-size
    binary. The fixed-width and binary leaves stripe over ``ctx`` in
    64-row-aligned stripes, so no two workers write one byte of ``equal``; the
    nested leaves compare their children through the same entry point.
    """

    comptime name = "key_compare"

    @staticmethod
    def equals(left: DynArray, right: DynArray) raises -> Bool:
        """Whether two arrays of one dtype and length hold the same key at
        every row, whatever their offsets and validity bitmaps."""
        var n = len(left)
        if left.dtype() != right.dtype() or n != len(right):
            return False
        var equal = Bitmap.alloc_zeroed(n)
        equal.set_range(0, n, True)
        var rows = arange[Int32Type](0, n)
        Self.apply(left, rows, right, rows, equal)
        return equal.view().all_set()

    @staticmethod
    def apply(
        left: DynArray,
        left_rows: Int32Array,
        right: DynArray,
        right_rows: Int32Array,
        mut equal: Bitmap[mut=True],
        ctx: ExecContext = ExecContext.serial(),
    ) raises:
        """Resolve both sides' runtime dtype and compare."""
        var n = len(left_rows)
        Self.expect_same_length(len(right_rows), n)
        if len(equal) < n:
            raise Self.error[InvalidError](
                String("result has ", len(equal), " bits for ", n, " rows")
            )
        var dt = left.dtype()
        Self.expect_same_dtype(dt, right.dtype())
        if dt.is_null():
            # Every row is NULL on both sides, and NULL is one key.
            pass
        elif dt.is_bool():
            Self.apply(
                left.as_bool(), left_rows, right.as_bool(), right_rows, equal
            )
        elif dt.is_primitive():

            def primitive[T: PrimitiveType](d: T) raises {mut equal, imm}:
                Self.apply(
                    left.as_primitive[T](),
                    left_rows,
                    right.as_primitive[T](),
                    right_rows,
                    equal,
                    ctx,
                )

            dt.dispatch_primitive(primitive)
        elif dt.is_binary_like():

            def binarylike[T: BinaryLikeType](d: T) raises {mut equal, imm}:
                Self.apply(
                    left.as_binary_like[T](),
                    left_rows,
                    right.as_binary_like[T](),
                    right_rows,
                    equal,
                    ctx,
                )

            dt.dispatch_binarylike(binarylike)
        elif dt.is_string_view() or dt.is_binary_view():

            def viewlike[T: BinaryViewLikeType](d: T) raises {mut equal, imm}:
                Self.apply(
                    left.as_binary_view_like[T](),
                    left_rows,
                    right.as_binary_view_like[T](),
                    right_rows,
                    equal,
                    ctx,
                )

            dt.dispatch_binaryview(viewlike)
        elif dt.is_fixed_size_binary():
            Self.apply(
                left.as_fixed_size_binary(),
                left_rows,
                right.as_fixed_size_binary(),
                right_rows,
                equal,
                ctx,
            )
        elif dt.is_dictionary():
            # By decoded value: two arrays of one column may carry different
            # dictionaries, and the hash is taken over decoded values too.
            Self.apply(
                decode_dictionary(left.as_dictionary(), ctx),
                left_rows,
                decode_dictionary(right.as_dictionary(), ctx),
                right_rows,
                equal,
                ctx,
            )
        elif dt.is_struct():
            Self.apply(
                left.as_struct(),
                left_rows,
                right.as_struct(),
                right_rows,
                equal,
                ctx,
            )
        elif dt.is_list_like():

            def listlike[T: ListLikeType](d: T) raises {mut equal, imm}:
                Self.apply(
                    left.as_list_like[T](),
                    left_rows,
                    right.as_list_like[T](),
                    right_rows,
                    equal,
                    ctx,
                )

            dt.dispatch_listlike(listlike)
        elif dt.is_fixed_size_list():
            Self.apply(
                left.as_fixed_size_list(),
                left_rows,
                right.as_fixed_size_list(),
                right_rows,
                equal,
                ctx,
            )
        else:
            raise Self.error[TypeError](String("unsupported dtype ", dt))

    # --- typed leaves -------------------------------------------------------

    @staticmethod
    def apply[
        T: PrimitiveType
    ](
        left: PrimitiveArray[T],
        left_rows: Int32Array,
        right: PrimitiveArray[T],
        right_rows: Int32Array,
        mut equal: Bitmap[mut=True],
        ctx: ExecContext,
    ):
        """Fixed-width values: a gather and compare per row still equal, the
        validity tests hoisted out when neither side has a null.

        Without nulls, a value of at most 8 bytes is compared ``W`` rows at a
        time: the rows' bits are loaded, a group none of whose rows is still
        equal is skipped, and otherwise both sides are gathered and the
        comparison ANDed back. ``W`` is a multiple of 8, so every store is
        whole bytes, and stripes are 64-aligned, so no two workers share one.
        """
        var l = left.values()
        var r = right.values()
        var li = left_rows.values()
        var ri = right_rows.values()
        var nullable = left.null_count() != 0 or right.null_count() != 0
        comptime W = min(
            32, max(8, simd_byte_width() // size_of[Scalar[T.native]]())
        )
        comptime vectorized = size_of[Scalar[T.native]]() <= 8
        # `-0.0 == 0.0` under IEEE already; `nan_safe` adds NaN equal to NaN.
        comptime Same = EqKernel[nan_safe=True]

        @always_inline
        def same(a: Scalar[T.native], b: Scalar[T.native]) -> Bool:
            return Bool(Same.core[T.native, 1](a, b))

        def body(wid: Int, start: Int, end: Int) {mut equal, imm}:
            if not nullable:
                var i = start
                comptime if vectorized:
                    var bits = equal.view()
                    while i + W <= end:
                        var keep = bits.load[W](i)
                        if keep.reduce_or():
                            var a = l.gather[W](
                                li.load[W](i).cast[DType.int64]()
                            )
                            var b = r.gather[W](
                                ri.load[W](i).cast[DType.int64]()
                            )
                            bits.store[W](
                                i, keep & Same.core[T.native, W](a, b)
                            )
                        i += W
                while i < end:
                    if equal.unsafe_test(i) and not same(
                        l.load[1](Int(li.load[1](i))),
                        r.load[1](Int(ri.load[1](i))),
                    ):
                        equal.unsafe_clear(i)
                    i += 1
            else:
                for i in range(start, end):
                    if equal.unsafe_test(i):
                        var a = Int(li.load[1](i))
                        var b = Int(ri.load[1](i))
                        var valid = left.is_valid(a)
                        if valid != right.is_valid(b) or (
                            valid and not same(l.load[1](a), r.load[1](b))
                        ):
                            equal.unsafe_clear(i)

        ctx.stripe(len(left_rows), body, align=64)

    @staticmethod
    def apply[
        T: BinaryLikeType
    ](
        left: BinaryLikeArray[T],
        left_rows: Int32Array,
        right: BinaryLikeArray[T],
        right_rows: Int32Array,
        mut equal: Bitmap[mut=True],
        ctx: ExecContext,
    ):
        """Variable-width values: lengths from the offsets first, then the
        bytes, widest loads first and never past a value's last byte."""
        # Not `left.unsafe_get(a) == right.unsafe_get(b)`: `unsafe_get` slices
        # the values buffer for every row, measurably slower here.
        var li = left_rows.values()
        var ri = right_rows.values()
        var nullable = left.null_count() != 0 or right.null_count() != 0
        var lo = left.offsets.view[T.offset](left.offset)
        var ro = right.offsets.view[T.offset](right.offset)
        var lb = left.values.view[DType.uint8]()
        var rb = right.values.view[DType.uint8]()

        def body(wid: Int, start: Int, end: Int) {mut equal, imm}:
            for i in range(start, end):
                if equal.unsafe_test(i):
                    var a = Int(li.load[1](i))
                    var b = Int(ri.load[1](i))
                    if nullable:
                        var valid = left.is_valid(a)
                        if valid != right.is_valid(b):
                            equal.unsafe_clear(i)
                            continue
                        if not valid:
                            continue
                    if not _bytes_equal(lo, lb, a, ro, rb, b):
                        equal.unsafe_clear(i)

        ctx.stripe(len(left_rows), body, align=64)

    @staticmethod
    def apply[
        T: BinaryViewLikeType
    ](
        left: BinaryViewLikeArray[T],
        left_rows: Int32Array,
        right: BinaryViewLikeArray[T],
        right_rows: Int32Array,
        mut equal: Bitmap[mut=True],
        ctx: ExecContext,
    ):
        """Views: every view carries its length and first four bytes, so a
        pair that differs in either is settled without reading a buffer; only
        a pair agreeing on both compares its bytes."""
        var li = left_rows.values()
        var ri = right_rows.values()
        var nullable = left.null_count() != 0 or right.null_count() != 0

        def body(wid: Int, start: Int, end: Int) {mut equal, imm}:
            for i in range(start, end):
                if equal.unsafe_test(i):
                    var a = Int(li.load[1](i))
                    var b = Int(ri.load[1](i))
                    if nullable:
                        var valid = left.is_valid(a)
                        if valid != right.is_valid(b):
                            equal.unsafe_clear(i)
                            continue
                        if not valid:
                            continue
                    if (
                        left.view_length(a) != right.view_length(b)
                        or left.view_prefix(a) != right.view_prefix(b)
                        or left.unsafe_get(UInt(a)) != right.unsafe_get(UInt(b))
                    ):
                        equal.unsafe_clear(i)

        ctx.stripe(len(left_rows), body, align=64)

    @staticmethod
    def apply(
        left: FixedSizeBinaryArray,
        left_rows: Int32Array,
        right: FixedSizeBinaryArray,
        right_rows: Int32Array,
        mut equal: Bitmap[mut=True],
        ctx: ExecContext,
    ):
        """Fixed-width bytes: no lengths to compare, only the bytes."""
        var li = left_rows.values()
        var ri = right_rows.values()
        var width = left.byte_width
        var lb = left.buffer.view[DType.uint8]()
        var rb = right.buffer.view[DType.uint8]()

        def body(wid: Int, start: Int, end: Int) {mut equal, imm}:
            for i in range(start, end):
                if equal.unsafe_test(i):
                    var a = Int(li.load[1](i))
                    var b = Int(ri.load[1](i))
                    var valid = left.is_valid(a)
                    if valid != right.is_valid(b):
                        equal.unsafe_clear(i)
                    elif valid:
                        var x = (left.offset + a) * width
                        var y = (right.offset + b) * width
                        for k in range(width):
                            if lb.load[1](x + k) != rb.load[1](y + k):
                                equal.unsafe_clear(i)
                                break

        ctx.stripe(len(left_rows), body, align=64)

    @staticmethod
    def apply(
        left: BoolArray,
        left_rows: Int32Array,
        right: BoolArray,
        right_rows: Int32Array,
        mut equal: Bitmap[mut=True],
    ):
        for i in range(len(left_rows)):
            if equal.test(i):
                var a = Int(left_rows.unsafe_get(i))
                var b = Int(right_rows.unsafe_get(i))
                var valid = left.is_valid(a)
                if valid != right.is_valid(b) or (
                    valid and left.values().test(a) != right.values().test(b)
                ):
                    equal.clear(i)

    @staticmethod
    def apply(
        left: StructArray,
        left_rows: Int32Array,
        right: StructArray,
        right_rows: Int32Array,
        mut equal: Bitmap[mut=True],
        ctx: ExecContext,
    ) raises:
        """Structs field by field; a NULL struct is one key whatever its fields
        hold, so only rows valid on both sides reach the fields."""
        if left.null_count() == 0 and right.null_count() == 0:
            for k in range(len(left.children)):
                Self.apply(
                    left.field(k),
                    left_rows,
                    right.field(k),
                    right_rows,
                    equal,
                    ctx,
                )
            return
        var n = len(left_rows)
        var fields_equal = Bitmap.alloc_zeroed(n)
        for i in range(n):
            if equal.test(i):
                var valid = left.is_valid(Int(left_rows.unsafe_get(i)))
                if valid != right.is_valid(Int(right_rows.unsafe_get(i))):
                    equal.clear(i)
                elif valid:
                    fields_equal.set(i)
        # A struct's offset is not propagated to its children by the layout.
        for k in range(len(left.children)):
            Self.apply(
                left.children[k].slice(left.offset, len(left)),
                left_rows,
                right.children[k].slice(right.offset, len(right)),
                right_rows,
                fields_equal,
                ctx,
            )
        for i in range(n):
            if (
                equal.test(i)
                and not fields_equal.test(i)
                and left.is_valid(Int(left_rows.unsafe_get(i)))
            ):
                equal.clear(i)

    @staticmethod
    def apply[
        T: ListLikeType
    ](
        left: ListLikeArray[T],
        left_rows: Int32Array,
        right: ListLikeArray[T],
        right_rows: Int32Array,
        mut equal: Bitmap[mut=True],
        ctx: ExecContext,
    ) raises:
        """Lists by validity and length, then every element pair of the rows
        still equal, compared as one flattened batch of the child arrays."""
        var elems_left = Int32Builder()
        var elems_right = Int32Builder()
        var owner = List[Int]()
        for i in range(len(left_rows)):
            if equal.test(i):
                var a = Int(left_rows.unsafe_get(i))
                var b = Int(right_rows.unsafe_get(i))
                var valid = left.is_valid(a)
                if valid != right.is_valid(b):
                    equal.clear(i)
                elif valid:
                    var x = left.child_range(a)
                    var y = right.child_range(b)
                    if x[1] - x[0] != y[1] - y[0]:
                        equal.clear(i)
                    else:
                        for j in range(x[1] - x[0]):
                            elems_left.append(Int32(x[0] + j))
                            elems_right.append(Int32(y[0] + j))
                            owner.append(i)
        Self._compare_elements(
            left.values(),
            elems_left.finish(),
            right.values(),
            elems_right.finish(),
            owner,
            equal,
            ctx,
        )

    @staticmethod
    def apply(
        left: FixedSizeListArray,
        left_rows: Int32Array,
        right: FixedSizeListArray,
        right_rows: Int32Array,
        mut equal: Bitmap[mut=True],
        ctx: ExecContext,
    ) raises:
        var size = left.dtype.as_fixed_size_list().size
        var elems_left = Int32Builder()
        var elems_right = Int32Builder()
        var owner = List[Int]()
        for i in range(len(left_rows)):
            if equal.test(i):
                var a = Int(left_rows.unsafe_get(i))
                var b = Int(right_rows.unsafe_get(i))
                var valid = left.is_valid(a)
                if valid != right.is_valid(b):
                    equal.clear(i)
                elif valid:
                    for j in range(size):
                        elems_left.append(Int32((left.offset + a) * size + j))
                        elems_right.append(Int32((right.offset + b) * size + j))
                        owner.append(i)
        Self._compare_elements(
            left.values(),
            elems_left.finish(),
            right.values(),
            elems_right.finish(),
            owner,
            equal,
            ctx,
        )

    @staticmethod
    def _compare_elements(
        left: DynArray,
        left_rows: Int32Array,
        right: DynArray,
        right_rows: Int32Array,
        owner: List[Int],
        mut equal: Bitmap[mut=True],
        ctx: ExecContext,
    ) raises:
        """Compare flattened element pairs and clear the row owning any pair
        that differs — the reduction the two list leaves share."""
        var m = len(owner)
        if m == 0:
            return
        var elems_equal = Bitmap.alloc_zeroed(m)
        elems_equal.set_range(0, m, True)
        Self.apply(left, left_rows, right, right_rows, elems_equal, ctx)
        for e in range(m):
            if not elems_equal.test(e):
                equal.clear(owner[e])
