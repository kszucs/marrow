"""The two hash functions marrow implements.

Neither is in the Mojo standard library, and neither is substitutable:

- **XXH64** is *mandated* by the Parquet spec for split-block bloom filters, so
  the value has to match other implementations byte for byte.
- **rapidhash v3** is the group-by / join key hash, chosen for throughput; its
  mixing steps are consumed by a SIMD kernel a lane at a time.

`Hasher` is the swap point: one static `hash(span, seed)` plus a `name`, which
is the shape marrow's callers actually use. `XxHash64` conforms. `RapidHash64`
does **not** — it is a set of mixing primitives consumed a SIMD lane at a time,
not a byte-span hash; porting rapidhash's byte-string path would let it conform,
and that is the follow-up if a second span hash is ever wanted.

`std.hashlib`'s own `Hasher` trait is deliberately not reused: it is a
*streaming* protocol (`__init__` / `_update_with_bytes` / `update` /
`finish(var self)`) for feeding arbitrary `Hashable` values into a dictionary.
Conforming would turn a one-shot span hash into a stateful accumulator and put a
consuming `finish` inside hot loops, for no gain — `AHasher` and `Fnv1a` are
different algorithms and neither can replace XXH64, which the Parquet spec
mandates.

Both are namespaces of static methods, and this module depends on `.byteorder`
alone — no arrays, no dtypes — so each algorithm can be read and checked against
its reference vectors without the array layer in scope.

The array-level machinery — hashing a `PrimitiveArray`, the null sentinel, dtype
dispatch, the GPU launch — is `marrow.kernels.hashing`.
"""

from .byteorder import LittleEndian


trait Hasher:
    """A 64-bit hash of a byte span — the contract for swapping the algorithm.

    Deliberately *not* `std.hashlib.Hasher`: that is a streaming protocol
    (`__init__` / `_update_with_bytes` / `update` / `finish(var self)`) for
    feeding arbitrary `Hashable` values into a dictionary, and conforming would
    mean turning a one-shot span hash into a stateful accumulator, then paying a
    consuming `finish` inside hot loops. This is the shape marrow's callers
    actually use: hand it bytes, get 64 bits.

    `name` exists so a caller can report which algorithm produced a digest —
    Parquet records the hash function in its bloom-filter header, and "which
    hash was this built with" must be answerable without inspecting the type.
    """

    comptime name: StaticString
    """This hash function's identity, for display and format headers."""

    @staticmethod
    def hash(data: Span[UInt8, _], seed: UInt64 = 0) -> UInt64:
        """The 64-bit hash of `data`."""
        ...


struct RapidHash64:
    """The rapidhash v3 primitives: 128-bit multiply, multiply-mix, and the
    fixed-width single-value hash, scalar and per-SIMD-lane.

    A namespace of static methods, like `XxHash64` and `Crc32` — these were six
    free functions and three module-level secrets, none of which mean anything
    apart from each other, and `_rapid_mum` / `_rapid_mix` read as general
    utilities when they are one hash function's internals.

    Every method is `@always_inline`: this is the hot path under
    `RapidHash.dispatch`, and the mixing steps are a handful of multiplies that
    must fold into the caller's loop.
    """

    comptime SECRET1 = UInt64(0x8BB84B93962EACC9)
    comptime SECRET2 = UInt64(0x4B33A62ED433D4A3)
    comptime SECRET7 = UInt64(0xAAAAAAAAAAAAAAAA)

    # --- scalar ---

    @staticmethod
    @always_inline
    def mum(A: UInt64, B: UInt64) -> Tuple[UInt64, UInt64]:
        """128-bit multiply, returning `(lo, hi)`. Port of `rapid_mum`."""
        var r = A.cast[DType.uint128]() * B.cast[DType.uint128]()
        return (UInt64(r), UInt64(r >> 64))

    @staticmethod
    @always_inline
    def mix(A: UInt64, B: UInt64) -> UInt64:
        """Multiply-mix: 128-bit multiply then XOR the halves. Port of
        `rapid_mix`."""
        var lo_hi = Self.mum(A, B)
        return lo_hi[0] ^ lo_hi[1]

    @staticmethod
    @always_inline
    def hash_fixed[byte_width: Int](value: UInt64) -> UInt64:
        """Rapidhash of a single fixed-width value.

        Exact port of `rapidhash_internal()` for `len=byte_width`, `seed=0`.
        C reference:
          seed ^= rapid_mix(seed ^ secret[2], secret[1])  // seed=0
          seed ^= len  // for len >= 4
          a = value ^ secret[1]
          b = value ^ seed
          rapid_mum(&a, &b)
          return rapid_mix(a ^ secret[7], b ^ secret[1] ^ len)
        """
        var seed = Self.mix(Self.SECRET2, Self.SECRET1) ^ UInt64(byte_width)
        var a = value ^ Self.SECRET1
        var b = value ^ seed
        var lo_hi = Self.mum(a, b)
        return Self.mix(
            lo_hi[0] ^ Self.SECRET7,
            lo_hi[1] ^ Self.SECRET1 ^ UInt64(byte_width),
        )

    # --- per-SIMD-lane (GPU-compatible: no uint128) ---

    @staticmethod
    @always_inline
    def mum_wide[
        W: Int
    ](a: SIMD[DType.uint64, W], b: SIMD[DType.uint64, W]) -> Tuple[
        SIMD[DType.uint64, W], SIMD[DType.uint64, W]
    ]:
        """128-bit multiply returning `(lo, hi)` from 32-bit sub-products.

        GPU-compatible: avoids `uint128`, which Metal does not support.
        """
        var a_lo = a & SIMD[DType.uint64, W](0xFFFFFFFF)
        var a_hi = a >> 32
        var b_lo = b & SIMD[DType.uint64, W](0xFFFFFFFF)
        var b_hi = b >> 32
        var t0 = a_lo * b_lo
        var t1 = a_lo * b_hi
        var t2 = a_hi * b_lo
        var t3 = a_hi * b_hi
        var mid = (
            (t0 >> 32)
            + (t1 & SIMD[DType.uint64, W](0xFFFFFFFF))
            + (t2 & SIMD[DType.uint64, W](0xFFFFFFFF))
        )
        var lo = (t0 & SIMD[DType.uint64, W](0xFFFFFFFF)) | (mid << 32)
        var hi = t3 + (t1 >> 32) + (t2 >> 32) + (mid >> 32)
        return (lo, hi)

    @staticmethod
    @always_inline
    def mix_wide[
        W: Int
    ](a: SIMD[DType.uint64, W], b: SIMD[DType.uint64, W]) -> SIMD[
        DType.uint64, W
    ]:
        """`mix` for SIMD lanes: 128-bit multiply then XOR the halves."""
        var lo_hi = Self.mum_wide[W](a, b)
        return lo_hi[0] ^ lo_hi[1]


comptime _P1: UInt64 = 0x9E3779B185EBCA87
comptime _P2: UInt64 = 0xC2B2AE3D27D4EB4F
comptime _P3: UInt64 = 0x165667B19E3779F9
comptime _P4: UInt64 = 0x85EBCA77C2B2AE63
comptime _P5: UInt64 = 0x27D4EB2F165667C5


struct XxHash64(Hasher):
    """XXH64 — the value hash for the Parquet split-block bloom filter.

    A namespace, like `Crc32`. `xxh64` and its three round helpers were four
    free functions here; three are private and meaningless apart from the
    fourth, so grouping them makes the entry point obvious and stops
    `_rotl`/`_round`/`_merge_round` reading as general utilities.
    """

    comptime name = StaticString("xxhash64")

    @staticmethod
    @always_inline
    def _rotl(x: UInt64, r: Int) -> UInt64:
        return (x << UInt64(r)) | (x >> UInt64(64 - r))

    @staticmethod
    @always_inline
    def _round(acc: UInt64, input: UInt64) -> UInt64:
        var a = acc + input * _P2
        a = Self._rotl(a, 31)
        return a * _P1

    @staticmethod
    @always_inline
    def _merge_round(acc: UInt64, val: UInt64) -> UInt64:
        var v = Self._round(0, val)
        var a = acc ^ v
        return a * _P1 + _P4

    @staticmethod
    def hash(data: Span[UInt8, _], seed: UInt64 = 0) -> UInt64:
        """The 64-bit xxHash of `data` (wrapping arithmetic throughout), matching the
        canonical XXH64 — the value hash for the Parquet split-block bloom filter.
        """
        var n = len(data)
        var h: UInt64
        var p = 0
        if n >= 32:
            var v1 = seed + _P1 + _P2
            var v2 = seed + _P2
            var v3 = seed + 0
            var v4 = seed - _P1
            var limit = n - 32
            while p <= limit:
                v1 = Self._round(v1, LittleEndian.fixed[DType.uint64](data, p))
                p += 8
                v2 = Self._round(v2, LittleEndian.fixed[DType.uint64](data, p))
                p += 8
                v3 = Self._round(v3, LittleEndian.fixed[DType.uint64](data, p))
                p += 8
                v4 = Self._round(v4, LittleEndian.fixed[DType.uint64](data, p))
                p += 8
            h = (
                Self._rotl(v1, 1)
                + Self._rotl(v2, 7)
                + Self._rotl(v3, 12)
                + Self._rotl(v4, 18)
            )
            h = Self._merge_round(h, v1)
            h = Self._merge_round(h, v2)
            h = Self._merge_round(h, v3)
            h = Self._merge_round(h, v4)
        else:
            h = seed + _P5

        h += UInt64(n)

        while p + 8 <= n:
            var k1 = Self._round(0, LittleEndian.fixed[DType.uint64](data, p))
            h ^= k1
            h = Self._rotl(h, 27) * _P1 + _P4
            p += 8
        if p + 4 <= n:
            h ^= UInt64(LittleEndian.fixed[DType.uint32](data, p)) * _P1
            h = Self._rotl(h, 23) * _P2 + _P3
            p += 4
        while p < n:
            h ^= UInt64(data[p]) * _P5
            h = Self._rotl(h, 11) * _P1
            p += 1

        h ^= h >> 33
        h *= _P2
        h ^= h >> 29
        h *= _P3
        h ^= h >> 32
        return h
