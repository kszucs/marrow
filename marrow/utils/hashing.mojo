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

Attribution
-----------
Both structs are ports, so they are derivative works and carry their upstream
licences. The notices are in `NOTICE.txt` at the repository root, as MIT and
BSD-2-Clause both require; marrow is Apache-2.0, with which both are compatible.

- `RapidHash64` — rapidhash V3, MIT, (C) 2025 Nicolas De Carli.
  `RapidSecret.make` further derives from wyhash's `make_secret` by Wang Yi
  (public domain). <https://github.com/Nicoshev/rapidhash>
- `XxHash64` — xxHash, BSD 2-Clause, (c) 2012-2021 Yann Collet.
  <https://github.com/Cyan4973/xxHash>

Reference vectors in `tests/test_hashing.mojo` were generated from those
implementations; neither is vendored.
"""

from std.bit import pop_count, rotate_bits_left
from std.hashlib._ahash import AHasher

from .byteorder import LittleEndian


@always_inline
def mul_wide[
    W: Int
](a: SIMD[DType.uint64, W], b: SIMD[DType.uint64, W]) -> Tuple[
    SIMD[DType.uint64, W], SIMD[DType.uint64, W]
]:
    """Per-lane 128-bit multiply returning `(lo, hi)`, built from four 32-bit
    sub-products.

    Belongs to no one algorithm — rapidhash's `mum`, aHash's `_folded_multiply`
    and the digest combiner are all this same operation, and each having its own
    copy meant picking `XxHash64` still pulled `RapidHash64` into the binary.
    Avoiding `uint128` is what keeps it usable on Metal, which has no such type.
    """
    comptime MASK = SIMD[DType.uint64, W](0xFFFFFFFF)
    var a_lo = a & MASK
    var a_hi = a >> 32
    var b_lo = b & MASK
    var b_hi = b >> 32
    var t0 = a_lo * b_lo
    var t1 = a_lo * b_hi
    var t2 = a_hi * b_lo
    var t3 = a_hi * b_hi
    var mid = (t0 >> 32) + (t1 & MASK) + (t2 & MASK)
    var lo = (t0 & MASK) | (mid << 32)
    var hi = t3 + (t1 >> 32) + (t2 >> 32) + (mid >> 32)
    return (lo, hi)


@always_inline
def mul_fold[
    W: Int
](a: SIMD[DType.uint64, W], b: SIMD[DType.uint64, W]) -> SIMD[DType.uint64, W]:
    """`mul_wide` with the halves XORed — the multiply-fold every hash here
    uses somewhere."""
    var lo_hi = mul_wide[W](a, b)
    return lo_hi[0] ^ lo_hi[1]


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
        """The 64-bit hash of `data` — variable-length keys (string, binary)."""
        ...

    @always_inline
    @staticmethod
    def hash_lanes[
        byte_width: Int, W: Int
    ](values: SIMD[DType.uint64, W]) -> SIMD[DType.uint64, W]:
        """One digest per lane, for `W` fixed-width values already widened to
        `uint64` — the hot path.

        Not expressible through `std.hashlib.Hasher`: that folds a whole vector
        into a single accumulator (one digest for W rows) and its `finish`
        consumes the hasher, so per-row hashing would need W instances per
        vector. This is a pure arithmetic function of the lane, with no memory
        access, which is what lets the kernel vectorise and run on a GPU.
        """
        ...

    @always_inline
    @staticmethod
    def combine_lanes[
        W: Int
    ](existing: SIMD[DType.uint64, W], new: SIMD[DType.uint64, W]) -> SIMD[
        DType.uint64, W
    ]:
        """Fold a second digest into the first, per lane — multi-column keys and
        values too wide for one lane.

        Defaulted because this is marrow's construction (golden-ratio XOR then
        a 128-bit multiply-fold), not part of any of these algorithms' specs —
        its job is to fold two already-well-mixed digests without losing
        entropy, which does not depend on which hash produced them. It builds
        on the free `mul_fold`, not on any hasher, so selecting one algorithm
        does not link another. Override if an algorithm has its own combine
        worth preferring.
        """
        return mul_fold[W](
            existing ^ SIMD[DType.uint64, W](0x9E3779B97F4A7C15), new
        )


@fieldwise_init
struct RapidSecret(Copyable, Movable):
    """The eight 64-bit words rapidhash mixes its input with.

    `DEFAULT` is upstream's `rapid_secret`, and is what `RapidHash64.hash`
    uses — so the default digest matches the reference implementation.

    `make` is the optional half: a port of wyhash's `make_secret`, which
    rapidhash ships in `secret.h`. It derives a *random* secret from a seed, so
    an attacker who cannot predict it cannot precompute colliding keys — the
    defence against hash-flooding a group-by or join into quadratic time.
    Nothing uses it by default; a caller opts in with `hash_with`.

    **`make` randomises only the first four words**, because wyhash has four
    secrets and rapidhash has eight. Words 4-7 keep their canonical values.
    That is upstream's behaviour, not an omission here.
    """

    comptime _WITNESSES = [
        UInt64(2),
        UInt64(3),
        UInt64(5),
        UInt64(7),
        UInt64(11),
        UInt64(13),
        UInt64(17),
        UInt64(19),
        UInt64(23),
        UInt64(29),
        UInt64(31),
        UInt64(37),
    ]
    """Deterministic Miller-Rabin witnesses: this set is proven exact for every
    64-bit integer, so `_is_prime` decides rather than estimates."""

    var words: Array[UInt64, 8]

    comptime DEFAULT = Self(
        Array[UInt64, 8](
            UInt64(0x2D358DCCAA6C78A5),
            UInt64(0x8BB84B93962EACC9),
            UInt64(0x4B33A62ED433D4A3),
            UInt64(0x4D5A2DA51DE1AA47),
            UInt64(0xA0761D6478BD642F),
            UInt64(0xE7037ED1A0B428DB),
            UInt64(0x90ED1765281C388C),
            UInt64(0xAAAAAAAAAAAAAAAA),
            __list_literal__=None,
        )
    )
    """Upstream `rapid_secret`. Words 1, 2 and 7 are the ones the short-input
    path uses, which is why they are the only three the lane hash needs."""

    @always_inline
    def __getitem__(self, i: Int) -> UInt64:
        return self.words[i]

    # --- optional randomisation ---

    @staticmethod
    @always_inline
    def _wyrand(mut seed: UInt64) -> UInt64:
        """wyhash's PRNG, as `secret.h` uses it to draw candidate bytes."""
        seed += 0x2D358DCCAA6C78A5
        return RapidHash64.mix(seed, seed ^ 0x8BB84B93962EACC9)

    @staticmethod
    def _is_prime(n: UInt64) -> Bool:
        """Deterministic Miller-Rabin. The witness set {2,3,5,7,11,13,17,19,23,
        29,31,37} is proven exact for every 64-bit integer, so this is a
        decision procedure rather than a probabilistic test."""
        if n < 2:
            return False
        for small in materialize[Self._WITNESSES]():
            if n == small:
                return True
            if n % small == 0:
                return False
        var d = n - 1
        var r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in materialize[Self._WITNESSES]():
            var x = Self._pow_mod(a, d, n)
            if x == 1 or x == n - 1:
                continue
            var composite = True
            for _ in range(r - 1):
                x = Self._mul_mod(x, x, n)
                if x == n - 1:
                    composite = False
                    break
            if composite:
                return False
        return True

    @staticmethod
    @always_inline
    def _mul_mod(a: UInt64, b: UInt64, m: UInt64) -> UInt64:
        """`a * b % m` via `uint128`, so the product cannot wrap."""
        var wide = a.cast[DType.uint128]() * b.cast[DType.uint128]()
        return UInt64(wide % m.cast[DType.uint128]())

    @staticmethod
    def _pow_mod(a: UInt64, b: UInt64, m: UInt64) -> UInt64:
        var result = UInt64(1)
        var base = a % m
        var exp = b
        while exp > 0:
            if exp & 1:
                result = Self._mul_mod(result, base, m)
            base = Self._mul_mod(base, base, m)
            exp >>= 1
        return result

    @staticmethod
    def make(seed: UInt64) -> Self:
        """A random secret derived from `seed` — the port of `make_secret`.

        Each of the first four words is redrawn until it satisfies all four of
        upstream's constraints simultaneously: every byte has popcount 4 (the
        `_CANDIDATES` table is exactly the 70 bytes with four bits set), the word
        is odd, it differs from each earlier word in exactly 32 bit positions,
        and it is prime. Deterministic in `seed`, so a process can log its seed
        and reproduce a run.
        """
        # Upstream tables 70 bytes; they are exactly the bytes with popcount
        # 4, so derive them rather than transcribing a magic table.
        var candidates = List[UInt8]()
        for byte in range(256):
            if pop_count(UInt8(byte)) == 4:
                candidates.append(UInt8(byte))

        var out = materialize[Self.DEFAULT]()
        var rng = seed
        for i in range(4):
            while True:
                var word = UInt64(0)
                for j in range(0, 64, 8):
                    var pick = Int(Self._wyrand(rng) % UInt64(len(candidates)))
                    word |= UInt64(candidates[pick]) << UInt64(j)
                if word % 2 == 0:
                    continue
                var clashes = False
                for j in range(i):
                    if pop_count(word ^ out.words[j]) != 32:
                        clashes = True
                        break
                if clashes:
                    continue
                if not Self._is_prime(word):
                    continue
                out.words[i] = word
                break
        return out^


struct RapidHash64(Hasher):
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

    comptime _SECRET1 = UInt64(0x8BB84B93962EACC9)
    comptime _SECRET2 = UInt64(0x4B33A62ED433D4A3)
    comptime _SECRET7 = UInt64(0xAAAAAAAAAAAAAAAA)
    """`rapid_secret[1]`, `[2]` and `[7]` — the three the fixed-width path
    mixes with. The full eight live in `RapidSecret`."""

    comptime name = StaticString("rapidhash64")

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

    # --- variable-length byte spans ---

    @staticmethod
    @always_inline
    def _r64(data: Span[UInt8, _], at: Int) -> UInt64:
        return LittleEndian.fixed[DType.uint64](data, at)

    @staticmethod
    @always_inline
    def _r32(data: Span[UInt8, _], at: Int) -> UInt64:
        return UInt64(LittleEndian.fixed[DType.uint32](data, at))

    @staticmethod
    def hash(data: Span[UInt8, _], seed: UInt64 = 0) -> UInt64:
        """Rapidhash of a byte span — the `Hasher` entry point.

        A faithful port of `rapidhash_internal` with the default secret and the
        non-`RAPIDHASH_COMPACT` unrolling, which is what upstream's `rapidhash()`
        uses. Every length branch is pinned by a reference vector in
        `tests/test_hashing.mojo`.

        This replaced an AHash fallback: `marrow.kernels.hashing` hashed numeric
        columns with rapidhash and string columns with `std.hashlib.hash`,
        because the multi-branch byte-string path did not exist yet.
        """
        return Self.hash_with(data, materialize[RapidSecret.DEFAULT](), seed)

    @staticmethod
    @always_inline
    def _mix_block(
        data: Span[UInt8, _],
        off: Int,
        secret: RapidSecret,
        mut acc: Array[UInt64, 7],
    ):
        """One 112-byte round: accumulator `k` absorbs bytes `16k` and `16k+8`
        under `secret[k]`.

        Upstream writes this as seven near-identical statements over seven named
        locals (`seed`, `see1`..`see6`); the pattern is exact, so it is a loop
        here. The `while i > 224` body is two of these back to back, which is
        upstream's manual unroll.
        """
        comptime for k in range(7):
            acc[k] = Self.mix(
                Self._r64(data, off + k * 16) ^ secret[k],
                Self._r64(data, off + k * 16 + 8) ^ acc[k],
            )

    @staticmethod
    def hash_with(
        data: Span[UInt8, _], secret: RapidSecret, seed: UInt64 = 0
    ) -> UInt64:
        """`hash`, against a caller-supplied secret.

        The opt-in half of `RapidSecret`: pass `RapidSecret.make(s)` for a
        per-process random secret so crafted keys cannot be precomputed to
        collide. `hash` uses the canonical secret, so it matches upstream.

        Restructured from the C in two ways that do not change the result, both
        pinned by the reference vectors in `tests/test_hashing.mojo`:

        - the seven accumulators are an array and one `comptime for`, not seven
          named locals and 21 near-identical statements;
        - upstream's `if (i > 32) { if (i > 48) { ... } }` ladder is flattened,
          since each condition implies every earlier one. The nesting is a
          short-circuit hint in C, not logic.
        """
        var n = len(data)
        var s = seed ^ Self.mix(seed ^ secret[2], secret[1])
        var a = UInt64(0)
        var b = UInt64(0)
        var off = 0
        var i = n

        if n <= 16:
            if n >= 4:
                s ^= UInt64(n)
                if n >= 8:
                    a = Self._r64(data, 0)
                    b = Self._r64(data, n - 8)
                else:
                    a = Self._r32(data, 0)
                    b = Self._r32(data, n - 4)
            elif n > 0:
                a = (UInt64(data[0]) << 45) | UInt64(data[n - 1])
                b = UInt64(data[n >> 1])
        else:
            if n > 112:
                var acc = Array[UInt64, 7](fill=s)
                while i > 224:
                    Self._mix_block(data, off, secret, acc)
                    Self._mix_block(data, off + 112, secret, acc)
                    off += 224
                    i -= 224
                if i > 112:
                    Self._mix_block(data, off, secret, acc)
                    off += 112
                    i -= 112
                # upstream's fold order, verbatim; acc[0] is its `seed`
                acc[0] ^= acc[1]
                acc[2] ^= acc[3]
                acc[4] ^= acc[5]
                acc[0] ^= acc[6]
                acc[2] ^= acc[4]
                acc[0] ^= acc[2]
                s = acc[0]

            # 17-112 bytes: one mix per 16, each guard implying the last
            comptime for step in range(6):
                # upstream alternates secret[2]/secret[1] as 2,2,1,1,2,1
                comptime sec = 2 if (step == 0 or step == 1 or step == 4) else 1
                if i > 16 + step * 16:
                    s = Self.mix(
                        Self._r64(data, off + step * 16) ^ secret[sec],
                        Self._r64(data, off + step * 16 + 8) ^ s,
                    )

            a = Self._r64(data, off + i - 16) ^ UInt64(i)
            b = Self._r64(data, off + i - 8)

        a ^= secret[1]
        b ^= s
        var lo_hi = Self.mum(a, b)
        return Self.mix(lo_hi[0] ^ secret[7], lo_hi[1] ^ secret[1] ^ UInt64(i))

    # --- per-SIMD-lane (GPU-compatible: no uint128) ---

    @always_inline
    @staticmethod
    def hash_lanes[
        byte_width: Int, W: Int
    ](values: SIMD[DType.uint64, W]) -> SIMD[DType.uint64, W]:
        """`hash_fixed` for `W` lanes at once, with the seed folded at comptime.

        `byte_width` enters the seed, so the same bits hashed as 4 and as 8
        bytes give different digests — that is what keeps an int32 and an int64
        column from colliding wholesale.
        """
        comptime seed = Self.mix(Self._SECRET2, Self._SECRET1) ^ UInt64(
            byte_width
        )
        var a = values ^ SIMD[DType.uint64, W](Self._SECRET1)
        var b = values ^ SIMD[DType.uint64, W](seed)
        var lo_hi = mul_wide[W](a, b)
        return mul_fold[W](
            lo_hi[0] ^ SIMD[DType.uint64, W](Self._SECRET7),
            lo_hi[1]
            ^ SIMD[DType.uint64, W](Self._SECRET1 ^ UInt64(byte_width)),
        )


struct XxHash64(Hasher):
    """XXH64 — the value hash for the Parquet split-block bloom filter.

    A namespace, like `Crc32`. `xxh64` and its three round helpers were four
    free functions here; three are private and meaningless apart from the
    fourth, so grouping them makes the entry point obvious and stops
    `_rotl`/`_round`/`_merge_round` reading as general utilities.
    """

    comptime name = StaticString("xxhash64")

    comptime _P1 = UInt64(0x9E3779B185EBCA87)
    comptime _P2 = UInt64(0xC2B2AE3D27D4EB4F)
    comptime _P3 = UInt64(0x165667B19E3779F9)
    comptime _P4 = UInt64(0x85EBCA77C2B2AE63)
    comptime _P5 = UInt64(0x27D4EB2F165667C5)

    @staticmethod
    @always_inline
    def hash_lanes[
        byte_width: Int, W: Int
    ](values: SIMD[DType.uint64, W]) -> SIMD[DType.uint64, W]:
        """XXH64 of `byte_width` little-endian bytes per lane, seed 0.

        For a *fixed* length below 32 bytes the algorithm has no loop and no
        memory access — the four-accumulator block path never runs, and which
        tail applies is a comptime decision — so it collapses to straight-line
        arithmetic that vectorises. `hash` and this must agree for the same
        bytes, which the tests assert lane by lane.
        """
        comptime assert byte_width <= 8, "hash_lanes covers widths up to 8"
        comptime P1 = SIMD[DType.uint64, W](Self._P1)
        comptime P2 = SIMD[DType.uint64, W](Self._P2)
        comptime P3 = SIMD[DType.uint64, W](Self._P3)
        comptime P4 = SIMD[DType.uint64, W](Self._P4)
        comptime P5 = SIMD[DType.uint64, W](Self._P5)
        var h = SIMD[DType.uint64, W](Self._P5 + UInt64(byte_width))

        comptime if byte_width == 8:
            h ^= rotate_bits_left[31](values * P2) * P1
            h = rotate_bits_left[27](h) * P1 + P4
        elif byte_width >= 4:
            h ^= (values & SIMD[DType.uint64, W](0xFFFFFFFF)) * P1
            h = rotate_bits_left[23](h) * P2 + P3
            comptime for i in range(4, byte_width):
                h ^= (
                    (values >> UInt64(i * 8)) & SIMD[DType.uint64, W](0xFF)
                ) * P5
                h = rotate_bits_left[11](h) * P1
        else:
            comptime for i in range(byte_width):
                h ^= (
                    (values >> UInt64(i * 8)) & SIMD[DType.uint64, W](0xFF)
                ) * P5
                h = rotate_bits_left[11](h) * P1

        h ^= h >> 33
        h *= P2
        h ^= h >> 29
        h *= P3
        h ^= h >> 32
        return h

    @staticmethod
    @always_inline
    def _round(acc: UInt64, input: UInt64) -> UInt64:
        var a = acc + input * Self._P2
        a = rotate_bits_left[31](a)
        return a * Self._P1

    @staticmethod
    @always_inline
    def _merge_round(acc: UInt64, val: UInt64) -> UInt64:
        var v = Self._round(0, val)
        var a = acc ^ v
        return a * Self._P1 + Self._P4

    @staticmethod
    def hash(data: Span[UInt8, _], seed: UInt64 = 0) -> UInt64:
        """The 64-bit xxHash of `data` (wrapping arithmetic throughout), matching the
        canonical XXH64 — the value hash for the Parquet split-block bloom filter.
        """
        var n = len(data)
        var h: UInt64
        var p = 0
        if n >= 32:
            var v1 = seed + Self._P1 + Self._P2
            var v2 = seed + Self._P2
            var v3 = seed + 0
            var v4 = seed - Self._P1
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
                rotate_bits_left[1](v1)
                + rotate_bits_left[7](v2)
                + rotate_bits_left[12](v3)
                + rotate_bits_left[18](v4)
            )
            h = Self._merge_round(h, v1)
            h = Self._merge_round(h, v2)
            h = Self._merge_round(h, v3)
            h = Self._merge_round(h, v4)
        else:
            h = seed + Self._P5

        h += UInt64(n)

        while p + 8 <= n:
            var k1 = Self._round(0, LittleEndian.fixed[DType.uint64](data, p))
            h ^= k1
            h = rotate_bits_left[27](h) * Self._P1 + Self._P4
            p += 8
        if p + 4 <= n:
            h ^= UInt64(LittleEndian.fixed[DType.uint32](data, p)) * Self._P1
            h = rotate_bits_left[23](h) * Self._P2 + Self._P3
            p += 4
        while p < n:
            h ^= UInt64(data[p]) * Self._P5
            h = rotate_bits_left[11](h) * Self._P1
            p += 1

        h ^= h >> 33
        h *= Self._P2
        h ^= h >> 29
        h *= Self._P3
        h ^= h >> 32
        return h


struct AHash64(Hasher):
    """AHash — the third selectable algorithm, and the one marrow does not
    implement itself.

    `hash` delegates to `std.hashlib`'s `AHasher`, so the digest is the standard
    library's, bit for bit; there is no second implementation of aHash here to
    drift. `hash_lanes` is marrow's, because std cannot express it: its `Hasher`
    folds a whole SIMD vector into one accumulator, and `finish` consumes the
    hasher, so a per-row digest would need one hasher instance per lane.

    The lane form reproduces aHash's own short-input path — `buffer` seeded from
    the pi-derived key, one `_large_update`, then `finish`'s folded multiply and
    rotate — which the tests pin against `std.hashlib.hash` of the same bytes.

    aHash is **not** a drop-in for the other two where digests are persisted: it
    is explicitly a hash-table hash, not a checksum, and the standard library
    makes no stability guarantee across versions. Parquet bloom filters must
    stay on `XxHash64`, which the spec mandates.
    """

    comptime name = StaticString("ahash64")

    # aHash's pi-derived default key, and the two mixing constants.
    comptime _BUFFER = UInt64(0x243F6A8885A308D3)
    comptime _PAD = UInt64(0x13198A2E03707344)
    comptime _XKEY0 = UInt64(0xA4093822299F31D0)
    comptime _XKEY1 = UInt64(0x082EFA98EC4E6C89)
    comptime _MULTIPLE = UInt64(6364136223846793005)
    comptime _ROT = 23

    @staticmethod
    def hash(data: Span[UInt8, _], seed: UInt64 = 0) -> UInt64:
        """Delegates to `std.hashlib`, so this is exactly the standard
        library's aHash."""
        var hasher = AHasher[SIMD[DType.uint64, 4](0)]()
        hasher._update_with_bytes(data)
        var h = hasher^.finish()
        return h if seed == 0 else h ^ seed

    @staticmethod
    @always_inline
    def hash_lanes[
        byte_width: Int, W: Int
    ](values: SIMD[DType.uint64, W]) -> SIMD[DType.uint64, W]:
        """AHash of `byte_width` little-endian bytes per lane."""
        comptime assert byte_width <= 8, "hash_lanes covers widths up to 8"
        var buffer = SIMD[DType.uint64, W](
            (Self._BUFFER + UInt64(byte_width)) * Self._MULTIPLE
        )

        # `_read_small`, specialised: for 4-8 bytes aHash reads the leading and
        # trailing u32; both are the same value once the lane holds it whole.
        var a: SIMD[DType.uint64, W]
        var b: SIMD[DType.uint64, W]
        comptime if byte_width >= 4:
            a = values & SIMD[DType.uint64, W](0xFFFFFFFF)
            b = values >> UInt64((byte_width - 4) * 8)
            b &= SIMD[DType.uint64, W](0xFFFFFFFF)
        elif byte_width >= 2:
            a = values & SIMD[DType.uint64, W](0xFFFF)
            b = (values >> UInt64((byte_width - 1) * 8)) & SIMD[
                DType.uint64, W
            ](0xFF)
        elif byte_width == 1:
            a = values & SIMD[DType.uint64, W](0xFF)
            b = a
        else:
            a = SIMD[DType.uint64, W](0)
            b = SIMD[DType.uint64, W](0)

        var combined = mul_fold[W](
            a ^ SIMD[DType.uint64, W](Self._XKEY0),
            b ^ SIMD[DType.uint64, W](Self._XKEY1),
        )
        buffer = (buffer + SIMD[DType.uint64, W](Self._PAD)) ^ combined
        buffer = rotate_bits_left[Self._ROT](buffer)

        var rot = buffer & SIMD[DType.uint64, W](63)
        var folded = mul_fold[W](buffer, SIMD[DType.uint64, W](Self._PAD))
        return (folded << rot) | (
            folded >> ((SIMD[DType.uint64, W](64) - rot) & 63)
        )
