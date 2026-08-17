"""`RapidHash64` and `XxHash64` against independent reference values.

The XXH64 vectors were produced by a from-scratch Python implementation of the
canonical algorithm, cross-checked against the two values already pinned in
`parquet/tests/test_bloom.mojo` before being trusted, and chosen to cover every
branch of `hash`: the 32-byte four-accumulator block, the 8-byte loop, the
4-byte tail and the byte-at-a-time tail. `a` and `abc` also match the
widely-published canonical vectors.

The `mum` products are plain 128-bit multiplications, computable by hand.
"""

from std.testing import assert_equal, assert_true, assert_false, assert_raises

from std.bit import pop_count

from std.hashlib._ahash import AHasher

from ..hashing import (
    mul_fold,
    mul_wide,
    AHash64,
    Hasher,
    RapidHash64,
    RapidSecret,
    XxHash64,
)


def _bytes(s: String) -> List[UInt8]:
    return List[UInt8](s.as_bytes())


def _iota_bytes(n: Int) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(n):
        out.append(UInt8(i))
    return out^


# ---------------------------------------------------------------------------
# RapidHash64 — mixing primitives
# ---------------------------------------------------------------------------


def test_rapidhash64_mum_products() raises:
    """`mum` is a plain 128-bit multiply split into (lo, hi)."""
    var r = RapidHash64.mum(0x0123456789ABCDEF, 0xFEDCBA9876543210)
    assert_equal(r[0], UInt64(0x2236D88FE5618CF0))
    assert_equal(r[1], UInt64(0x0121FA00AD77D742))

    var small = RapidHash64.mum(3, 5)
    assert_equal(small[0], UInt64(15))
    assert_equal(small[1], UInt64(0))

    # the widest product: (2^64-1)^2 = 2^128 - 2^65 + 1
    var big = RapidHash64.mum(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)
    assert_equal(big[0], UInt64(1))
    assert_equal(big[1], UInt64(0xFFFFFFFFFFFFFFFE))


def test_rapidhash64_mix_is_xor_of_halves() raises:
    """`mix` is exactly `mum`'s two halves XORed."""
    assert_equal(
        RapidHash64.mix(0x0123456789ABCDEF, 0xFEDCBA9876543210),
        UInt64(0x2317228F48165BB2),
    )
    assert_equal(
        RapidHash64.mix(UInt64(0x8BB84B93962EACC9), UInt64(0x4B33A62ED433D4A3)),
        UInt64(0x422765567D8FBFD6),
    )


def test_rapidhash64_secrets_match_reference() raises:
    """The three secrets are rapidhash.h's, and a typo in one silently changes
    every hash the library produces."""
    var d = materialize[RapidSecret.DEFAULT]()
    assert_equal(d[1], UInt64(0x8BB84B93962EACC9))
    assert_equal(d[2], UInt64(0x4B33A62ED433D4A3))
    assert_equal(d[7], UInt64(0xAAAAAAAAAAAAAAAA))


def test_rapidhash64_wide_matches_scalar() raises:
    """The GPU-compatible 32-bit-sub-product path must agree with the `uint128`
    one lane for lane.

    This is the invariant that matters most here: `mum_wide` exists only because
    Metal has no `uint128`, so the two implementations must never diverge — and
    nothing else in the suite compares them.
    """
    var a = SIMD[DType.uint64, 4](
        0x0123456789ABCDEF,
        0xFFFFFFFFFFFFFFFF,
        0x0000000100000001,
        3,
    )
    var b = SIMD[DType.uint64, 4](
        0xFEDCBA9876543210,
        0xFFFFFFFFFFFFFFFF,
        0x00000000FFFFFFFF,
        5,
    )
    var wide = mul_wide[4](a, b)
    var mixed = mul_fold[4](a, b)
    for i in range(4):
        var scalar = RapidHash64.mum(a[i], b[i])
        assert_equal(wide[0][i], scalar[0])
        assert_equal(wide[1][i], scalar[1])
        assert_equal(mixed[i], RapidHash64.mix(a[i], b[i]))


def test_rapidhash64_hash_fixed_separates_widths() raises:
    """`byte_width` is folded into the seed, so the same bits hashed as 4 and as
    8 bytes give different digests — that is what keeps int32 and int64 columns
    from colliding wholesale."""
    assert_equal(
        RapidHash64.hash_lanes[4, 1](SIMD[DType.uint64, 1](7))[0],
        RapidHash64.hash_lanes[4, 1](SIMD[DType.uint64, 1](7))[0],
    )
    assert_true(
        RapidHash64.hash_lanes[4, 1](SIMD[DType.uint64, 1](7))[0]
        != RapidHash64.hash_lanes[8, 1](SIMD[DType.uint64, 1](7))[0]
    )
    assert_true(
        RapidHash64.hash_lanes[8, 1](SIMD[DType.uint64, 1](0))[0]
        != RapidHash64.hash_lanes[8, 1](SIMD[DType.uint64, 1](1))[0]
    )


# ---------------------------------------------------------------------------
# XxHash64 — canonical vectors, one per branch of `hash`
# ---------------------------------------------------------------------------


def test_xxhash64_empty_and_short() raises:
    """`n < 32`: seed + P5, then the 4-byte and byte-wise tails."""
    var empty = List[UInt8]()
    assert_equal(XxHash64.hash(Span(empty)), UInt64(0xEF46DB3751D8E999))
    assert_equal(XxHash64.hash(Span(_bytes("a"))), UInt64(0xD24EC4F1A98C6E5B))
    assert_equal(XxHash64.hash(Span(_bytes("abc"))), UInt64(0x44BC2CF5AD770999))
    assert_equal(
        XxHash64.hash(Span(_bytes("abcd"))), UInt64(0xDE0327B0D25D92CC)
    )
    assert_equal(
        XxHash64.hash(Span(_bytes("abcdefg"))), UInt64(0x1860940E2902822D)
    )


def test_xxhash64_eight_byte_loop() raises:
    """`8 <= n < 32`: the `while p + 8 <= n` loop, with and without tails."""
    assert_equal(
        XxHash64.hash(Span(_bytes("abcdefgh"))), UInt64(0x3AD351775B4634B7)
    )
    assert_equal(
        XxHash64.hash(Span(_bytes("123456789"))), UInt64(0x8CB841DB40E6AE83)
    )
    assert_equal(
        XxHash64.hash(Span(_iota_bytes(31))), UInt64(0xC346D2B59B4D8EE1)
    )


def test_xxhash64_block_path() raises:
    """`n >= 32`: the four-accumulator block path plus every tail after it."""
    assert_equal(
        XxHash64.hash(Span(_iota_bytes(32))), UInt64(0xCBF59C5116FF32B4)
    )
    assert_equal(
        XxHash64.hash(Span(_iota_bytes(64))), UInt64(0xF7C67301DB6713F0)
    )
    assert_equal(
        XxHash64.hash(Span(_iota_bytes(77))), UInt64(0x93F85C1B6280EAD3)
    )


def test_xxhash64_seed_changes_the_digest() raises:
    """A non-zero seed is threaded into the accumulators, not ignored."""
    var hello = _bytes("Hello")
    assert_equal(XxHash64.hash(Span(hello)), UInt64(0x0A75A91375B27D44))
    assert_equal(
        XxHash64.hash(Span(hello), seed=0x9E3779B1),
        UInt64(0xB011219B6933E0AB),
    )


def test_xxhash64_conforms_to_hasher() raises:
    """`XxHash64` is usable through the `Hasher` contract, which is what makes
    the algorithm swappable."""

    def digest[H: Hasher](data: Span[UInt8, _]) -> UInt64:
        return H.hash(data)

    var abc = _bytes("abc")
    assert_equal(digest[XxHash64](Span(abc)), UInt64(0x44BC2CF5AD770999))
    assert_equal(XxHash64.name, StaticString("xxhash64"))


# ---------------------------------------------------------------------------
# RapidSecret — the optional randomised secret
#
# Expected words come from upstream's own `make_secret` (rapidhash's
# `secret.h`, itself wyhash's, public domain), compiled and run once against
# seeds 0-2. The C source is not vendored; these are the values it produced.
# Regenerate by building `secret.h` with `make_secret(seed, sec)` and printing
# `sec[0..3]`.
# ---------------------------------------------------------------------------


def test_rapid_secret_default_is_upstream() raises:
    """`DEFAULT` is upstream's `rapid_secret`. Words 1, 2 and 7 are the ones the
    short-input path mixes with, and must equal the constants the lane hash
    already uses."""
    var s = materialize[RapidSecret.DEFAULT]()
    assert_equal(s[0], UInt64(0x2D358DCCAA6C78A5))
    assert_equal(s[1], UInt64(0x8BB84B93962EACC9))
    assert_equal(s[2], UInt64(0x4B33A62ED433D4A3))
    assert_equal(s[3], UInt64(0x4D5A2DA51DE1AA47))
    assert_equal(s[4], UInt64(0xA0761D6478BD642F))
    assert_equal(s[5], UInt64(0xE7037ED1A0B428DB))
    assert_equal(s[6], UInt64(0x90ED1765281C388C))
    assert_equal(s[7], UInt64(0xAAAAAAAAAAAAAAAA))


def test_rapid_secret_make_matches_reference() raises:
    """The port reproduces upstream `make_secret` word for word."""
    var s0 = RapidSecret.make(0)
    assert_equal(s0[0], UInt64(0x39D43C5C4E3A724B))
    assert_equal(s0[1], UInt64(0x6596E14753CCA38B))
    assert_equal(s0[2], UInt64(0xC68D954B2B339353))
    assert_equal(s0[3], UInt64(0x96B4A6E45C65AA55))

    var s1 = RapidSecret.make(1)
    assert_equal(s1[0], UInt64(0x0FF09999359563A5))
    assert_equal(s1[1], UInt64(0x2D6C35B48DE81B2B))
    assert_equal(s1[2], UInt64(0xD2CA93C527B49C2B))
    assert_equal(s1[3], UInt64(0x47C3660FC5E1C9B1))

    var s2 = RapidSecret.make(2)
    assert_equal(s2[0], UInt64(0x74B48B33D8556339))
    assert_equal(s2[1], UInt64(0xE2C93A631B9A1B1D))
    assert_equal(s2[2], UInt64(0xB22BD1D85A743A27))
    assert_equal(s2[3], UInt64(0xB887C563B139D853))


def test_rapid_secret_make_holds_every_invariant() raises:
    """The four constraints upstream retries until it satisfies. A generator
    that dropped one would still look random and would still produce vectors;
    only these assertions catch it."""
    for seed in range(6):
        var s = RapidSecret.make(UInt64(seed))
        for i in range(4):
            var w = s[i]
            assert_true(w % 2 == 1, "word must be odd")
            for j in range(0, 64, 8):
                assert_equal(Int(pop_count(UInt8((w >> UInt64(j)) & 0xFF))), 4)
            for j in range(i):
                assert_equal(Int(pop_count(w ^ s[j])), 32)


def test_rapid_secret_make_leaves_the_upper_words_canonical() raises:
    """Wyhash has four secrets and rapidhash has eight, so `make_secret`
    randomises words 0-3 only. Upstream behaviour, pinned so a future change
    is deliberate."""
    var s = RapidSecret.make(7)
    var d = materialize[RapidSecret.DEFAULT]()
    for i in range(4, 8):
        assert_equal(s[i], d[i])


def test_rapid_secret_make_is_deterministic() raises:
    """Same seed, same secret — so a process can log its seed and reproduce a
    run. Different seeds must differ, or the randomisation buys nothing."""
    var a = RapidSecret.make(12345)
    var b = RapidSecret.make(12345)
    var c = RapidSecret.make(12346)
    for i in range(8):
        assert_equal(a[i], b[i])
    var any_differs = False
    for i in range(4):
        if a[i] != c[i]:
            any_differs = True
    assert_true(any_differs, "different seeds must give different secrets")


def test_rapid_secret_is_prime_decides() raises:
    """Miller-Rabin over the exact 64-bit witness set."""
    assert_false(RapidSecret._is_prime(0))
    assert_false(RapidSecret._is_prime(1))
    assert_true(RapidSecret._is_prime(2))
    assert_true(RapidSecret._is_prime(3))
    assert_false(RapidSecret._is_prime(4))
    assert_true(RapidSecret._is_prime(97))
    assert_false(RapidSecret._is_prime(561))  # Carmichael
    assert_false(RapidSecret._is_prime(1105))  # Carmichael
    assert_true(RapidSecret._is_prime(2147483647))  # 2^31 - 1
    assert_true(
        RapidSecret._is_prime(18446744073709551557)
    )  # largest u64 prime
    assert_false(RapidSecret._is_prime(18446744073709551615))  # 2^64 - 1


# ---------------------------------------------------------------------------
# RapidHash64.hash — the byte-span path, one vector per length branch
#
# Generated from upstream rapidhash V3 (`rapidhash.h`, MIT) compiled with the
# default macros, i.e. the non-RAPIDHASH_COMPACT unrolling that `rapidhash()`
# uses. Lengths chosen to cross every branch of `rapidhash_internal`: the <= 16
# sub-cases (0, 1-3, 4-7, 8-16), the nested `if (i > 32/48/64/80/96)` chain, and
# the 112/113 and 224/225 boundaries of the multi-accumulator loop.
# ---------------------------------------------------------------------------


def _iota_span(n: Int) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(n):
        out.append(UInt8(i & 0xFF))
    return out^


def _assert_rapid(n: Int, expected: UInt64) raises:
    var d = _iota_span(n)
    var got = RapidHash64.hash(Span(d))
    if got != expected:
        raise Error(
            String(
                "rapidhash(iota[",
                n,
                "]) = 0x",
                hex(got),
                " expected 0x",
                hex(expected),
            )
        )


def test_rapidhash64_span_short_branches() raises:
    """`len <= 16`: empty, the 1-3 byte shuffle, the 4-7 u32 pair, 8-16 u64."""
    _assert_rapid(0, 0x0338DC4BE2CECDAE)
    _assert_rapid(1, 0x4F23C791B16EBA02)
    _assert_rapid(2, 0xFD31E98C893FA10F)
    _assert_rapid(3, 0xDBD091BCF57AE814)
    _assert_rapid(4, 0x46FEF26DB4943ADF)
    _assert_rapid(5, 0x780350D4A370D3F5)
    _assert_rapid(7, 0x7F403E573BB8EBC1)
    _assert_rapid(8, 0xDA56413FF396AF3E)
    _assert_rapid(9, 0xE48A75B5CBF2AF29)
    _assert_rapid(15, 0x8EC6DFEA933104BB)
    _assert_rapid(16, 0xD6BFC1BCF7E9CA19)


def test_rapidhash64_span_nested_chain() raises:
    """17-112: the `if (i > 32/48/64/80/96)` ladder, one length past each step.
    """
    _assert_rapid(17, 0x7508C9E74D5B5366)
    _assert_rapid(32, 0xC0186990F026B180)
    _assert_rapid(48, 0xECD5ED3E946F9C91)
    _assert_rapid(64, 0xD1A6CC5FE6CF87F4)
    _assert_rapid(80, 0xE7E477A0DFFEAE1F)
    _assert_rapid(96, 0x1E9A8D81B63DE536)
    _assert_rapid(112, 0x667174637FD34AE7)


def test_rapidhash64_span_multi_accumulator() raises:
    """> 112: the six-accumulator path, including both loop boundaries.

    113 enters it, 224 is the last length the `while (i > 224)` loop skips, 225
    is the first it runs. The fold that closes it (`s ^= see1; see2 ^= see3; …`)
    is the one line a summary of the C source omitted, so it is pinned here.
    """
    _assert_rapid(113, 0xABAF0E2BDACF7E23)
    _assert_rapid(224, 0xEEFA9C2E54FC0DF1)
    _assert_rapid(225, 0xF6C6E7081AB8456D)
    _assert_rapid(300, 0x289A29F3A6F04813)


def test_rapidhash64_span_text_vectors() raises:
    var abc = _bytes("abc")
    assert_equal(RapidHash64.hash(Span(abc)), UInt64(0xCB475BEAFA9C0DA2))
    var hello = _bytes("Hello")
    assert_equal(RapidHash64.hash(Span(hello)), UInt64(0x341C16AEF48B463D))
    var marrow = _bytes("marrow")
    assert_equal(RapidHash64.hash(Span(marrow)), UInt64(0x88F5B66AA4FBE618))
    # 69 bytes: the `i > 64` step of the nested chain, then the 16-byte tail
    var long69 = _bytes(
        "the quick brown fox jumps over the lazy dog, repeatedly and at length"
    )
    assert_equal(RapidHash64.hash(Span(long69)), UInt64(0x59AE603799C6601A))


def test_rapidhash64_span_custom_secret() raises:
    """`hash_with` under a generated secret — the opt-in HashDoS path. Values
    are upstream's, computed with `rapidhash_internal(.., make_secret(s))`."""
    var abc = _bytes("abc")
    var iota64 = _iota_span(64)

    var s0 = RapidSecret.make(0)
    assert_equal(
        RapidHash64.hash_with(Span(abc), s0), UInt64(0xF7C375FF4F1E75D8)
    )
    assert_equal(
        RapidHash64.hash_with(Span(iota64), s0), UInt64(0x5BFEC89449579721)
    )

    var s1 = RapidSecret.make(1)
    assert_equal(
        RapidHash64.hash_with(Span(abc), s1), UInt64(0xC775A0A7DE77C6EF)
    )
    assert_equal(
        RapidHash64.hash_with(Span(iota64), s1), UInt64(0xC5E28E87D688EC52)
    )

    var s2 = RapidSecret.make(2)
    assert_equal(
        RapidHash64.hash_with(Span(abc), s2), UInt64(0x1FFF151CEB2E2D84)
    )
    assert_equal(
        RapidHash64.hash_with(Span(iota64), s2), UInt64(0x17F94055E7EC8B3F)
    )


def test_rapidhash64_span_agrees_with_lane_path() raises:
    """The 4- and 8-byte span path and `hash_fixed` are the same branch of
    `rapidhash_internal`, so they must agree — this is what ties the new
    variable-length code to the lane hash the kernels already use."""
    var eight = List[UInt8]()
    for _ in range(8):
        eight.append(0)
    assert_equal(
        RapidHash64.hash(Span(eight)),
        RapidHash64.hash_lanes[8, 1](SIMD[DType.uint64, 1](0))[0],
    )

    var four = List[UInt8]()
    for _ in range(4):
        four.append(0)
    assert_equal(
        RapidHash64.hash(Span(four)),
        RapidHash64.hash_lanes[4, 1](SIMD[DType.uint64, 1](0))[0],
    )

    var v = List[UInt8]()
    v.append(0x78)
    v.append(0x56)
    v.append(0x34)
    v.append(0x12)
    assert_equal(
        RapidHash64.hash(Span(v)),
        RapidHash64.hash_lanes[4, 1](SIMD[DType.uint64, 1](0x12345678))[0],
    )


# ---------------------------------------------------------------------------
# hash_lanes — the pluggable hot path
# ---------------------------------------------------------------------------


def _std_ahash(data: Span[UInt8, _]) -> UInt64:
    """`std.hashlib`'s aHash of `data`, as the oracle for `AHash64.hash`."""
    var h = AHasher[SIMD[DType.uint64, 4](0)]()
    h._update_with_bytes(data)
    return h^.finish()


def _le_bytes(value: UInt64, width: Int) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(width):
        out.append(UInt8((value >> UInt64(i * 8)) & 0xFF))
    return out^


def test_hash_lanes_agrees_with_hash_rapidhash() raises:
    """`hash_lanes[bw]` of a value must equal `hash` of its `bw` little-endian
    bytes. If these ever diverge, a numeric column and the same value arriving
    as bytes would land in different group-by buckets."""
    for v in [UInt64(0), UInt64(1), UInt64(0x12345678), UInt64(0xDEADBEEF)]:
        var b8 = _le_bytes(v, 8)
        assert_equal(
            RapidHash64.hash_lanes[8, 1](SIMD[DType.uint64, 1](v))[0],
            RapidHash64.hash(Span(b8)),
        )
        var masked = v & 0xFFFFFFFF
        var b4 = _le_bytes(masked, 4)
        assert_equal(
            RapidHash64.hash_lanes[4, 1](SIMD[DType.uint64, 1](masked))[0],
            RapidHash64.hash(Span(b4)),
        )


def test_hash_lanes_agrees_with_hash_xxhash() raises:
    """The same invariant for XXH64, whose lane form is a separate derivation
    from its span form — so agreeing is real evidence, not a tautology."""
    for v in [UInt64(0), UInt64(1), UInt64(0x12345678), UInt64(0xCAFEBABE)]:
        var b8 = _le_bytes(v, 8)
        assert_equal(
            XxHash64.hash_lanes[8, 1](SIMD[DType.uint64, 1](v))[0],
            XxHash64.hash(Span(b8)),
        )
        for w in [1, 2, 4]:
            var mask = (UInt64(1) << UInt64(w * 8)) - 1
            var masked = v & mask
            var bw = _le_bytes(masked, w)
            var lanes: UInt64
            if w == 1:
                lanes = XxHash64.hash_lanes[1, 1](
                    SIMD[DType.uint64, 1](masked)
                )[0]
            elif w == 2:
                lanes = XxHash64.hash_lanes[2, 1](
                    SIMD[DType.uint64, 1](masked)
                )[0]
            else:
                lanes = XxHash64.hash_lanes[4, 1](
                    SIMD[DType.uint64, 1](masked)
                )[0]
            assert_equal(lanes, XxHash64.hash(Span(bw)))


def test_hash_lanes_are_independent() raises:
    """Every lane must produce the digest it would alone — the property that
    distinguishes this from `std.hashlib.Hasher._update_with_simd`, which folds
    a vector into one accumulator."""
    var v = SIMD[DType.uint64, 8](
        0, 1, 2, 3, 0xFF, 0x1234, 0xDEADBEEF, ~UInt64(0)
    )
    var wide_rapid = RapidHash64.hash_lanes[8, 8](v)
    var wide_xx = XxHash64.hash_lanes[8, 8](v)
    for i in range(8):
        assert_equal(
            wide_rapid[i],
            RapidHash64.hash_lanes[8, 1](SIMD[DType.uint64, 1](v[i]))[0],
        )
        assert_equal(
            wide_xx[i],
            XxHash64.hash_lanes[8, 1](SIMD[DType.uint64, 1](v[i]))[0],
        )


def test_hash_lanes_separates_widths() raises:
    """`byte_width` participates, so the same bits at 4 and 8 bytes differ."""
    var v = SIMD[DType.uint64, 1](7)
    assert_true(
        RapidHash64.hash_lanes[4, 1](v)[0] != RapidHash64.hash_lanes[8, 1](v)[0]
    )
    assert_true(
        XxHash64.hash_lanes[4, 1](v)[0] != XxHash64.hash_lanes[8, 1](v)[0]
    )


def test_hashers_are_interchangeable_through_the_trait() raises:
    """Both algorithms are reachable generically — the point of the trait."""

    def digest[H: Hasher](data: Span[UInt8, _]) -> UInt64:
        return H.hash(data)

    def lanes[H: Hasher](v: UInt64) -> UInt64:
        return H.hash_lanes[8, 1](SIMD[DType.uint64, 1](v))[0]

    var abc = _bytes("abc")
    assert_equal(digest[RapidHash64](Span(abc)), UInt64(0xCB475BEAFA9C0DA2))
    assert_equal(digest[XxHash64](Span(abc)), UInt64(0x44BC2CF5AD770999))
    assert_true(lanes[RapidHash64](42) != lanes[XxHash64](42))
    assert_equal(RapidHash64.name, StaticString("rapidhash64"))
    assert_equal(XxHash64.name, StaticString("xxhash64"))


def test_combine_lanes_default_is_shared() raises:
    """`combine_lanes` is defaulted on the trait, so both hashers get the same
    fold without either defining one."""
    var a = SIMD[DType.uint64, 4](1, 2, 3, 4)
    var b = SIMD[DType.uint64, 4](10, 20, 30, 40)
    var r = RapidHash64.combine_lanes[4](a, b)
    var x = XxHash64.combine_lanes[4](a, b)
    for i in range(4):
        assert_equal(r[i], x[i])
    # order matters, and combining is not the identity
    var swapped = RapidHash64.combine_lanes[4](b, a)
    assert_true(r[0] != swapped[0])
    assert_true(r[0] != a[0])


def test_ahash64_matches_the_standard_library() raises:
    """`AHash64.hash` delegates, so it must equal `std.hashlib.hash` of the same
    bytes — there is no second aHash implementation here to drift."""
    for n in [0, 1, 3, 8, 17, 64]:
        var d = _iota_span(n)
        assert_equal(AHash64.hash(Span(d)), _std_ahash(Span(d)))


def test_ahash64_lanes_agree_with_hash() raises:
    """The lane form reproduces aHash's short-input path, so it must agree with
    the delegated span hash for the same bytes."""
    for v in [UInt64(0), UInt64(1), UInt64(0x12345678), UInt64(0xFEEDFACE)]:
        var b8 = _le_bytes(v, 8)
        assert_equal(
            AHash64.hash_lanes[8, 1](SIMD[DType.uint64, 1](v))[0],
            AHash64.hash(Span(b8)),
        )


def test_ahash64_lanes_are_independent() raises:
    var v = SIMD[DType.uint64, 4](0, 1, 0xDEAD, ~UInt64(0))
    var wide = AHash64.hash_lanes[8, 4](v)
    for i in range(4):
        assert_equal(
            wide[i], AHash64.hash_lanes[8, 1](SIMD[DType.uint64, 1](v[i]))[0]
        )


def test_three_hashers_are_distinct_and_reachable() raises:
    """All three conform, and they are genuinely different functions."""

    def digest[H: Hasher](data: Span[UInt8, _]) -> UInt64:
        return H.hash(data)

    var abc = _bytes("abc")
    var r = digest[RapidHash64](Span(abc))
    var x = digest[XxHash64](Span(abc))
    var a = digest[AHash64](Span(abc))
    assert_true(r != x)
    assert_true(r != a)
    assert_true(x != a)
    assert_equal(AHash64.name, StaticString("ahash64"))
