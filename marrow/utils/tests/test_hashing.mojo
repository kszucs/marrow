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

from ..hashing import Hasher, RapidHash64, XxHash64


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
        RapidHash64.mix(RapidHash64.SECRET1, RapidHash64.SECRET2),
        UInt64(0x422765567D8FBFD6),
    )


def test_rapidhash64_secrets_match_reference() raises:
    """The three secrets are rapidhash.h's, and a typo in one silently changes
    every hash the library produces."""
    assert_equal(RapidHash64.SECRET1, UInt64(0x8BB84B93962EACC9))
    assert_equal(RapidHash64.SECRET2, UInt64(0x4B33A62ED433D4A3))
    assert_equal(RapidHash64.SECRET7, UInt64(0xAAAAAAAAAAAAAAAA))


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
    var wide = RapidHash64.mum_wide[4](a, b)
    var mixed = RapidHash64.mix_wide[4](a, b)
    for i in range(4):
        var scalar = RapidHash64.mum(a[i], b[i])
        assert_equal(wide[0][i], scalar[0])
        assert_equal(wide[1][i], scalar[1])
        assert_equal(mixed[i], RapidHash64.mix(a[i], b[i]))


def test_rapidhash64_hash_fixed_separates_widths() raises:
    """`byte_width` is folded into the seed, so the same bits hashed as 4 and as
    8 bytes give different digests — that is what keeps int32 and int64 columns
    from colliding wholesale."""
    assert_equal(RapidHash64.hash_fixed[4](7), RapidHash64.hash_fixed[4](7))
    assert_true(RapidHash64.hash_fixed[4](7) != RapidHash64.hash_fixed[8](7))
    assert_true(RapidHash64.hash_fixed[8](0) != RapidHash64.hash_fixed[8](1))


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
