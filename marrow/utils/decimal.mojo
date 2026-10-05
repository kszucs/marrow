# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Decimal arithmetic on unscaled integers: a decimal of scale `s` stores the
value `v` as the integer `v * 10^s`."""


def float_to_decimal(
    x: Float64, scale: Int, precision: Int
) -> Optional[Int256]:
    """The integer nearest `x * 10^scale`, ties to even, computed exactly; or
    `None` when `x` is not finite or the result needs more than `precision`
    digits. `scale` and `precision` must each be in `[0, 76]`.
    """
    # `x` is exactly `mant * 2^k`, so `x * 10^scale` is `mant * 5^scale`
    # shifted by `k + scale` bits. With `scale <= 76` the product stays below
    # 2^231, and the early bound keeps a left shift below 2^255.
    var magnitude = abs(x)
    if not (magnitude <= pow(Float64(10), precision - scale)):
        return None  # NaN, an infinity, or far too many digits
    var bits = UInt64(magnitude.to_bits())
    var biased = Int((bits >> 52) & 0x7FF)
    var mant = bits & ((UInt64(1) << 52) - 1)
    if biased != 0:
        mant |= UInt64(1) << 52
    else:
        biased = 1  # subnormal: no implicit bit, the smallest exponent
    var shift = biased - 1075 + scale
    var r = mant.cast[DType.int256]() * pow(Int256(5), scale)
    if shift >= 0:
        r <<= Int256(shift)
    elif shift < -240:
        r = 0  # below 2^-9: rounds to zero
    else:
        var n = Int256(-shift)
        var q = r >> n
        var rem = r - (q << n)
        var half = Int256(1) << (n - 1)
        if rem > half or (rem == half and (q & 1) == 1):
            q += 1
        r = q
    if r >= pow(Int256(10), precision):
        return None
    return -r if x < 0 else r
