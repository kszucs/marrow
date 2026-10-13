# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Decimal arithmetic on unscaled integers: a decimal of scale `s` stores the
value `v` as the integer `v * 10^s`."""

from ..errors import InvalidError


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


def decimal_bound[T: DType](precision: Int) -> Scalar[T]:
    """`10^precision` — the first unscaled magnitude a decimal of `precision`
    digits cannot represent — or 0 for `precision` 0, which is how a
    non-decimal target spells "no digit bound".

    A decimal bounds its *digits* as well as its backing integer, and the two
    are different limits: a 128-bit integer holds a 21-digit value that a
    `decimal128(5, 0)` cannot.
    """
    return pow(Scalar[T](10), precision) if precision > 0 else Scalar[T](0)


def format_decimal[N: DType](value: Scalar[N], scale: Int) -> String:
    """`value` read at `scale`, the way Arrow renders a decimal.

    Every fractional digit the scale declares is written, trailing zeros
    included — `150` at scale 2 is `1.50`, never `1.5` — because the scale is
    part of the value's type and a reader of the text should be able to see
    it. A magnitude below one keeps its leading `0`, so `-5` at scale 2 is
    `-0.05`.

    Built from the integer's own text rather than by dividing by `10^scale`,
    which is what keeps the most negative value right: its magnitude does not
    fit the signed type, but its digits do.
    """
    var text = String(value)
    var negative = value < 0
    var digits = text[byte= 1 if negative else 0:]
    var out = String("-") if negative else String()
    var n = digits.byte_length()
    if scale <= 0:
        out += digits
    elif n <= scale:
        out += "0."
        for _ in range(scale - n):
            out += "0"
        out += digits
    else:
        out += digits[byte = 0 : n - scale]
        out += "."
        out += digits[byte = n - scale :]
    return out^


def parse_decimal[
    N: DType
](
    text: StringSlice, digits: Int, bound: Scalar[N], scale: Int
) raises InvalidError -> Scalar[N]:
    """Parse `text` as a decimal of `scale`, unscaled, in `N`.

    The grammar: an optional sign, digits with at most one `.`, and an
    optional exponent — `+1.5`, `.5`, `5.`, `1.5e-1` are all accepted, and
    whitespace is not. The parsed value is rescaled to `scale` exactly: a
    digit the scale cannot hold raises rather than rounding (`1.555` is not a
    `decimal(10, 2)`), and so does a magnitude of `bound` (`10^precision`) or
    more.

    Overflow cannot happen silently. Leading zeros are not counted, and a
    value with more than `digits` significant digits raises before the next
    one is accumulated; `N` holds `digits` of them with a digit to spare, so
    `10 * acc + d` always fits.
    """
    var b = text.as_bytes()
    var n = len(b)
    var negative, i = _sign(b, 0)
    var acc = Scalar[N](0)
    var significant = 0
    var fraction = 0
    var seen_dot = False
    var seen_digit = False
    while i < n:
        var c = Int(b[i])
        if c >= ord("0") and c <= ord("9"):
            seen_digit = True
            if acc != 0 or c != ord("0"):
                significant += 1
                if significant > digits:
                    raise InvalidError(t"decimal: '{text}' has too many digits")
            acc = acc * 10 + Scalar[N](c - ord("0"))
            if seen_dot:
                fraction += 1
        elif c == ord(".") and not seen_dot:
            seen_dot = True
        elif c == ord("e") or c == ord("E"):
            break
        else:
            raise InvalidError(t"decimal: '{text}' is not a decimal number")
        i += 1
    if not seen_digit:
        raise InvalidError(t"decimal: '{text}' is not a decimal number")

    var delta = scale - fraction
    if i < n:
        delta += _exponent(text, i + 1, digits)
    if acc == 0:
        return acc
    elif delta >= 0:
        if significant + delta > digits:
            raise InvalidError(t"decimal: '{text}' is out of range")
        acc *= pow(Scalar[N](10), delta)
    elif -delta > digits:
        raise InvalidError(t"decimal: rescaling '{text}' would cause data loss")
    else:
        var f = pow(Scalar[N](10), -delta)
        if acc % f != 0:
            raise InvalidError(
                t"decimal: rescaling '{text}' would cause data loss"
            )
        acc //= f
    if acc >= bound:
        raise InvalidError(t"decimal: '{text}' does not fit its precision")
    return -acc if negative else acc


def _sign(b: Span[Byte, _], i: Int) -> Tuple[Bool, Int]:
    """An optional `+` or `-` at `b[i]`: whether it negates, and the index
    after it."""
    if i < len(b) and Int(b[i]) == ord("-"):
        return (True, i + 1)
    elif i < len(b) and Int(b[i]) == ord("+"):
        return (False, i + 1)
    return (False, i)


def _exponent(
    text: StringSlice, start: Int, digits: Int
) raises InvalidError -> Int:
    """The signed exponent after an `e`, which begins at `start`. Bounded by
    `2 * digits` so a runaway exponent is an error, not an overflowing `Int`.
    """
    var b = text.as_bytes()
    var negative, i = _sign(b, start)
    if i >= len(b):
        raise InvalidError(t"decimal: '{text}' is not a decimal number")
    var exponent = 0
    while i < len(b):
        var c = Int(b[i])
        if c < ord("0") or c > ord("9"):
            raise InvalidError(t"decimal: '{text}' is not a decimal number")
        exponent = exponent * 10 + (c - ord("0"))
        if exponent > 2 * digits:
            raise InvalidError(t"decimal: '{text}' is out of range")
        i += 1
    return -exponent if negative else exponent
