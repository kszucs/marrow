# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Hexadecimal digits."""


def hex_digit(c: UInt8) -> Int:
    """The value of one hex digit, or -1 if it is not one."""
    var v = Int(c)
    if v >= ord("0") and v <= ord("9"):
        return v - ord("0")
    if v >= ord("a") and v <= ord("f"):
        return v - ord("a") + 10
    if v >= ord("A") and v <= ord("F"):
        return v - ord("A") + 10
    return -1
