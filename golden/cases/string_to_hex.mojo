# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT to_hex(s) AS h FROM words

    Each UTF-8 byte as two upper-case digits: `héllo`'s `é` is two bytes and
    so four digits, and the empty string encodes to the empty string.

    -- skip python

    -- expected
    h:string
    '48656C6C6F'
    '774F524C44'
    '20207061642020'
    ''
    '68C3A96C6C6F'
    NULL
    """
    var t = table("words")
    return t.project(["h"], [to_hex(col("s"))])
