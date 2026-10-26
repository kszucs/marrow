# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT to_base64(from_hex(to_hex(s))) AS b FROM words

    Padded base64 of each element's UTF-8 bytes: lengths that are not a
    multiple of three end in `=` or `==`. `from_hex(to_hex(s))` is the text's
    UTF-8 bytes as a blob, which is what DuckDB's `to_base64` takes.

    -- skip python

    -- expected
    b:string
    'SGVsbG8='
    'd09STEQ='
    'ICBwYWQgIA=='
    ''
    'aMOpbGxv'
    NULL
    """
    var t = table("words")
    return t.project(["b"], [to_base64(from_hex(to_hex(col("s"))))])
