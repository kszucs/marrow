# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(timezone('UTC', ts) AS VARCHAR) AS t FROM events

    Interpreting a naive timestamp in a zone, and rendering the result as
    text: the expectation block's `timestamp` is zone-free by construction.
    A zoned timestamp renders with its offset, so attaching UTC is `+00` on
    every row, and a fraction appears only on the row that has one.

    DuckDB renders a TIMESTAMPTZ in its session zone, which the corpus pins to
    UTC, so this is also the only zone in which DuckDB's reading and Arrow's
    (the zone on the type) coincide without a conversion.
    `temporal_timezone_convert` covers a zone with an offset.

    -- expected
    t:string
    '2021-01-01 00:00:00+00'
    '2021-06-15 12:30:45+00'
    '2021-06-15 12:30:45+00'
    NULL
    '2020-02-29 23:59:59+00'
    '2021-12-31 23:59:59.999999+00'
    """
    var t = table("events")
    return t.project(
        ["t"],
        [col("ts", timestamp(microsecond)).assume_timezone("UTC").cast(string)],
    )
