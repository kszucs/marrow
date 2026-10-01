# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(timezone('Europe/Budapest', ts) AS VARCHAR) AS t FROM events

    Reading a naive timestamp as Budapest wall-clock time, then rendering the
    instant in UTC: CET in winter (one hour back), CEST in summer (two).

    DuckDB renders a TIMESTAMPTZ in the session zone, pinned to UTC for the
    corpus; Arrow renders it in the zone on its type. marrow follows Arrow, so
    the plan converts to UTC before rendering, which is what DuckDB does
    implicitly. The SQL frontend has no `timezone()`.

    -- expected
    t:string
    '2020-12-31 23:00:00+00'
    '2021-06-15 10:30:45+00'
    '2021-06-15 10:30:45+00'
    NULL
    '2020-02-29 22:59:59+00'
    '2021-12-31 22:59:59.999999+00'
    """
    var t = table("events")
    return t.project(
        ["t"],
        [
            col("ts", timestamp(microsecond))
            .assume_timezone("Europe/Budapest")
            .convert_timezone("UTC")
            .cast(string)
        ],
    )
