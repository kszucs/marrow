# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for reading newline-delimited JSON: marrow vs pyarrow and Polars.

Every reader reads the same file under the same ``group`` so pytest-benchmark
prints them side by side. Run with::

    pixi run -e bench pytest python/marrow/tests/bench_json.py --benchmark

marrow's reader is single-threaded, so pyarrow is measured twice: with
``use_threads=False``, which is the like-for-like row, and with its default
thread pool. Polars runs with its default thread count. Two shapes: ``flat``
(int, float, string, bool) and ``nested`` (a struct and a list besides).
"""

import json
import random

import polars as pl
import pyarrow.json as pj
import pytest

import marrow.json as mj


SIZES = [100_000, 1_000_000]
SHAPES = ["flat", "nested"]


def _row(rng, i, shape):
    row = {
        "id": i,
        "price": round(rng.uniform(0, 1000), 2),
        "name": f"item-{rng.randint(0, 10_000)}",
        "ok": rng.random() < 0.5,
    }
    if shape == "nested":
        row["tags"] = [rng.randint(0, 9) for _ in range(rng.randint(0, 4))]
        row["dims"] = {"w": rng.randint(1, 100), "h": rng.randint(1, 100)}
    return row


@pytest.fixture(
    params=[(s, n) for s in SHAPES for n in SIZES],
    ids=[f"{s}-n={n}" for s in SHAPES for n in SIZES],
    scope="session",
)
def path(request, tmp_path_factory):
    shape, n = request.param
    rng = random.Random(42)
    out = tmp_path_factory.mktemp("json") / f"{shape}-{n}.jsonl"
    with open(out, "w") as f:
        for i in range(n):
            f.write(json.dumps(_row(rng, i, shape)))
            f.write("\n")
    return out


@pytest.mark.benchmark(group="read_json")
def test_read_json_marrow(benchmark, path):
    benchmark(lambda: mj.read_json(path))


@pytest.mark.benchmark(group="read_json")
def test_read_json_pyarrow_single_thread(benchmark, path):
    options = pj.ReadOptions(use_threads=False)
    benchmark(lambda: pj.read_json(path, read_options=options))


@pytest.mark.benchmark(group="read_json")
def test_read_json_pyarrow(benchmark, path):
    benchmark(lambda: pj.read_json(path))


@pytest.mark.benchmark(group="read_json")
def test_read_json_polars(benchmark, path):
    benchmark(lambda: pl.read_ndjson(path))
