# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""``marrow.datasets.load_dataset`` over local files, in every spelling of
``data_files``. The Hub side is covered offline by the Mojo suite."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marrow import col
from marrow.datasets import load_dataset


@pytest.fixture
def shards(tmp_path):
    for i in range(3):
        table = pa.table({"id": list(range(i * 10, i * 10 + 10))})
        pq.write_table(table, tmp_path / f"train-{i}.parquet")
    pq.write_table(pa.table({"id": [100, 101]}), tmp_path / "test-0.parquet")
    return tmp_path


def test_a_pattern_is_the_train_split(shards):
    t = load_dataset("parquet", data_files=str(shards / "train-*.parquet"))
    assert "ParquetScan" in t.explain()
    assert t.collect().column("id").to_pylist() == list(range(30))


def test_a_mapping_names_the_splits(shards):
    files = {
        "train": str(shards / "train-*.parquet"),
        "test": [str(shards / "test-0.parquet")],
    }
    t = load_dataset("parquet", data_files=files, split="test")
    assert t.collect().column("id").to_pylist() == [100, 101]


def test_a_query_runs_over_every_shard(shards):
    t = load_dataset("parquet", data_files=str(shards / "train-*.parquet"))
    out = t.filter(col("id") >= 25).optimize().collect()
    assert out.column("id").to_pylist() == list(range(25, 30))


def test_json_lines(tmp_path):
    for name, rows in [("a", [1, 2]), ("b", [3])]:
        lines = "".join(json.dumps({"id": r}) + "\n" for r in rows)
        (tmp_path / f"{name}.jsonl").write_text(lines)
    t = load_dataset("json", data_files=str(tmp_path / "*.jsonl"))
    assert t.collect().column("id").to_pylist() == [1, 2, 3]


def test_a_missing_split_is_a_key_error(shards):
    files = {"train": str(shards / "train-*.parquet")}
    with pytest.raises(KeyError, match="no split 'test'"):
        load_dataset("parquet", data_files=files, split="test")
