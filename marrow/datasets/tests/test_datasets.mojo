# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`marrow.datasets`: data files, Hub metadata, and `load_dataset` plans
over local shards in each format. Nothing here touches the network: the Hub's
answers are parsed from captured JSON."""

from std.os.path import join
from std.testing import assert_equal, assert_false, assert_raises, assert_true

from ...arrays import DynArray
from ...builders import array
from ...dtypes import field, int64, string
from ...ipc import write_ipc_file
from ...parquet.writer import write_table
from ...schema import Schema
from ...tabular import RecordBatch, Table, record_batch
from ...utils.testing import ScratchDir
from ...expr.builders import col, lit
from ...expr.logical import IpcScan, JsonScan, ParquetScan
from ...expr.optimizer import AllRules
from ...io.uri import Uri
from ..files import DataFiles
from ..formats import FileFormat
from ..huggingface import HubConfig, HubDataset
from ..load import load_dataset


def _ids(start: Int, stop: Int) raises -> DynArray:
    var ids = List[Optional[Int]]()
    for i in range(start, stop):
        ids.append(i)
    return array(ids^, int64).to_dyn()


def _batch(start: Int, n: Int) raises -> RecordBatch:
    var names = List[Optional[String]]()
    for i in range(start, start + n):
        names.append(String("row", i))
    return record_batch(
        [_ids(start, start + n), array(names^)], names=["id", "name"]
    )


def _write_parquet(path: String, start: Int, n: Int) raises:
    var b = _batch(start, n)
    write_table(Table.from_batches(b.schema.copy(), [b.copy()]), path)


def _touch(path: String) raises:
    with open(path, "w") as f:
        f.write("")


def _bytes(text: String) -> List[UInt8]:
    return List[UInt8](text.as_bytes())


def _files(*patterns: String) -> DataFiles:
    var out = List[String]()
    for p in patterns:
        out.append(p)
    return DataFiles(out^)


# ---------------------------------------------------------------------------
# Data files and formats
# ---------------------------------------------------------------------------


def test_data_files_list_is_train() raises:
    var files = _files("a/*.parquet", "b/*.parquet")
    assert_equal(len(files.splits()), 1)
    assert_equal(files.splits()[0], "train")
    assert_equal(len(files.patterns("train")), 2)
    with assert_raises(contains="no split 'test'"):
        _ = files.patterns("test")


def test_data_files_resolve_drops_duplicates() raises:
    with ScratchDir() as dir:
        _touch(join(dir, "a.parquet"))
        _touch(join(dir, "b.parquet"))
        var files = _files(join(dir, "a.parquet"), join(dir, "*.parquet"))
        var got = files.resolve("train")
        assert_equal(len(got), 2)
        assert_equal(got[0], join(dir, "a.parquet"))
        assert_equal(got[1], join(dir, "b.parquet"))


def test_file_format_reads_the_extension() raises:
    assert_true(FileFormat.of("a/train-0.parquet") == FileFormat.PARQUET)
    assert_true(FileFormat.of("x.jsonl") == FileFormat.JSON)
    assert_true(FileFormat.of("x.json") == FileFormat.OTHER)
    assert_true(FileFormat.of("x.arrow") == FileFormat.ARROW)
    assert_true(FileFormat.of("Iris.csv") == FileFormat.OTHER)
    assert_true(FileFormat.of("x.jsonl.gz") == FileFormat.OTHER)
    assert_true(FileFormat.of("README.md") == FileFormat.NONE)
    assert_true(FileFormat.named("json").value() == FileFormat.JSON)
    assert_false(FileFormat.named("csv"))


def test_file_format_shared_needs_one_format() raises:
    var same: List[String] = ["a.parquet", "b.parquet"]
    var mixed: List[String] = ["a.parquet", "b.jsonl"]
    assert_true(FileFormat.shared(same) == FileFormat.PARQUET)
    assert_true(FileFormat.shared(mixed) == FileFormat.OTHER)
    assert_true(FileFormat.shared(List[String]()) == FileFormat.NONE)
    assert_false(FileFormat.OTHER.is_readable())


# ---------------------------------------------------------------------------
# Hub metadata
# ---------------------------------------------------------------------------


def _gsm8k_info() -> String:
    """The Hub's `api/datasets/openai/gsm8k`, trimmed to what is read."""
    return (
        '{"id": "openai/gsm8k", "sha": "740312ad", "cardData": {"configs": ['
        + '{"config_name": "main", "data_files": ['
        + '{"split": "train", "path": "main/train-*"},'
        + '{"split": "test", "path": "main/test-*"}]},'
        + '{"config_name": "socratic", "data_files": ['
        + '{"split": "train", "path": "socratic/train-*"},'
        + '{"split": "test", "path": "socratic/test-*"}]}]},'
        + '"siblings": [{"rfilename": ".gitattributes"},'
        + '{"rfilename": "README.md"},'
        + '{"rfilename": "main/test-00000-of-00001.parquet"},'
        + '{"rfilename": "main/train-00000-of-00001.parquet"},'
        + '{"rfilename": "socratic/test-00000-of-00001.parquet"},'
        + '{"rfilename": "socratic/train-00000-of-00001.parquet"}]}'
    )


def _gsm8k(revision: String = "") raises -> HubDataset:
    return HubDataset.parse("openai/gsm8k", revision, _bytes(_gsm8k_info()))


def test_hub_configs_come_from_the_card() raises:
    var ds = _gsm8k()
    var names = ds.config_names()
    assert_equal(len(names), 2)
    assert_equal(names[0], "main")
    assert_equal(names[1], "socratic")
    var splits = ds.config("main").data_files.splits()
    assert_equal(len(splits), 2)
    assert_equal(splits[0], "train")
    assert_equal(splits[1], "test")
    var files = ds.locate(ds.config("socratic").data_files, "test")
    assert_equal(len(files), 1)
    assert_equal(
        files[0],
        "hf://datasets/openai/gsm8k/socratic/test-00000-of-00001.parquet",
    )


def test_hub_locates_files_at_the_revision() raises:
    var got = _gsm8k("v1").locate(_files("main/train-*"), "train")
    assert_equal(len(got), 1)
    assert_equal(
        got[0],
        "hf://datasets/openai/gsm8k@v1/main/train-00000-of-00001.parquet",
    )


def test_hub_uri_escapes_the_file_name() raises:
    var ds = _gsm8k()
    var uri = ds.uri("data/a%20b?.parquet")
    assert_equal(uri, "hf://datasets/openai/gsm8k/data/a%2520b%3F.parquet")
    assert_equal(Uri.parse(uri).object_key(), "data/a%20b?.parquet")


def test_hub_locate_lists_each_file_once() raises:
    assert_equal(
        len(_gsm8k().locate(_files("main/*", "main/train-*"), "train")), 2
    )


def test_hub_refuses_a_revision_with_a_slash() raises:
    with assert_raises(contains="name its commit instead"):
        _ = HubDataset.fetch("o/n", "refs/pr/1")


def test_hub_several_configs_need_a_name() raises:
    var ds = _gsm8k()
    with assert_raises(contains="several configs"):
        _ = ds.config()
    with assert_raises(contains="no config 'nope'"):
        _ = ds.config("nope")
    with assert_raises(contains="no split 'validation'"):
        _ = ds.config("main").data_files.patterns("validation")


def test_hub_data_files_spellings() raises:
    var info = String(
        '{"cardData": {"configs": ['
        + '{"config_name": "one", "data_files": "data/*.jsonl"},'
        + '{"config_name": "map", "default": true, "data_files": '
        + '{"train": ["a/*.jsonl", "b/*.jsonl"], "test": "t.jsonl"}}]},'
        + '"siblings": [{"rfilename": "data/x.jsonl"},'
        + '{"rfilename": "a/1.jsonl"}, {"rfilename": "b/2.jsonl"},'
        + '{"rfilename": "t.jsonl"}]}'
    )
    var ds = HubDataset.parse("o/n", "", _bytes(info))
    assert_equal(ds.config().name, "map")
    var map = ds.config("map").data_files.copy()
    assert_equal(len(ds.locate(map, "train")), 2)
    assert_equal(ds.locate(map, "test")[0], "hf://datasets/o/n/t.jsonl")
    var one = ds.config("one").data_files.copy()
    assert_equal(one.splits()[0], "train")
    assert_equal(ds.locate(one, "train")[0], "hf://datasets/o/n/data/x.jsonl")


def test_hub_splits_are_inferred_without_a_card() raises:
    var files: List[String] = [
        ".gitattributes",
        "README.md",
        "data/train-00000-of-00002.parquet",
        "data/train-00001-of-00002.parquet",
        "data/validation-00000-of-00001.parquet",
        "test/part.parquet",
        "notes.txt",
    ]
    var config = HubConfig.infer(files)
    assert_equal(config.name, "default")
    var splits = config.data_files.splits()
    assert_equal(len(splits), 3)
    assert_equal(splits[0], "train")
    assert_equal(len(config.data_files.patterns("train")), 2)
    assert_equal(splits[1], "validation")
    assert_equal(splits[2], "test")
    assert_equal(
        String(config.data_files.patterns("test")[0]), "test/part.parquet"
    )


def test_hub_without_split_names_is_all_train() raises:
    var files: List[String] = ["README.md", "Iris.csv", "database.sqlite"]
    var config = HubConfig.infer(files)
    assert_equal(len(config.data_files.splits()), 1)
    assert_equal(String(config.data_files.patterns("train")[0]), "Iris.csv")


def test_hub_refuses_an_answer_of_another_shape() raises:
    with assert_raises(contains="siblings is not a list"):
        _ = HubDataset.parse("o/n", "", _bytes('{"siblings": {}}'))
    with assert_raises(contains="rfilename is not a string"):
        _ = HubDataset.parse(
            "o/n", "", _bytes('{"siblings": [{"rfilename": 3}]}')
        )
    with assert_raises(contains="config_name is not a string"):
        _ = HubDataset.parse(
            "o/n", "", _bytes('{"cardData": {"configs": [{"config_name": 1}]}}')
        )
    with assert_raises(contains="file URL is not a string"):
        _ = HubDataset.parse_urls(_bytes("[1]"))


def test_hub_parquet_listing() raises:
    var urls = HubDataset.parse_urls(
        _bytes(
            '["https://huggingface.co/api/datasets/o/n/parquet/default/'
            + 'train/0.parquet"]'
        )
    )
    assert_equal(len(urls), 1)
    with assert_raises(contains="busier than usual"):
        _ = HubDataset.parse_urls(
            _bytes('{"error": "The server is busier than usual"}')
        )


# ---------------------------------------------------------------------------
# load_dataset over local files
# ---------------------------------------------------------------------------


def test_load_dataset_parquet_reads_every_shard() raises:
    with ScratchDir() as dir:
        _write_parquet(join(dir, "train-0.parquet"), 0, 5)
        _write_parquet(join(dir, "train-1.parquet"), 5, 5)
        _write_parquet(join(dir, "test-0.parquet"), 100, 3)
        var rel = load_dataset("parquet", _files(join(dir, "train-*.parquet")))
        assert_true(rel.isa[ParquetScan]())
        var out = rel.execute()
        assert_equal(out.num_rows(), 10)
        assert_true(out.column("id") == _ids(0, 10))


def test_load_dataset_picks_the_split() raises:
    with ScratchDir() as dir:
        _write_parquet(join(dir, "train-0.parquet"), 0, 5)
        _write_parquet(join(dir, "test-0.parquet"), 100, 3)
        var files = DataFiles()
        files.add("train", [join(dir, "train-*.parquet")])
        files.add("test", [join(dir, "test-*.parquet")])
        var out = load_dataset("parquet", files, split="test").execute()
        assert_equal(out.num_rows(), 3)
        with assert_raises(contains="no split 'validation'"):
            _ = load_dataset("parquet", files, split="validation")


def test_load_dataset_filters_across_shards() raises:
    with ScratchDir() as dir:
        for i in range(3):
            _write_parquet(
                join(dir, String("part-", i, ".parquet")), i * 10, 10
            )
        var plan = load_dataset(
            "parquet", _files(join(dir, "part-*.parquet"))
        ).filter(col("id", int64) >= lit(15, int64))
        var out = plan.optimize[AllRules]().execute()
        assert_equal(out.num_rows(), 15)
        assert_true(out.column("id") == _ids(15, 30))


def test_load_dataset_names_a_mismatched_shard() raises:
    with ScratchDir() as dir:
        _write_parquet(join(dir, "a.parquet"), 0, 2)
        var other = record_batch(
            [array([1, 2], int64).to_dyn()], names=["other"]
        )
        var bad = join(dir, "b.parquet")
        write_table(
            Table.from_batches(other.schema.copy(), [other.copy()]), bad
        )
        var rel = load_dataset("parquet", _files(join(dir, "*.parquet")))
        with assert_raises(contains="b.parquet"):
            _ = rel.execute()


def test_load_dataset_json_reads_every_file() raises:
    with ScratchDir() as dir:
        with open(join(dir, "a.jsonl"), "w") as f:
            f.write('{"id": 1, "name": "a"}\n{"id": 2, "name": "b"}\n')
        with open(join(dir, "b.jsonl"), "w") as f:
            f.write('{"id": 3, "name": "c"}\n')
        var rel = load_dataset("json", _files(join(dir, "*.jsonl")))
        assert_true(rel.isa[JsonScan]())
        var out = rel.execute()
        assert_equal(out.num_rows(), 3)
        assert_true(out.column("id") == array([1, 2, 3], int64).to_dyn())


def test_load_dataset_arrow_reads_every_file() raises:
    with ScratchDir() as dir:
        write_ipc_file(join(dir, "a.arrow"), [_batch(0, 2)])
        write_ipc_file(join(dir, "b.arrow"), [_batch(2, 2), _batch(4, 1)])
        var rel = load_dataset("arrow", _files(join(dir, "*.arrow")))
        assert_true(rel.isa[IpcScan]())
        var out = rel.execute()
        assert_equal(out.num_rows(), 5)
        assert_true(out.column("id") == array([0, 1, 2, 3, 4], int64).to_dyn())


def test_load_dataset_format_needs_data_files() raises:
    with assert_raises(contains="needs data_files"):
        _ = load_dataset("parquet")


def test_load_dataset_refuses_an_unreadable_format() raises:
    with ScratchDir() as dir:
        _touch(join(dir, "a.csv"))
        with assert_raises(contains="neither a format marrow reads"):
            _ = load_dataset("csv", _files(join(dir, "a.csv")))


def test_load_dataset_refuses_a_malformed_repo() raises:
    with assert_raises(contains="neither a format marrow reads"):
        _ = load_dataset("not-a-repo")
