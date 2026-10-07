# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`load_dataset` -- a dataset's split as a plan, shaped like
`datasets.load_dataset`."""

from ..errors import InvalidError
from ..expr.logical import DynRelation
from .files import DataFiles
from .formats import FileFormat
from .huggingface import HubDataset


def load_dataset(
    path: String,
    name: String = "",
    split: String = "train",
    revision: String = "",
) raises -> DynRelation:
    """A Hub dataset's split, as a scan over every file in it.

    `path` names the dataset (`owner/name`), `name` its config -- the
    default one when empty -- and `revision` a branch, tag or commit. Only
    the Hub's listing and the first file's schema are read here; the data is
    read when the plan runs. A split stored in a format marrow cannot read,
    or with no data files of its own, is read from the Hub's Parquet
    conversion of it:

    ```mojo
    var gsm = load_dataset("openai/gsm8k", "main", split="test")
    var long = gsm.filter(col("question").char_length() > 300).execute()
    ```
    """
    if FileFormat.named(path):
        raise InvalidError(
            t"datasets: load_dataset('{path}') reads files, so it needs "
            t"data_files"
        )
    var hub = HubDataset.fetch(path, revision)
    var config = hub.config(name)
    # A config with no data files of its own -- images, audio, a loading
    # script -- has splits only in the Hub's conversion.
    if len(config.data_files.splits()):
        var files = hub.locate(config.data_files, split)
        var format = FileFormat.shared(files)
        if format.is_readable():
            return format.scan(files^)
    return FileFormat.PARQUET.scan(hub.parquet_conversion(config.name, split))


def load_dataset(
    path: String,
    data_files: DataFiles,
    split: String = "train",
    revision: String = "",
) raises -> DynRelation:
    """A split of `data_files`, as a scan over every file it names.

    `path` is a format -- `parquet`, `json` or `arrow` -- and the patterns
    are paths or URIs (`hf://`, `s3://`, …), globs expanded by listing; or
    `path` is a Hub dataset and the patterns are relative to its root.

    ```mojo
    var js = load_dataset("json", DataFiles(["/data/train-*.jsonl"]))
    ```
    """
    var format = FileFormat.named(path)
    if format:
        return format.value().scan(data_files.resolve(split))
    var files = HubDataset.fetch(path, revision).locate(data_files, split)
    return FileFormat.shared(files).scan(files^)
