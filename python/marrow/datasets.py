# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Datasets as lazy tables, shaped like Hugging Face's ``datasets``.

    from marrow.datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm.filter(col("question").char_length() > 300).collect()

``load_dataset`` resolves one split to its files -- from the Hub's dataset
card, or from the ``data_files`` given -- and returns a
:class:`~marrow.lazy.LazyTable` scanning them. Nothing is read beyond the
first file's schema until the plan runs. Parquet, newline-delimited JSON and
Arrow IPC files are read directly; a Hub dataset stored in another format
(CSV, compressed JSON) is read from the Hub's Parquet conversion of it.
``HF_TOKEN`` is used for private and gated datasets.
"""

from . import libmarrow as _ma
from .lazy import LazyTable


def _data_files(data_files):
    """``data_files`` as ``{split: [pattern, ...]}``: a pattern or a list of
    them is ``train``'s, as ``datasets`` reads it."""
    if data_files is None:
        return None
    if not isinstance(data_files, dict):
        data_files = {"train": data_files}
    return {
        str(split): [
            str(p) for p in ([patterns] if isinstance(patterns, str) else patterns)
        ]
        for split, patterns in data_files.items()
    }


def load_dataset(path, name=None, split="train", data_files=None, revision=None):
    """A lazy table over one split of a dataset.

    Parameters
    ----------
    path : str
        A Hub dataset (``"openai/gsm8k"``), or a format -- ``"parquet"``,
        ``"json"`` or ``"arrow"`` -- reading ``data_files``.
    name : str, optional
        The config of a Hub dataset; its default one when omitted.
    split : str, default "train"
        The split to read.
    data_files : str, list of str or dict, optional
        Paths, URIs or glob patterns: one pattern or a list of them for the
        ``train`` split, or a mapping of split names to them. Relative to the
        repository for a Hub dataset; ``hf://``, ``s3://`` and other URIs
        otherwise.
    revision : str, optional
        A branch, tag or commit of a Hub dataset.
    """
    files = _data_files(data_files)
    return LazyTable.wrap(
        _ma.load_dataset(str(path), name or "", split, files, revision or "")
    )


def get_dataset_config_names(path, revision=None):
    """The config names of a Hub dataset."""
    return _ma.hub_config_names(str(path), revision or "")


def get_dataset_split_names(path, config_name=None, revision=None):
    """The split names of a Hub dataset's config, its default one when
    omitted."""
    return _ma.hub_split_names(str(path), config_name or "", revision or "")
