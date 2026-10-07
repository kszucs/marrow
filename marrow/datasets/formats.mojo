# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`FileFormat`: what a data file is stored as, and the scan that reads it."""

from ..errors import InvalidError, NotImplementedError
from ..expr.builders import scan, scan_ipc, scan_json
from ..expr.logical import DynRelation
from ..io import DynSource
from ..ipc import RecordBatchFileReader
from ..json import open_json
from ..parquet.reader import ParquetFile


struct FileFormat(Equatable, ImplicitlyCopyable, Movable, Writable):
    """The format of a data file: `parquet`, `json` (newline-delimited) or
    `arrow` (the IPC file format), the ones marrow reads; `OTHER` for data it
    cannot -- CSV, anything compressed -- and `NONE` for a file holding no
    data, a README say.

    A `.json` file is `OTHER`: it may hold one JSON document rather than a
    line per row, and only `.jsonl` and `.ndjson` say which.
    """

    var name: StaticString

    comptime PARQUET = Self("parquet")
    comptime JSON = Self("json")
    comptime ARROW = Self("arrow")
    comptime OTHER = Self("other")
    comptime NONE = Self("none")

    @staticmethod
    def named(name: StringSlice) -> Optional[Self]:
        """The readable format called `name`, as `datasets` names its
        builders, or `None`."""
        for format in [Self.PARQUET, Self.JSON, Self.ARROW]:
            if name == format.name:
                return format
        return None

    @staticmethod
    def of(path: StringSlice) -> Self:
        """The format `path`'s extension names."""
        var file = String(path).lower()
        if file.endswith(".parquet"):
            return Self.PARQUET
        if file.endswith(".jsonl") or file.endswith(".ndjson"):
            return Self.JSON
        if file.endswith(".arrow"):
            return Self.ARROW
        var other: List[StaticString] = [
            ".json",
            ".csv",
            ".tsv",
            ".txt",
            ".xml",
            ".avro",
            ".orc",
            ".gz",
            ".bz2",
            ".xz",
            ".zst",
            ".lz4",
            ".zip",
        ]
        for ext in other:
            if file.endswith(ext):
                return Self.OTHER
        return Self.NONE

    @staticmethod
    def shared(paths: List[String]) -> Self:
        """The format every path names; `OTHER` when they name several, and
        `NONE` when there are none."""
        if len(paths) == 0:
            return Self.NONE
        var first = Self.of(paths[0])
        for ref p in paths:
            if Self.of(p) != first:
                return Self.OTHER
        return first

    def __init__(out self, name: StaticString):
        self.name = name

    def is_readable(self) -> Bool:
        """Whether `scan` reads files of this format."""
        return self != Self.OTHER and self != Self.NONE

    def scan(self, var paths: List[String]) raises -> DynRelation:
        """One scan over `paths`, all in this format, read in order.

        Its schema is the first file's own -- a Parquet or Arrow footer, or a
        JSON file's first block -- so this opens that file; every other is
        opened only when the scan reaches it, and must match.
        """
        if len(paths) == 0:
            raise InvalidError("datasets: no files to read")
        if not self.is_readable():
            raise NotImplementedError(
                t"datasets: the files, {paths[0]} first, are not all in one "
                t"format marrow reads: parquet, json or arrow"
            )
        if self == Self.PARQUET:
            var schema = ParquetFile(DynSource.open(paths[0])).schema()
            return scan(paths^, schema^)
        if self == Self.JSON:
            var schema = open_json(paths[0]).schema.copy()
            return scan_json(paths^, schema^)
        var schema = RecordBatchFileReader[DynSource](
            DynSource.open(paths[0])
        ).schema.copy()
        return scan_ipc(paths^, schema^)

    def __eq__(self, other: Self) -> Bool:
        return self.name == other.name

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.name)
