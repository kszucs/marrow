# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`FileFormat`: what a data file is stored as, and the scan that reads it."""

from ..errors import InvalidError, NotImplementedError
from ..expr.builders import scan, scan_avro, scan_ipc, scan_json
from ..expr.logical import DynRelation
from ..formats import Format
from ..formats.avro import AvroFormat
from ..formats.ipc import IpcFormat
from ..formats.json import JsonFormat
from ..formats.parquet import ParquetFormat
from ..io import StorageOptions
from ..schema import Schema


def _names[F: Format](file: String) -> Bool:
    """Whether `file`, lower-cased, carries one of `F`'s extensions."""
    for ext in F.extensions():
        if file.endswith(ext):
            return True
    return False


def _schema[F: Format](path: String, options: StorageOptions) raises -> Schema:
    """The schema of the file at `path`, as `F` reads it."""
    return F.open(path, None, options).schema()


struct FileFormat(Equatable, ImplicitlyCopyable, Movable, Writable):
    """The format of a data file: `parquet`, `json` (newline-delimited),
    `arrow` (the IPC file format) or `avro`, the ones marrow reads, each
    recognised by the extensions its `Format` names; `OTHER` for data it
    cannot -- CSV, anything compressed -- and `NONE` for a file holding no
    data, a README say.
    """

    var name: StaticString

    comptime PARQUET = Self("parquet")
    comptime JSON = Self("json")
    comptime ARROW = Self("arrow")
    comptime AVRO = Self("avro")
    comptime OTHER = Self("other")
    comptime NONE = Self("none")

    @staticmethod
    def named(name: StringSlice) -> Optional[Self]:
        """The readable format called `name`, as `datasets` names its
        builders, or `None`."""
        for format in [Self.PARQUET, Self.JSON, Self.ARROW, Self.AVRO]:
            if name == format.name:
                return format
        return None

    @staticmethod
    def of(path: StringSlice) -> Self:
        """The format `path`'s extension names."""
        var file = String(path).lower()
        if _names[ParquetFormat](file):
            return Self.PARQUET
        if _names[JsonFormat](file):
            return Self.JSON
        if _names[IpcFormat](file):
            return Self.ARROW
        if _names[AvroFormat](file):
            return Self.AVRO
        var other: List[StaticString] = [
            ".json",
            ".csv",
            ".tsv",
            ".txt",
            ".xml",
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

    def scan(
        self,
        var paths: List[String],
        options: StorageOptions = StorageOptions(),
    ) raises -> DynRelation:
        """One scan over `paths`, all in this format, read in order, each
        opened with `options`.

        Its schema is the first file's own -- a Parquet or Arrow footer, a
        JSON file's first block, an Avro header -- so this opens that file;
        every other is opened only when the scan reaches it, and must match.
        """
        if len(paths) == 0:
            raise InvalidError("datasets: no files to read")
        if not self.is_readable():
            raise NotImplementedError(
                t"datasets: the files, {paths[0]} first, are not all in one "
                t"format marrow reads: parquet, json, arrow or avro"
            )
        if self == Self.PARQUET:
            var schema = _schema[ParquetFormat](paths[0], options)
            return scan(paths^, schema^, options.copy())
        if self == Self.JSON:
            var schema = _schema[JsonFormat](paths[0], options)
            return scan_json(paths^, schema^, options.copy())
        if self == Self.AVRO:
            var schema = _schema[AvroFormat](paths[0], options)
            return scan_avro(paths^, schema^, options.copy())
        var schema = _schema[IpcFormat](paths[0], options)
        return scan_ipc(paths^, schema^, options.copy())

    def __eq__(self, other: Self) -> Bool:
        return self.name == other.name

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.name)
