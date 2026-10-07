# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Finding a table's metadata and opening it.

A table is named by its `metadata.json` — read from any storage marrow can
read — or, for a local table, by its directory: the version named in
`metadata/version-hint.text` if there is one, else the newest metadata file,
which is what DuckDB's `iceberg_scan` does without a catalog. Object stores
cannot be listed here, so a remote table is named by its metadata file.

[Reference](https://iceberg.apache.org/spec/#file-system-tables)
"""

from std.os import listdir
from std.os.path import exists, isdir

from ..errors import InvalidError, KeyError
from ..io import DynSource
from ..io.uri import StorageOptions, Uri

from .metadata import TableMetadata, relocate

comptime METADATA_SUFFIX = ".metadata.json"
comptime VERSION_HINT = "version-hint.text"


def _read_text(
    uri: String, options: StorageOptions = StorageOptions()
) raises -> String:
    """The whole of a small text file — metadata, a version hint."""
    var source = DynSource.open(uri, options)
    var data = source.read_at(0, source.size())
    return String(StringSlice(unsafe_from_utf8=data))


def _local_path(uri: String) raises -> Optional[String]:
    """The filesystem path behind a bare path or `file://` URI."""
    var u = Uri.parse(uri)
    if u.scheme == "":
        return uri
    if u.scheme == "file":
        return "/" + u.path
    return None


def _version(name: String) -> Optional[Int]:
    """The version a metadata file name carries: `v3.metadata.json` (the
    file-system layout) or `00003-<uuid>.metadata.json` (what catalogs
    write)."""
    if not name.endswith(METADATA_SUFFIX):
        return None
    var stem = String(name.removesuffix(METADATA_SUFFIX))
    var digits = String(stem.removeprefix("v")).split("-")[0]
    try:
        return Int(digits)
    except:
        return None


def metadata_location(table: String) raises -> String:
    """The metadata file `table` names: itself if it is one, else the current
    version in a local table directory."""
    if table.endswith(METADATA_SUFFIX):
        return table
    var local = _local_path(table)
    if not local or not isdir(local.value()):
        raise InvalidError(
            t"iceberg: '{table}' is neither a metadata file nor a local table"
            t" directory"
        )
    var dir = local.value() + "/metadata"
    var hint = dir + "/" + VERSION_HINT
    var wanted: Optional[Int] = None
    if exists(hint):
        try:
            wanted = Int(_read_text(hint).strip())
        except:
            raise InvalidError(t"iceberg: {hint} does not hold a version")
    var best = String()
    var best_version = -1
    for name in listdir(dir):
        var version = _version(name)
        if not version:
            continue
        var v = version.value()
        if wanted and v != wanted.value():
            continue
        if v > best_version:
            best_version = v
            best = name
    if best_version < 0:
        raise KeyError(t"iceberg: no metadata file in {dir}")
    return dir + "/" + best


def _table_root(metadata_path: String) -> String:
    """The directory holding `metadata/`, which a table's recorded `location`
    is relocated to."""
    var cut = metadata_path.rfind("/metadata/")
    if cut < 0:
        return metadata_path
    return String(metadata_path[byte=0:cut])


@fieldwise_init
struct IcebergTable(Copyable, Movable):
    """A table's metadata, the directory it was actually opened from, and
    the storage options every file of it is read with."""

    var metadata: TableMetadata
    var root: String
    """Where `metadata.location` resolves to now: the table was written
    under `location`, and may since have been copied or moved."""
    var options: StorageOptions
    """Credentials and service settings for the table's storage; the
    environment fills what they leave unset."""

    @staticmethod
    def open(
        table: String, options: StorageOptions = StorageOptions()
    ) raises -> Self:
        """A table by its metadata file, or a local table directory."""
        var path = metadata_location(table)
        var metadata = TableMetadata.parse(_read_text(path, options))
        return Self(metadata^, _table_root(path), options.copy())

    def relocate(self, path: String) -> String:
        """`path`, as the table's metadata records it, where it can be
        opened now: under `root` rather than the recorded `location`."""
        return relocate(path, self.metadata.location, self.root)
