# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Scan planning: from a snapshot to the files to read.

A snapshot's manifest list names its manifests, and the live entries of those
are the table's files at that snapshot — data files to read, delete files to
apply to them. Which data files a query can skip is not decided here: that
needs its predicates, which live in `marrow.expr` (`Index.from_manifest`), so
planning is two steps with the choice in between —

    var files = snapshot_files(table, snapshot_id)
    var tasks = scan_tasks(table, files)      # or with `keep` from a read plan

[Reference](https://iceberg.apache.org/spec/#scan-planning)
"""

from ..execution import ExecContext

from .catalog import IcebergTable
from .manifest import (
    DataFile,
    ManifestEntry,
    ManifestFile,
    read_manifest,
    read_manifest_list,
)
from .scan import FileScanTask, POSITION_DELETE_FILE_PATH_ID


@fieldwise_init
struct SnapshotFiles(Copyable, Movable):
    """The live entries of one snapshot, data and deletes apart. Each entry
    carries the id of the partition spec its manifest was written with."""

    var data: List[ManifestEntry]
    var deletes: List[ManifestEntry]


def snapshot_files(
    table: IcebergTable,
    snapshot_id: Int,
    ctx: ExecContext = ExecContext.auto(),
) raises -> SnapshotFiles:
    """The snapshot's live entries, in manifest-list order.

    The manifests are read concurrently on the I/O pool: one at a time under
    `ExecContext.serial()`, at most `num_threads` under `parallel(n)`, one
    per I/O thread under `auto()`. A failure is the one the first failing
    manifest raised, as a serial read would report."""
    var snapshot = table.metadata.snapshot(snapshot_id)
    var manifests = read_manifest_list(
        table.relocate(snapshot.manifest_list), table.options
    )
    var n = len(manifests)
    var slots = List[Optional[List[ManifestEntry]]](capacity=n)
    for _ in range(n):
        slots.append(None)

    def read(wid: Int, i: Int) raises {mut slots, imm}:
        ref manifest = manifests[i]
        slots[i] = read_manifest(
            table.relocate(manifest.path), manifest, table.options
        )

    ctx.fan_out_blocking(n, read)

    var files = SnapshotFiles([], [])
    for i in range(n):
        var is_data = manifests[i].content == ManifestFile.DATA
        for var entry in slots[i].take():
            if not entry.is_live():
                continue
            if is_data:
                files.data.append(entry^)
            else:
                files.deletes.append(entry^)
    return files^


def _may_name(deletes: DataFile, data_path: String) raises -> Bool:
    """Whether the position delete file `deletes` may hold rows for the data
    file recorded at `data_path`: false only when its `file_path` column's
    bounds both exist and exclude it. A truncated lower bound is still at or
    below every value, and a truncated upper bound is incremented, so it is
    still at or above every value."""
    comptime key = POSITION_DELETE_FILE_PATH_ID
    if key not in deletes.lower_bounds or key not in deletes.upper_bounds:
        return True
    # Bytes compared as UTF-8 text, which orders them bytewise.
    var path = StringSlice(data_path)
    var lower = StringSlice(unsafe_from_utf8=Span(deletes.lower_bounds[key]))
    var upper = StringSlice(unsafe_from_utf8=Span(deletes.upper_bounds[key]))
    return lower <= path and path <= upper


def scan_tasks(
    table: IcebergTable,
    files: SnapshotFiles,
    keep: Optional[List[Bool]] = None,
) raises -> List[FileScanTask]:
    """One task per data file `keep` does not drop (all, without it), with
    the delete files that may apply to it, every path relocated to where the
    table now lives.

    A position delete applies to data at or below its sequence number, and
    only to the files its `file_path` bounds admit; an equality delete
    applies to data strictly below its sequence number. Delete files are not
    matched on partition: a position delete names the file it deletes from,
    and the reader matches on that, so a delete the bounds cannot exclude is
    read and contributes nothing — slower, never wrong."""
    # Delete files are only ever opened, so they are relocated once here.
    var deletes = List[DataFile](capacity=len(files.deletes))
    for ref d in files.deletes:
        var file = d.data_file.copy()
        file.file_path = table.relocate(file.file_path)
        deletes.append(file^)

    var tasks = List[FileScanTask]()
    for i in range(len(files.data)):
        if keep and not keep.value()[i]:
            continue
        ref entry = files.data[i]
        ref path = entry.data_file.file_path
        var seq = entry.sequence_number.or_else(0)
        var positions = List[DataFile]()
        var equalities = List[DataFile]()
        for j in range(len(files.deletes)):
            ref d = files.deletes[j]
            var delete_seq = d.sequence_number.or_else(0)
            if d.data_file.content == DataFile.POSITION_DELETES:
                if delete_seq >= seq and _may_name(d.data_file, path):
                    positions.append(deletes[j].copy())
            elif delete_seq > seq:
                equalities.append(deletes[j].copy())
        tasks.append(
            FileScanTask(
                file=entry.data_file.copy(),
                path=table.relocate(path),
                spec=table.metadata.spec(entry.spec_id),
                position_deletes=positions^,
                equality_deletes=equalities^,
            )
        )
    return tasks^
