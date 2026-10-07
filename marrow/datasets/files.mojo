# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`DataFiles`: the patterns of each split of a dataset."""

from ..errors import KeyError
from ..io.glob import Glob
from ..io.uri import StorageOptions


struct DataFiles(Copyable, Movable, Writable):
    """The patterns of each split, in the order the splits were named.

    A list of patterns is the `train` split's, as `datasets` reads
    `data_files`.
    """

    var _splits: Dict[String, List[String]]

    def __init__(out self):
        self._splits = {}

    @implicit
    def __init__(out self, var patterns: List[String]):
        self._splits = {"train": patterns^}

    def add(mut self, split: String, var patterns: List[String]) raises:
        """Add `patterns` to `split`, naming it if it is new."""
        if split in self._splits:
            self._splits[split].extend(patterns^)
        else:
            self._splits[split] = patterns^

    def splits(self) -> List[String]:
        """The split names."""
        var out = List[String]()
        for split in self._splits.keys():
            out.append(split.copy())
        return out^

    def patterns(self, split: String) raises -> List[Glob]:
        """The patterns of `split`; a `KeyError` naming the splits there are
        when it is not one."""
        if split not in self._splits:
            raise KeyError(t"datasets: no split '{split}' in {self}")
        var out = List[Glob]()
        for ref p in self._splits[split]:
            out.append(Glob(p.copy()))
        return out^

    def resolve(
        self, split: String, options: StorageOptions = StorageOptions()
    ) raises -> List[String]:
        """The locations `split`'s patterns expand to, each once, in pattern
        order."""
        var out = List[String]()
        for ref glob in self.patterns(split):
            for ref location in glob.expand(options):
                if location not in out:
                    out.append(location.copy())
        return out^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("{")
        var first = True
        for entry in self._splits.items():
            if not first:
                writer.write(", ")
            first = False
            writer.write(entry.key, ": [", String(", ").join(entry.value), "]")
        writer.write("}")
