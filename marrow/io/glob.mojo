# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`Glob`: a path or URI pattern, matched against keys and expanded into the
locations it names by listing a directory or an object store."""

from std.builtin.sort import sort
from std.os import listdir
from std.os.path import isdir, join

from ..errors import InvalidError
from .opendal import OpenDalStore
from .uri import StorageOptions, Uri


struct Glob(Copyable, Movable, Writable):
    """A path or URI whose key may hold `*` (any run of characters within one
    path segment), `**` (any run of segments) and `?` (one character other
    than `/`), as `datasets` and `fsspec` spell `data_files`. `**/` also
    matches no directory at all, and `\\` makes the character after it
    literal. In a URI, `?` starts the query, as it does in any URI, so there
    it is not a wildcard.
    """

    var pattern: String

    def __init__(out self, var pattern: String):
        self.pattern = pattern^

    @staticmethod
    def escape(path: StringSlice) -> String:
        """A pattern matching exactly `path`."""
        var out = String()
        for c in path.codepoints():
            if (
                c == Codepoint.ord("*")
                or c == Codepoint.ord("?")
                or c == Codepoint.ord("\\")
            ):
                out += "\\"
            out += String(c)
        return out^

    def matches(self, path: StringSlice) -> Bool:
        """Whether `path` matches, both read as `/`-separated relative paths."""
        return Self._match(self.pattern.as_bytes(), 0, path.as_bytes(), 0)

    def expand(
        self, options: StorageOptions = StorageOptions()
    ) raises -> List[String]:
        """The locations the pattern names, sorted; a pattern matching
        nothing raises.

        A literal pattern is returned unchecked, so naming one file costs no
        listing, and an `http(s)` URL with a query is one file. Otherwise the
        directory before the first wildcard is listed
        -- recursively when what follows spans directories -- from the
        filesystem for a local path and through OpenDAL for anything else,
        whose service must be able to list: `http(s)://` cannot.
        """
        var u = Uri.parse(self.pattern)
        var key = u.local_path() if u.is_local() else u.object_key()
        if key.find("*") < 0 and (u.scheme != "" or key.find("?") < 0):
            return [self.pattern.copy()]
        # The query, which every expanded URI carries on, is what follows the
        # `?` of a URI; a bare path has none.
        var query = String()
        if u.scheme != "" and self.pattern.find("?") >= 0:
            query = String(self.pattern[byte = self.pattern.find("?") :])
        var head = Self._base(self.pattern, u.scheme == "")
        var dir = Self._base(key, u.scheme == "")
        var rest = String(key[byte = dir.byte_length() :])
        if u.scheme != "":
            # A `?` in a URI's key was written `%3F`, so it is a character.
            rest = rest.replace("\\", "\\\\").replace("?", "\\?")
        var glob = Glob(rest^)
        var recursive = (
            glob.pattern.find("/") >= 0 or glob.pattern.find("**") >= 0
        )
        var listed = List[String]()
        if u.is_local():
            listed = Self._walk(dir, "", recursive)
        elif u.scheme == "http" or u.scheme == "https":
            raise InvalidError(
                t"datasets: cannot expand '{self.pattern}': http(s) cannot "
                t"list a directory, so name each file"
            )
        else:
            var store = OpenDalStore(u.service(), options.resolve(u))
            for ref entry in store.list(dir, recursive=recursive):
                if entry.startswith(dir) and not entry.endswith("/"):
                    listed.append(String(entry[byte = dir.byte_length() :]))
        var found = List[String]()
        for ref rel in listed:
            if not glob.matches(rel):
                continue
            # A URI's path is percent-decoded when it is parsed again; a bare
            # path is taken literally.
            if u.scheme == "":
                found.append(String(head, rel))
            else:
                found.append(String(head, Uri.quote(rel), query))
        if len(found) == 0:
            raise InvalidError(t"datasets: no files match '{self.pattern}'")
        sort(found)
        return found^

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.pattern)

    @staticmethod
    def _base(pattern: StringSlice, question: Bool) -> String:
        """`pattern` up to and including the last `/` before its first
        wildcard -- `*`, and `?` too when `question` -- or empty."""
        var b = pattern.as_bytes()
        var end = 0
        for i in range(len(b)):
            if b[i] == UInt8(ord("*")) or (
                question and b[i] == UInt8(ord("?"))
            ):
                break
            if b[i] == UInt8(ord("/")):
                end = i + 1
        return String(pattern[byte=:end])

    @staticmethod
    def _walk(
        dir: String, prefix: String, recursive: Bool
    ) raises -> List[String]:
        """The files under `dir` as paths relative to it, behind `prefix`;
        its subdirectories too, with a trailing `/`, unless `recursive`."""
        var out = List[String]()
        var base = dir if dir != "" else String(".")
        for name in listdir(base):
            var rel = String(prefix, name)
            if isdir(join(base, name)):
                if recursive:
                    out.extend(
                        Self._walk(
                            String(join(base, name), "/"),
                            String(rel, "/"),
                            True,
                        )
                    )
                else:
                    out.append(String(rel, "/"))
            else:
                out.append(rel^)
        return out^

    @staticmethod
    def _match(p: Span[UInt8, _], pi: Int, s: Span[UInt8, _], si: Int) -> Bool:
        comptime STAR = UInt8(ord("*"))
        comptime QUESTION = UInt8(ord("?"))
        comptime SLASH = UInt8(ord("/"))
        comptime ESCAPE = UInt8(ord("\\"))
        if pi == len(p):
            return si == len(s)
        if p[pi] == STAR:
            if pi + 1 < len(p) and p[pi + 1] == STAR:
                var rest = pi + 2
                if rest < len(p) and p[rest] == SLASH:
                    if Self._match(p, rest + 1, s, si):
                        return True
                for k in range(si, len(s) + 1):
                    if Self._match(p, rest, s, k):
                        return True
                return False
            for k in range(si, len(s) + 1):
                if Self._match(p, pi + 1, s, k):
                    return True
                if k < len(s) and s[k] == SLASH:
                    return False
            return False
        if si == len(s):
            return False
        if p[pi] == ESCAPE and pi + 1 < len(p):
            return p[pi + 1] == s[si] and Self._match(p, pi + 2, s, si + 1)
        if p[pi] == QUESTION:
            return s[si] != SLASH and Self._match(p, pi + 1, s, si + 1)
        return p[pi] == s[si] and Self._match(p, pi + 1, s, si + 1)
