# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The Apache-2.0 header every source file carries.

`python -m devkit license check` lists the files missing it and `fix` adds it.
`pixi run fmt` runs `fix`, so the lint job's `fmt_check` fails on a file
committed without one.

The header is two lines, in each format's own comment syntax: the copyright
line from NOTICE.txt and an SPDX identifier naming the licence exactly, which
is what licence scanners and SBOM tools read -- the Linux kernel's form rather
than the thirteen-line boilerplate from LICENSE.txt's appendix.  A file that
also carries code under another licence names it too, as `Apache-2.0 AND MIT`;
the identifier must still start with `Apache-2.0`.

The header opens the file, except after a `#!` line -- the kernel reads it only
from the first two bytes -- and after an XML declaration, which must open the
document.  A `.qmd` page's front matter is still found below it, provided a
blank line separates the two, and one always does.

A file derived from another project carries that project's attribution in prose
*below* the header, with the licence texts in NOTICE.txt.  The check asks only
for the header, so the attribution stays free text.

Every file git knows about is covered, and a format with no rule here is an
error rather than a skip: silently passing over what it does not recognise is
exactly what disqualifies `addlicense`, which skips every `.mojo` and `.qmd`
file.  skywalking-eyes, the checker most Apache projects use, does the job
given a hand-written Mojo mapping, but it is a Go binary more in the
environment for what this module does with the standard library.
"""

import re
import subprocess

TEXT = """\
Copyright 2024 Szűcs Krisztián
SPDX-License-Identifier: Apache-2.0"""

SPDX = "SPDX-License-Identifier: "


class Comment:
    """A format's comment syntax: a prefix on every line, or a pair of lines
    around the block."""

    def __init__(self, prefix=None, opening=None, closing=None):
        self._prefix = prefix
        self._opening = opening
        self._closing = closing

    def render(self, text):
        if self._prefix is None:
            return [self._opening, *text.splitlines(), self._closing]
        return [f"{self._prefix} {line}".rstrip() for line in text.splitlines()]


HASH = Comment(prefix="#")
SLASHES = Comment(prefix="//")
DASHES = Comment(prefix="--")
HTML = Comment(opening="<!--", closing="-->")
C = Comment(opening="/*", closing="*/")


class Placement:
    """Where the header goes in one format, and how it is commented there.

    `after` matches a first line the header must follow rather than precede.
    """

    def __init__(self, comment, after=None):
        self.comment = comment
        self._after = re.compile(after) if after else None

    def split(self, lines):
        """`(prolog, rest)` -- the lines that stay above the header, and the
        lines below it."""
        if self._after and lines and self._after.match(lines[0]):
            return lines[:1], lines[1:]
        return [], lines


SCRIPT = Placement(HASH, after=r"#!")

PLACEMENTS = {
    ".mojo": SCRIPT,
    ".py": SCRIPT,
    ".sh": SCRIPT,
    ".toml": SCRIPT,
    ".yml": SCRIPT,
    ".yaml": SCRIPT,
    ".ini": SCRIPT,
    ".md": Placement(HTML),
    ".html": Placement(HTML),
    ".xml": Placement(HTML, after=r"<\?xml "),
    ".css": Placement(C),
    ".scss": Placement(SLASHES),
    ".lua": Placement(DASHES),
    ".qmd": Placement(HTML),
    # Dotfiles have no suffix, so these are matched on the whole name.
    ".dockerignore": SCRIPT,
    ".envrc": SCRIPT,
    ".gitattributes": SCRIPT,
    ".gitignore": SCRIPT,
}

#: Files that take no header: formats with no comment syntax (JSON, and the
#: `.theme` files, which are JSON carrying their own `license` field),
#: binaries, lockfiles, the licence texts themselves, images -- the logos are
#: drawn by `docs/assets/logo.py`, which does carry it -- and files a tool
#: reads whole.
EXEMPT = {
    ".arrow",
    ".arrow_file",
    ".avro",
    ".gitkeep",
    ".json",
    ".lock",
    ".parquet",
    ".png",
    ".python-version",
    ".stream",
    ".svg",
    ".theme",
    ".txt",
}


class LicenseHeaders:
    """The header over one checkout."""

    def __init__(self, root):
        self._root = root

    @staticmethod
    def placement(path):
        """The `Placement` for *path*, `None` if it is exempt; `KeyError` for a
        format with no rule, which is a decision to make, not a file to skip."""
        name = path.rsplit("/", 1)[-1]
        if name.startswith("Dockerfile"):
            return SCRIPT
        if name not in PLACEMENTS and name not in EXEMPT:
            name = "." + name.rsplit(".", 1)[-1]
        if name in EXEMPT:
            return None
        return PLACEMENTS[name]

    @classmethod
    def has_header(cls, path, text):
        """True when *text* carries the header where *path*'s format puts it."""
        placement = cls.placement(path)
        _, rest = placement.split(text.splitlines())
        header = placement.comment.render(TEXT)
        found = rest[: len(header)]
        return len(found) == len(header) and all(
            line == want or (SPDX in want and line.startswith(want + " AND "))
            for line, want in zip(found, header)
        )

    @classmethod
    def with_header(cls, path, text):
        """*text* with the header added, or unchanged if it already has one."""
        if cls.has_header(path, text):
            return text
        placement = cls.placement(path)
        prolog, rest = placement.split(text.splitlines())
        while rest and not rest[0].strip():
            rest.pop(0)
        lines = [*prolog, *placement.comment.render(TEXT)]
        if rest:
            lines += ["", *rest]
        return "\n".join(lines) + "\n"

    def files(self):
        """Every file git knows about: tracked, plus untracked and not ignored.

        A symlink is left to its target, which is listed in its own right.
        """
        listed = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=self._root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        paths = sorted(set(listed.split("\0")) - {""})
        return [
            p
            for p in paths
            if (self._root / p).is_file() and not (self._root / p).is_symlink()
        ]

    def scan(self):
        """`(missing, unknown)` -- the files without the header, and those in a
        format with no rule."""
        missing, unknown = [], []
        for path in self.files():
            try:
                placement = self.placement(path)
            except KeyError:
                unknown.append(path)
                continue
            if placement is not None:
                text = (self._root / path).read_text(encoding="utf-8")
                if not self.has_header(path, text):
                    missing.append(path)
        return missing, unknown

    def fix(self):
        """Add the header wherever it is missing; `(fixed, unknown)`."""
        missing, unknown = self.scan()
        for path in missing:
            file = self._root / path
            file.write_text(
                self.with_header(path, file.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
        return missing, unknown
