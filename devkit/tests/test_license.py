# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The licence header: where each format takes it, and what the check reports."""

import subprocess

import pytest
from click.testing import CliRunner

from devkit.cli import Context, cli
from devkit.license import TEXT, LicenseHeaders
from devkit.mojo import Repo

HASHED = [f"# {line}".rstrip() for line in TEXT.splitlines()]
HTML = ["<!--", *TEXT.splitlines(), "-->"]


def added(path, text):
    return LicenseHeaders.with_header(path, text).splitlines()


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def test_the_header_opens_a_source_file_and_a_blank_line_follows():
    lines = added("marrow/x.mojo", '"""Doc."""\n\ndef f():\n    pass\n')

    assert lines[: len(HASHED)] == HASHED
    assert lines[len(HASHED) :] == ["", '"""Doc."""', "", "def f():", "    pass"]


def test_a_shebang_stays_on_the_first_line():
    """The kernel reads `#!` only from the first two bytes of the file."""
    lines = added("scripts/x.sh", "#!/usr/bin/env bash\necho hi\n")

    assert lines[0] == "#!/usr/bin/env bash"
    assert lines[1 : 1 + len(HASHED)] == HASHED


def test_an_xml_declaration_stays_on_the_first_line():
    lines = added("docs/theme/x.xml", '<?xml version="1.0"?>\n<language/>\n')

    assert lines[0] == '<?xml version="1.0"?>'
    assert lines[1 : 1 + len(HTML)] == HTML


def test_a_page_keeps_a_blank_line_between_the_header_and_its_front_matter():
    """Pandoc reads a metadata block that does not open the page only when a
    blank line precedes it; without one, `title:` would render as text."""
    lines = added("docs/x.qmd", '---\ntitle: "X"\n---\n\nBody.\n')

    assert lines == [*HTML, "", "---", 'title: "X"', "---", "", "Body."]


def test_leading_blank_lines_collapse_into_the_one_after_the_header():
    """Golden cases open with a blank line; the header must not double it."""
    lines = added("golden/cases/x.mojo", "\nfrom golden.prelude import *\n")

    assert lines == [*HASHED, "", "from golden.prelude import *"]


def test_an_empty_file_becomes_just_the_header():
    assert added("marrow/tests/__init__.mojo", "") == HASHED


def test_adding_the_header_twice_is_adding_it_once():
    once = LicenseHeaders.with_header("devkit/x.py", "import os\n")

    assert LicenseHeaders.with_header("devkit/x.py", once) == once
    assert LicenseHeaders.has_header("devkit/x.py", once)


def test_an_attribution_below_the_header_still_counts_as_having_it():
    """A derived file adds the upstream project's notice in prose after the
    header; the check asks only for the header itself."""
    text = "\n".join([*HASHED, "", "# Ported from Apache Arrow.", "", "x = 1", ""])

    assert LicenseHeaders.has_header("devkit/x.py", text)


def test_a_file_mixing_in_other_licences_names_them_after_apache():
    """`hashing.mojo` ports MIT and BSD code, so its identifier says so."""
    text = "\n".join(
        [HASHED[0], "# SPDX-License-Identifier: Apache-2.0 AND MIT", "", "x = 1", ""]
    )

    assert LicenseHeaders.has_header("devkit/x.py", text)


@pytest.mark.parametrize(
    "identifier", ["MIT", "Apache-2.0-or-later", "MIT AND Apache-2.0"]
)
def test_an_identifier_not_led_by_apache_is_not_the_header(identifier):
    text = "\n".join([HASHED[0], f"# SPDX-License-Identifier: {identifier}", ""])

    assert not LicenseHeaders.has_header("devkit/x.py", text)


def test_a_header_in_the_wrong_place_is_not_the_header():
    text = "\n".join(["import os", *HASHED, ""])

    assert not LicenseHeaders.has_header("devkit/x.py", text)


@pytest.mark.parametrize(
    "path",
    ["pixi.lock", "LICENSE.txt", "docs/assets/logo.svg", "golden/fixtures/a.arrow"],
)
def test_files_that_cannot_or_need_not_carry_it_are_exempt(path):
    assert LicenseHeaders.placement(path) is None


@pytest.mark.parametrize("path", ["Dockerfile.ci", "docs/.gitignore", ".envrc"])
def test_dockerfiles_and_dotfiles_are_matched_on_their_name(path):
    assert LicenseHeaders.placement(path) is not None


@pytest.mark.parametrize("path", ["Makefile", "src/x.rs"])
def test_a_format_with_no_rule_is_an_error_not_a_skip(path):
    with pytest.raises(KeyError):
        LicenseHeaders.placement(path)


# ---------------------------------------------------------------------------
# Over a checkout
# ---------------------------------------------------------------------------


def git_repo(tmp_path, files):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def test_scan_covers_untracked_files_and_skips_ignored_ones_and_symlinks(tmp_path):
    root = git_repo(
        tmp_path,
        {
            ".gitignore": LicenseHeaders.with_header(".gitignore", "build/\n"),
            "a.py": "x = 1\n",
            "b.mojo": LicenseHeaders.with_header("b.mojo", "def f():\n    pass\n"),
            "build/c.py": "x = 1\n",
            "notes.rst": "Notes.\n",
            "data.json": "{}\n",
        },
    )
    (root / "d.py").symlink_to("a.py")

    missing, unknown = LicenseHeaders(root).scan()

    assert missing == ["a.py"]
    assert unknown == ["notes.rst"]


def test_fix_leaves_the_checkout_clean(tmp_path):
    root = git_repo(tmp_path, {"a.py": "x = 1\n", "docs/b.qmd": "---\nt: 1\n---\n"})
    headers = LicenseHeaders(root)

    fixed, unknown = headers.fix()

    assert fixed == ["a.py", "docs/b.qmd"]
    assert unknown == []
    assert headers.scan() == ([], [])


def test_the_check_command_fails_until_the_fix_command_has_run(tmp_path):
    root = git_repo(tmp_path, {"a.py": "x = 1\n"})
    context = Context(repo=Repo(root))

    before = CliRunner().invoke(cli, ["license", "check"], obj=context)
    fix = CliRunner().invoke(cli, ["license", "fix"], obj=context)
    after = CliRunner().invoke(cli, ["license", "check"], obj=context)

    assert before.exit_code == 1
    assert "a.py: no license header" in before.output
    assert fix.exit_code == 0
    assert after.exit_code == 0, after.output
