"""The docs snippet check: what gets collected, and what gets wrapped."""

from devkit.docs import SnippetCheck, as_program
from devkit.mojo import Repo


def repo_with_docs(tmp_path, pages=(), snippets=()):
    """A scratch tree holding just the parts the check reads."""
    (tmp_path / Repo.MARKER).write_text("")
    repo = Repo(tmp_path)
    repo.snippets_dir.mkdir(parents=True)
    for name, text in snippets:
        (repo.snippets_dir / name).write_text(text)
    for name, text in pages:
        (repo.docs_dir / name).write_text(text)
    return repo


def collect(repo):
    return SnippetCheck(repo, toolchain=None).listings()


# ---------------------------------------------------------------------------
# Wrapping an inline block
# ---------------------------------------------------------------------------


def test_imports_are_lifted_above_the_generated_main():
    """A block reads as statements, but `from ... import` is only legal at
    module scope, so wrapping the block whole would fail to compile every
    listing that imports anything -- which is all of them."""
    program = as_program("from marrow import array\nvar a = array([1], int32)\n")
    lines = program.splitlines()

    assert lines[0] == "from marrow import array"
    assert "def main() raises:" in lines
    assert lines[-1] == "    var a = array([1], int32)"


def test_a_block_that_declares_main_is_taken_as_written():
    """Wrapping it again would nest `main` inside `main`."""
    source = "def main() raises:\n    pass\n"
    assert as_program(source) == source


def test_blank_lines_between_the_imports_and_the_body_are_dropped():
    """They would otherwise land between `def main():` and its first
    statement, which is a body Mojo cannot parse."""
    program = as_program("from marrow import array\n\n\nvar a = 1\n")
    assert "def main() raises:\n    var a = 1" in program


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def test_snippet_files_are_collected_verbatim(tmp_path):
    """The page includes the file, so the rendered listing and the compiled
    bytes must be the same thing -- no wrapping."""
    source = "def main():\n    pass\n"
    repo = repo_with_docs(tmp_path, snippets=[("a.mojo", source)])

    listings, _ = collect(repo)

    assert listings == [("docs/snippets/a.mojo", source)]


def test_an_inline_fence_is_collected_and_wrapped(tmp_path):
    repo = repo_with_docs(tmp_path, pages=[("guide.qmd", "```mojo\nvar a = 1\n```\n")])

    listings, fragments = collect(repo)

    assert [label for label, _ in listings] == ["guide.qmd#0"]
    assert "def main() raises:" in listings[0][1]
    assert fragments == 0


def test_a_fragment_opts_out_and_is_counted(tmp_path):
    """The count is the point: a guide can always be made to pass by marking
    every block a fragment, so the number has to be visible."""
    page = "```{.mojo .fragment}\nfn f(x: Int)\n```\n"
    repo = repo_with_docs(tmp_path, pages=[("guide.qmd", page)])

    listings, fragments = collect(repo)

    assert listings == []
    assert fragments == 1


def test_a_gpu_block_is_skipped_without_an_accelerator(tmp_path):
    """`DeviceContext()` is instantiated against the host's accelerator, so a
    CPU-only runner cannot build the listing however correct it is -- which is
    a different claim from `.fragment`, and counted separately."""
    page = "```{.mojo .gpu}\nvar ctx = DeviceContext()\n```\n"
    repo = repo_with_docs(tmp_path, pages=[("guide.qmd", page)])
    check = SnippetCheck(repo, toolchain=None)

    listings, fragments = check.listings(gpu=False)

    assert listings == []
    assert fragments == 0  # not a fragment; the machine is the problem
    assert check._skipped == ["guide.qmd#0"]


def test_a_gpu_block_is_compiled_where_there_is_an_accelerator(tmp_path):
    page = "```{.mojo .gpu}\nvar ctx = DeviceContext()\n```\n"
    repo = repo_with_docs(tmp_path, pages=[("guide.qmd", page)])
    check = SnippetCheck(repo, toolchain=None)

    listings, _ = check.listings(gpu=True)

    assert [label for label, _ in listings] == ["guide.qmd#0"]
    assert check._skipped == []


def test_an_include_fence_is_skipped_rather_than_compiled_twice(tmp_path):
    """`{{< include >}}` names a file already collected from the snippets
    directory; compiling the fence as well would report the same failure under
    two labels."""
    page = "```mojo\n{{< include snippets/a.mojo >}}\n```\n"
    repo = repo_with_docs(
        tmp_path, pages=[("guide.qmd", page)], snippets=[("a.mojo", "def main():\n")]
    )

    listings, _ = collect(repo)

    assert [label for label, _ in listings] == ["docs/snippets/a.mojo"]


def test_pages_are_found_below_the_top_level(tmp_path):
    """The guides live in `docs/guide/`, so a non-recursive glob would check
    nothing that ships."""
    repo = repo_with_docs(tmp_path)
    (repo.docs_dir / "guide").mkdir()
    (repo.docs_dir / "guide" / "expressions.qmd").write_text(
        "```mojo\nvar a = 1\n```\n"
    )

    listings, _ = collect(repo)

    assert [label for label, _ in listings] == ["guide/expressions.qmd#0"]
