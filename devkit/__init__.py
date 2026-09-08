"""marrow's developer tooling.

One package for everything that builds, tests, benchmarks, profiles and measures
the tree, plus the click CLI that drives it (`python -m devkit --help`).

Layout, by dependency rather than by class -- a module boundary exists here only
where one side must stay free of something the other needs:

* `mojo`        -- the checkout, the compiler and the subprocess underneath it.
* `runner`      -- options, lanes, selections, drivers, suites.  No pytest.
* `benches`     -- Mojo timings into pytest-benchmark; its one importer.
* `golden`      -- the cross-lane query corpus; reaches duckdb when regenerating.
* `conformance` -- the Arrow archery suite; reaches archery.
* `footprint`   -- the AOT size gate: nm, size -m, strip.
* `profiling`   -- Instruments and macOS `sample`.
* `cli`         -- the only module that imports click.

`Repo` is the single source of layout truth: no other module spells a directory
or a Mojo namespace, so a rename cannot leave stale strings behind.  `golden` is
the one exception and owns the corpus's own layout -- `cases/`, `fixtures/`,
`.exp/`, `generated/` -- because that is the corpus's format rather than the
tree's.

Only `benches` and `golden` reach into pytest, and both do it inside a function.
Everything else is plain Python and is tested as such.

`conftest.py` is the whole pytest boundary: it owns the hooks and the collector
classes and delegates every decision here.
"""
