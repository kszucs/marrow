# repros/ — bisection drivers for the F1 SIGSEGV

Scratch tooling kept with the investigation it belongs to; see
`docs/alpha-findings/f1-distinct-segfault.md` for what each one established.

| file | what it does |
|---|---|
| `bisect.py` / `run_bisect.sh` | the first cut: q11 with one ingredient removed at a time |
| `bisect2.py` / `run2.sh` | projection x predicate x morsel-size x source matrix over `hits_0.parquet` |
| `synth.py` / `run3.sh` | synthetic Parquet with a page index — tested (and cleared) page-level skipping |
| `synth2.py` / `run4.sh` | synthetic Parquet with clustered selectivity — tested (and cleared) zero-row morsels |
| `showips.py` / `lastcrash.sh` | decode the macOS `.ips` crash reports, which is how the tcmalloc stack was recovered |
| `truth.py` / `verify_q11.py` | PyArrow reference answers for the predicates and for Q11/Q12 |
| `run_with.sh` / `run_cb.sh` / `run_assert.sh` | run a case against a `repros/build/*.so` variant |
| `pt.sh` | `pytest` into a log file (never pipe pytest through `tail`) |
| `asan_probe.mojo` / `build_asan.sh` | probe for whether ASAN intercepts Mojo's allocator — it hangs, see the findings doc |

`repros/build/` and `repros/*.parquet` are build artefacts and are not committed.
