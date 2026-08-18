# G2 — Should marrow adopt `mojo-regex` to close ClickBench Q29?

Evaluation of [`msaelices/mojo-regex`](https://github.com/msaelices/mojo-regex)
as the regex engine behind a `replace_substring_regex` kernel, which is the one
thing standing between marrow's Python lazy frontend and 43/43 on ClickBench.

Measured on `alpha` (`557fc34`), osx-arm64, Mojo `1.1.0.dev2026081705`.
Everything below was compiled and run; nothing is taken from a README.
Throwaway spikes lived in `/tmp/regex-spike`, never in the repo.

**Verdict up front: do not adopt `mojo-regex`.** It fails on three independent
grounds, and the first one alone settles it — *it computes Q29's answer
incorrectly*, so it does not close the gap it would be adopted to close.
If regex is wanted, `dlopen` PCRE2, exactly as marrow already `dlopen`s
zstd/snappy/lz4/brotli. That route is correct, 10.6x faster, and costs
**zero** binary size.

---

## 0. The query

```sql
SELECT REGEXP_REPLACE(Referer, '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS k,
       AVG(length(Referer)) AS l, COUNT(*) AS c, MIN(Referer)
FROM hits WHERE Referer <> '' GROUP BY k HAVING COUNT(*) > 100000
ORDER BY l DESC LIMIT 25;
```

The pattern needs `^`, `$`, `?`, `(?:...)`, `[^/]`, `+`, `*`, `.`, alternation
via `s?`, one capture group, and `\1` substitution. Reference semantics
(PyArrow, which uses RE2 — verified in the `dev` env):

```
'http://www.example.com/path/to/page' -> 'example.com'
'https://sub.domain.org/a'            -> 'sub.domain.org'
'http://plain.net/'                   -> 'plain.net'
'not a url at all'                    -> 'not a url at all'   (no match: unchanged)
null                                  -> null
```

Note `example.com`, not `www.example.com`: the greedy optional `(?:www\.)?`
**must** consume `www.` so the capture starts after it. That single detail is
where `mojo-regex` fails, and it is the common case in the `hits` data.

---

## 1. `mojo-regex` — the evaluation

### 1.1 Licence — compatible, not a blocker

MIT (`LICENSE`, "Copyright (c) 2025 Manuel Saelices"). marrow is Apache-2.0
(`LICENSE.txt`) and already carries `NOTICE.txt` for redistributed third-party
code with exactly this shape ("Both licences are compatible with Apache-2.0").
Vendoring MIT into an Apache-2.0 project is fine provided the MIT notice travels
with the source. **Licence is not the problem here.**

### 1.2 Maturity — young, single-maintainer, but genuinely active

| | |
|---|---|
| Version / HEAD | v0.21.0, `c4352cb`, pushed **2026-08-17** (one day before this review) |
| Created | 2025-06-22 |
| Stars / forks / watchers | 14 / 3 / 0 |
| Open issues | 7 |
| Size | 15,433 LOC across 19 modules in `src/regex/` |
| Test files | 14, plus a benchmark suite and a `playground/` |

It is not abandoned — the author bumps the Mojo pin within days of nightlies and
had upgraded to stable Mojo 1.0.0 two weeks ago. But it is a one-person project
with effectively no dependent user base, and the open issues include
memory-safety and robustness items that matter for a library marrow would embed:

- **#164** — "AST `regex_ptr` back-pointer is untracked (memory-safety tech debt)"
- **#97** — "Flaky test CI: double-free in global regex cache destructor at process exit"
- **#31** — "Protect the library against Regular expression Denial of Service - ReDoS" (open since 2025-08)
- **#162** — "backend codegen crash + SIMD comparison reduces to scalar Bool"

A double-free at process exit in a global cache is precisely the class of defect
that would surface in marrow's test harness as an unrelated mass failure — the
same signature as the C-Data alignment flakiness already on record.

### 1.3 Mojo compatibility — it **does** compile. This is not the blocker.

This was the expected blocker and it turned out not to be one. `mojo-regex`
pins `mojo = "==1.0.0"`; marrow pins `==1.1.0.dev2026081705`. Built anyway:

```
$ pixi run -e dev mojo build -I /tmp/mojo-regex/src /tmp/regex-spike/spike.mojo -o ...
/tmp/mojo-regex/src/regex/onepass.mojo:164:6: warning: '@parameter' is deprecated; use '@__parameter'
```

One deprecation warning, no errors. Its own test suite under marrow's toolchain:

```
pass=13 fail=1 buildfail=0
```

All 14 test files **build** clean. 13 pass. `test_matcher` **hangs
indefinitely** — two independent runs spun at 100% CPU for 9m40s and 5m10s
before I killed them by PID, having emitted a single line of output
(`Range alone [c-n] on h: True`), i.e. it hangs almost immediately.

*Fair caveat:* the hang may be an artefact of running against a Mojo the library
does not claim to support, and might not reproduce on its pinned 1.0.0. I did
not download a second toolchain to check. It is not the reason for the verdict —
but "compiles" and "works" are different claims, and only the first is
established.

### 1.4 Feature coverage — **it gets Q29 wrong**

This is the finding that decides it. Direct string literals, diffed against
CPython's `re` (`/tmp/regex-spike/probe3.mojo` + `ref3.py`):

```
FAIL pat=^https?://(?:www\.)?([^/]+)/.*$  repl=\1  text=http://www.example.com/p
     python=example.com          mojo-regex=www.example.com
```

**Q29's exact pattern, on the common input shape, produces the wrong group.**
The `GROUP BY` key would be wrong for every `www.`-prefixed referer, which
throws off the buckets, the `HAVING COUNT(*) > 100000` filter and the final
ordering. Adopting this library would take marrow from "42/43, Q29 unsupported"
to "43/43, Q29 silently wrong" — strictly worse.

**10 of 25** `sub()` cases disagree with Python:

```
FAIL (?:www\.)?([^/]+)   [\1]  www.example.com  py=[example.com]  mr=[www.example.com]
FAIL (www\.)?([^/]+)     [\2]  www.example.com  py=[example.com]  mr=[www.example.com]
FAIL (?:ab)?(c+)         [\1]  abccc            py=[ccc]          mr=ab[ccc]
FAIL (ab)?(c+)           [\2]  abccc            py=[ccc]          mr=ab[ccc]
FAIL ^(?:x)?(y)$         [\1]  xy               py=[y]            mr=xy        (no match)
FAIL (?:https?)://(.+)   [\1]  http://q         py=[q]            mr=http://q  (no match)
FAIL (?:foo)?bar         B     foobar           py=B              mr=fooB
FAIL (?:foo)?(bar)       [\1]  foobar           py=[bar]          mr=foo[bar]
FAIL ^https?://(?:www\.)?([^/]+)/.*$  \1  http://www.example.com/p
FAIL ^.*$                M     anything         py=M              mr=MM        (trailing empty match)
```

There is a single root cause and it is narrow: **an optional group is never
entered.** The engine skips `(?:X)?` / `(X)?` and starts the match *after* it
rather than trying the greedy alternative first. Minimal reproduction, two
characters of pattern:

```mojo
sub("(?:foo)?bar", "B", "foobar")   # -> "fooB",  Python: "B"
```

Everything that does not involve an optional group is correct: alternation
`^(foo|bar)$`, character classes, `\d`/`\w`, multi-group reordering
`(\d+)-(\d+)` → `\2-\1`, greedy `(a*)(b+)`, `colou?r` (an optional *char* is
fine — only optional *groups* break), and `(?:www\.)([^/]+)` when the group is
**not** optional.

Why it survived: the upstream test suite uses `(?:` in 17 places and **never
once** with a trailing `?`. The path is entirely untested. This appears to be a
new bug — nothing matching it is in the open issues.

`sub()` itself is otherwise well-shaped: `\1..\9` in the replacement, an
optional `count`, a module-level cached form and a `CompiledRegex.sub` for
compile-once reuse. The API is not the problem; the matcher is.

### 1.5 Distribution — **`pixi add mojo-regex` does not work**

The README says:

> 2. **Add the Package** (at the top level of your project):
>    ```bash
>    pixi add mojo-regex
>    ```

It is not published. Resolving it in a scratch project using marrow's exact
channel list (`prefix.dev/mojo-community`, `conda.modular.com/max`,
`conda-forge`):

```
× failed to solve the environment
╰─▶ Cannot solve the request because of: No candidates were found for mojo-regex *.
```

The `mojo-community` channel carries 30 packages — `marrow` is one of them —
and no regex package. Nothing on anaconda.org either. Its CI (`ci.yml`, 515
bytes) runs tests and has no publish job.

**So adoption means vendoring 15,433 lines of someone else's engine into the
repo**, and owning it against every Mojo bump — a permanent maintenance
liability for a library marrow does not otherwise need, whose most important
path is already known-broken.

### 1.6 Performance — no ReDoS blowup, but 10.6x slower than the alternative

Compiled `-O3`, `CompiledRegex.sub` on ~50-byte URLs, single-threaded:

```
rows: 200,000   0.363 s   551,315 rows/s
```

Engine selected for Q29's pattern: **NFA**, complexity MEDIUM. The library has
DFA / one-pass / PikeVM / literal-prefilter paths and picks per pattern.
`(a+)+b` on 30 `a`s returned in 0.14 ms — **no catastrophic backtracking** on
the classic case, which is a genuine point in its favour, though issue #31 says
ReDoS hardening is explicitly not done.

At 551K rows/s, ClickBench's 1M-row `hits_0.parquet` costs ~1.8 s of
single-threaded regex — parallelisable over `ctx.stripe`, so workable, but an
order of magnitude off what the alternative gives for free.

### 1.7 Binary size — it would blow the gate

`__text`, `-O3`:

| binary | `__text` |
|---|---|
| hello-world | 780 |
| hello-world + `regex.sub` | **493,792** |

So linking the engine costs **~493 KB of `__text`**. Against the gates
(`benchmarks/binary_size/baseline.json`, threshold **0.5%**):

| gate | baseline | if regex were reachable |
|---|---|---|
| `query_streaming_agg_fused` | 1,417,476 | **+34.8%** |
| `query_streaming` | 1,484,652 | +33.3% |
| `query_dynvalue` | 4,871,156 | **+10.1%** |

The closed-erasure/DCE property (§1.2 of `README.md` in this directory) means a
runtime-lane-only kernel contributes **0 symbols** to the fused/AOT targets, as
`marrow::expr::dynamic` already demonstrates — so the four fused gates could in
principle be held at zero. But `query_dynvalue` is a runtime-lane gate by
definition and *would* take the full hit: **+10.1% against a 0.5% threshold**,
i.e. a 20x overrun requiring a deliberate baseline raise. That is a real
architectural cost, and it buys an engine that computes the wrong answer.

---

## 2. The options

### (a) Adopt `mojo-regex` — dependency or vendored

Three independent disqualifiers, any one of which is sufficient:

1. **It gets Q29 wrong.** Optional groups are never entered. It does not close
   the gap; it converts an honest gap into a silent wrong answer.
2. **It cannot be a dependency.** Not on any channel; `pixi add` fails. Adoption
   = vendoring 15.4K LOC and owning it across Mojo bumps.
3. **+493 KB `__text`**, a 20x overrun of the `query_dynvalue` gate.

Plus: single maintainer, 0 watchers, open memory-safety and double-free issues,
and its own `test_matcher` hangs under marrow's toolchain.

*In fairness:* MIT-compatible, actively maintained, compiles against marrow's
Mojo, has a well-shaped `sub()` API with `\1..\9`, no catastrophic backtracking,
and 13/14 of its own tests pass. The optional-group bug is narrow and probably a
small fix upstream. **If it were fixed, published to `mojo-community`, and the
size question answered, this would be worth revisiting** — the pure-Mojo,
no-native-dependency story is genuinely attractive. It is not there today.

### (b) Hand-write `extract_host` — cheap, but answers a different question

Q29's pattern is a hostname extraction. A purpose-built kernel is maybe 60 lines
with no dependency, no binary-size concern and no engine to maintain.

But it is **not** `REGEXP_REPLACE`, and the harness would have to mark Q29
`DEVIATED` (`clickbench_alpha.py` already supports this; two queries carry it).
The existing coverage doc reached this same conclusion and rejected it:

> would be served by a `url_host()` scalar function at a fraction of the cost —
> but that would be answering a different question than ClickBench asked.

It also generalises to nothing: the next regex query needs another hand-written
kernel. 42/43 + a deviation is not 43/43.

### (c) `dlopen` PCRE2 — the established in-repo pattern

`marrow/utils/compression.mojo` opens `libzstd`/`libsnappy`/`liblz4`/`libz`/
`libbrotli` at runtime through `_try_find_dylib`, with the libraries declared as
ordinary conda dependencies in `pixi.toml`. Its own docstring states the
principle: *"the standard C libraries are `dlopen`-ed at runtime and their block
APIs called directly — the same approach arrow-rs and duckdb take, just without
a link-time dependency."* Regex is the identical shape.

Verified end-to-end (`/tmp/regex-spike/pcre2_test.c`, hand-declared prototypes,
no headers needed):

```
dlopen OK: libpcre2-8.dylib
syms: compile=0x272684c80 md=0x2726a622c substitute=0x2726a9104
rc= 1  in=http://www.example.com/path/to/page    out=example.com
rc= 1  in=https://sub.domain.org/a               out=sub.domain.org
rc= 1  in=http://plain.net/                      out=plain.net
rc= 0  in=not a url at all                       out=not a url at all
rc= 1  in=https://www.a.co/                      out=a.co

rows=1000000 secs=0.1709 rows/s=5850314
(a+)+b on 40 a's rc=0 secs=0.000004
```

- **Correct on every case**, matching PyArrow and CPython exactly — including
  `example.com`, the one `mojo-regex` gets wrong.
- **5,850,314 rows/s** single-threaded — **10.6x `mojo-regex`**. The full 1M-row
  Q29 regex pass costs 0.17 s on one thread.
- **Zero `__text` cost.** The engine lives in the shared library; only a thin
  FFI shim is linked, bounded by the codec precedent at a few KB. Every gate
  including `query_dynvalue` stays green.
- **Distributable.** `pcre2` 10.47 (830 KiB) resolves from conda-forge for both
  `osx-arm64` and `linux-64` under marrow's exact channel list — verified by
  actually installing it. One line in `pixi.toml` next to `zstd`/`snappy`.
- Available at `/usr/lib/libpcre2-8.dylib` on macOS as a fallback, though the
  conda package is what should be depended on.

Trade-offs, stated honestly:

- **A new native runtime dependency.** CLAUDE.md says avoid unnecessary
  dependencies — but marrow already has four of exactly this kind, and this is
  the pattern the project chose for "C library we should not reimplement".
- **PCRE2 is a backtracking engine; RE2 (what Arrow C++ uses) is not.** So
  marrow must set `match_limit`/`depth_limit` to bound ReDoS on user-supplied
  patterns. PCRE2 exposes both. RE2 would be the closer semantic match to Arrow
  C++, but it is C++ with no stable C ABI, so it cannot be `dlopen`ed cleanly —
  PCRE2 is the right pick for this pattern.
- **Replacement syntax differs.** PCRE2 uses `$1`; SQL and PyArrow use `\1`. A
  ~10-line `\N` → `$N` translation at the kernel boundary, well-defined.
- **Minor semantic divergence from RE2** in corners (PCRE2 permits
  backreferences *in patterns*, RE2 rejects them). Worth one documented line.

### (d) Ship 42/43 and document Q29 — the status quo

Zero cost, honest, and what `docs/alpha-clickbench-coverage.md:187` already
recommends ("Writing one (or binding one) for a single query is not a good
trade"). If Q29 is the *only* motivation, this remains defensible.

---

## 3. Recommendation

**Reject (a). If regex is wanted, do (c). Otherwise stay on (d). Do not do (b).**

The reasoning turns on one reframing. The prior recommendation — "not worth
binding an engine for a single query" — is correct *if the payoff is Q29*. But
the payoff is not Q29. marrow has **no regex kernel at all**, and PyArrow ships
seven regex compute functions that marrow's users will expect:

```
match_substring_regex   replace_substring_regex   extract_regex
count_substring_regex   find_substring_regex      split_pattern_regex
extract_regex_span
```

Q29 is the *forcing function*, not the *benefit*. Framed as "add the regex
string kernels PyArrow has, and get 43/43 as a side effect", option (c) is
clearly worth it — especially since the dlopen route makes it cheap in the two
currencies marrow actually guards: **binary size (zero) and compile time
(zero)**. Reimplementing an engine in Mojo pays in both; borrowing libc-adjacent
C pays in neither.

If the decision is instead "we only care about the ClickBench number", then (d)
is right and this document is the record of why (a) is not an option — the
adoption everyone would reach for first is the one that returns wrong results.

**Do not do (b).** It buys a DEVIATED row, generalises to nothing, and would
have to be deleted the moment real regex lands.

---

## 4. Integration sketch for (c) — *not built; reporting first*

**`pixi.toml`** — one dependency beside the codecs:

```toml
# PCRE2, opened at runtime via dlopen by marrow/utils/regex.mojo
# (no link-time dependency), same as the compression codecs above.
pcre2 = ">=10.40"
```

**`marrow/utils/regex.mojo`** *(new, ~300 LOC)* — mirrors
`marrow/utils/compression.mojo` exactly: `comptime _PCRE2_PATHS: List[Path]`
(`libpcre2-8.dylib` / `.0.dylib` / `.so.0` / `.so`), a lazily-opened
`Optional[OwnedDLHandle]` via `_try_find_dylib["pcre2-8"]`, and thin wrappers
over `pcre2_compile_8`, `pcre2_match_data_create_from_pattern_8`,
`pcre2_substitute_8`, `pcre2_match_8`, `pcre2_code_free_8`. Holds a compiled
pattern so a kernel compiles **once per call, not per row** (the 5.85M rows/s
number assumes this). Sets `match_limit` / `depth_limit` for ReDoS. This file
joins the `unsafe_ptr()` allowlist in CLAUDE.md, as the codec layer already has.

**`marrow/kernels/regex.mojo`** *(new, ~300 LOC)* — typed-first per the kernel
convention: a `StringArray`/`LargeStringArray` overload holding all the logic,
then a `DynArray` overload delegating. PyArrow names and semantics
(nulls propagate, `max_replacements` default unlimited):

- `replace_substring_regex(arr, pattern, replacement, max_replacements=-1)`
- `match_substring_regex(arr, pattern)` → `BoolArray`
- `extract_regex(arr, pattern)` → `StructArray` of groups

Build output with `StringBuilder`; reuse one `pcre2_match_data` and one output
buffer across rows.

**Wiring** — a `DynValue` leaf in `marrow/expr/dynamic.mojo`, the Python binding,
and a `LazyTable` expression method. **Runtime lane only** — nothing in
`values.mojo`/`aggregates.mojo`, so the four fused gates keep contributing zero
symbols and the DCE property is preserved.

**Tests** — `marrow/kernels/tests/test_regex.mojo` against Arrow C++/pyarrow
vectors, plus Q29 end-to-end in `python/marrow/tests/test_clickbench.py`
(removing the `UNSUPPORTED` row from `docs/alpha-clickbench-coverage.md:87`).

**Verification** — `pixi run binary_size` must stay green with no baseline
raise; that is the load-bearing check and the main risk to confirm early.

**Estimate: 2–3 days.** Roughly half a day for the FFI shim (the codec module is
a direct template), a day for the kernels and their tests, half a day for the
expression/binding wiring, and half a day for the ClickBench end-to-end plus
size verification. The main unknown is `OwnedDLHandle.call` ergonomics for
PCRE2's 11-argument `pcre2_substitute_8`, which is more parameters than any
codec call currently passes.

---

## 5. Reproducing

Everything here is a throwaway spike under `/tmp/regex-spike` (nothing was
cloned into or written to the repo beyond this document):

| file | what it shows |
|---|---|
| `spike.mojo` | mojo-regex builds against marrow's Mojo; Q29 output |
| `probe3.mojo` + `ref3.py` | the 25-case `sub()` table diffed against CPython `re` |
| `probe2.mojo` | optional-group isolation, throughput, ReDoS probe |
| `runtests2.sh` | mojo-regex's own 14 test files under marrow's toolchain |
| `pcre2_test.c` | PCRE2 via `dlopen`: correctness, 5.85M rows/s, ReDoS |
| `scratch/pixi.toml` | `pixi add mojo-regex` fails; `pcre2` resolves |

Upstream ref: `msaelices/mojo-regex` @ `c4352cb` (v0.21.0, 2026-08-17).
