"""Golden cases — string kernels, the runtime lane.

Case folding, trimming, length, prefix tests and LIKE/ILIKE. No regex: there
is no regex engine in the tree (backlog M2.6), so nothing to compare against.

`length` is worth its own case because it has to commit to *bytes or
codepoints*: `héllo` is five characters and six bytes.
"""


def test_golden_string_upper(golden):
    """SELECT upper(s) AS u FROM words"""
    t = golden.table("words")
    golden.check(t.project(u=t["s"].upper()))


def test_golden_string_lower(golden):
    """SELECT lower(s) AS l FROM words"""
    t = golden.table("words")
    golden.check(t.project(l=t["s"].lower()))


def test_golden_string_strip(golden):
    """SELECT trim(s) AS p FROM words"""
    t = golden.table("words")
    golden.check(t.project(p=t["s"].strip()))


def test_golden_string_length_counts_bytes(golden):
    """SELECT CAST(strlen(s) AS INTEGER) AS n FROM words

    Bytes, not codepoints: `héllo` is 6. The twin says `octet_length`
    deliberately — DuckDB's `length` counts characters, which is the other
    answer and would make this case assert the wrong thing.
    """
    t = golden.table("words")
    golden.check(t.project(n=t["s"].length()))


def test_golden_string_starts_with(golden):
    """SELECT starts_with(s, 'H') AS b FROM words"""
    t = golden.table("words")
    golden.check(t.project(b=t["s"].startswith("H")))


def test_golden_string_like_is_case_sensitive(golden):
    """SELECT s LIKE 'h%' AS b FROM words"""
    t = golden.table("words")
    golden.check(t.project(b=t["s"].like("h%")))


def test_golden_string_ilike_is_not(golden):
    """SELECT s ILIKE 'h%' AS b FROM words"""
    t = golden.table("words")
    golden.check(t.project(b=t["s"].ilike("h%")))


def test_golden_string_filter_on_like(golden):
    """SELECT s FROM words WHERE s LIKE '%o%'"""
    t = golden.table("words")
    golden.check(t.filter(t["s"].like("%o%")))
