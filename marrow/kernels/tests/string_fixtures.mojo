"""The string columns the string benchmarks measure over.

Shared rather than repeated because those benchmarks are spread across four
files, and a file is one `-O3` compilation unit -- see `bench_string.mojo` for
why they are spread.

Not named `bench_*`, so neither the harness nor the benchmark workflow's
`find` picks it up as a selection of its own.
"""

from ...arrays import StringArray
from ...builders import StringBuilder


def urls(n: Int) raises -> StringArray:
    """ClickBench-ish URLs, roughly a quarter of which contain 'google'."""
    var b = StringBuilder(capacity=n)
    for i in range(n):
        var r = i % 4
        if r == 0:
            b.append("http://www.google.com/search?q=" + String(i))
        elif r == 1:
            b.append("http://example.org/page/" + String(i))
        elif r == 2:
            b.append("https://news.site.ru/article/" + String(i))
        else:
            b.append("http://shop.example.com/item/" + String(i))
    return b.finish()


def broadcast(pattern: String, n: Int) raises -> StringArray:
    var b = StringBuilder(capacity=n)
    for _ in range(n):
        b.append(pattern)
    return b.finish()
