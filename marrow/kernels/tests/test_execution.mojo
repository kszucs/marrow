"""Tests for `ExecContext` — the CPU/GPU dispatch policy kernels share.

`stripe` is the contract every striped kernel now depends on, so its invariants
are pinned here rather than inferred from whichever kernel happens to use it:
the stripes must tile `[0, length)` exactly once, `wid` must index the scratch
the caller allocated from `stripe_workers`, and `align` must keep every boundary
on a multiple of itself. A kernel that violates any of these corrupts data or
reads past its scratch, and neither shows up as a compile error.
"""

from std.testing import assert_equal, assert_true, assert_false

from ...execution import ExecContext


# ---------------------------------------------------------------------------
# Strategy: how `num_threads` resolves and when it parallelises
# ---------------------------------------------------------------------------


def test_context_serial_is_forced() raises:
    """`serial()` never parallelises, however big the problem."""
    var ctx = ExecContext.serial()
    assert_equal(ctx.resolved_num_threads(), 1)
    assert_false(ctx.wants_parallel(1))
    assert_false(ctx.wants_parallel(1_000_000_000))


def test_context_parallel_n_is_forced() raises:
    """`parallel(n)` for n >= 2 always parallelises, bypassing the threshold."""
    var ctx = ExecContext.parallel(4)
    assert_equal(ctx.resolved_num_threads(), 4)
    assert_true(ctx.wants_parallel(1))
    assert_true(ctx.wants_parallel(1_000_000))


def test_context_auto_consults_the_threshold() raises:
    """`auto()` parallelises only above `min_parallel_size`; below it the
    stripe-dispatch overhead exceeds the work."""
    var ctx = ExecContext.auto()
    assert_true(ctx.resolved_num_threads() >= 1)
    assert_false(ctx.wants_parallel(100, min_parallel_size=32768))
    assert_true(ctx.wants_parallel(100_000, min_parallel_size=32768))


def test_context_default_is_serial() raises:
    """The default constructor is the serial context — relied on by every
    kernel whose `ctx` argument defaults, and by `HashJoin()`."""
    var ctx = ExecContext()
    assert_equal(ctx.resolved_num_threads(), 1)
    assert_false(ctx.is_gpu())
    assert_false(ctx.wants_parallel(1_000_000))


# ---------------------------------------------------------------------------
# stripe_workers — callers size per-worker scratch with this
# ---------------------------------------------------------------------------


def test_stripe_workers_is_one_when_serial() raises:
    """A serial run is one stripe, so a caller allocates exactly one slot."""
    assert_equal(ExecContext.serial().stripe_workers(1_000_000), 1)


def test_stripe_workers_matches_forced_thread_count() raises:
    assert_equal(ExecContext.parallel(4).stripe_workers(1_000), 4)


def test_stripe_workers_is_one_below_the_auto_threshold() raises:
    """Under `auto`, a small problem runs serially — so the scratch is one slot,
    not `num_physical_cores()` of them."""
    assert_equal(ExecContext.auto().stripe_workers(10), 1)


# ---------------------------------------------------------------------------
# stripe — the tiling contract
#
# Each stripe writes only its own slot, so these run race-free under real
# parallelism and double as a check that `wid` is a valid scratch index.
# ---------------------------------------------------------------------------


def _stripes(
    ctx: ExecContext, length: Int, align: Int = 1
) raises -> Tuple[List[Int], List[Int]]:
    """Run `stripe` and return the `(start, end)` each stripe received.

    Unvisited slots stay `-1`, which is how the empty-stripe cases are told
    apart from a stripe that ran with an empty range.
    """
    var workers = ctx.stripe_workers(length)
    var starts = List[Int](length=workers, fill=-1)
    var ends = List[Int](length=workers, fill=-1)

    @always_inline
    @parameter
    def record(wid: Int, start: Int, end: Int):
        starts[wid] = start
        ends[wid] = end

    ctx.stripe[record](length, align=align)
    return (starts^, ends^)


def _assert_tiles(starts: List[Int], ends: List[Int], length: Int) raises:
    """Assert the visited stripes tile `[0, length)` exactly once."""
    var covered = List[Int](length=length, fill=0)
    for w in range(len(starts)):
        if starts[w] < 0:
            continue  # stripe never ran (empty tail)
        assert_true(starts[w] <= ends[w])
        for i in range(starts[w], ends[w]):
            covered[i] += 1
    for i in range(length):
        assert_equal(covered[i], 1)


def test_stripe_serial_runs_one_stripe_over_everything() raises:
    """The serial arm is a single `body(0, 0, length)` — one stripe, wid 0."""
    var pair = _stripes(ExecContext.serial(), 1000)
    ref starts = pair[0]
    ref ends = pair[1]
    assert_equal(len(starts), 1)
    assert_equal(starts[0], 0)
    assert_equal(ends[0], 1000)


def test_stripe_parallel_tiles_the_range_exactly_once() raises:
    """No element is skipped and none is processed twice — the property that
    makes a striped write to a shared output safe."""
    var pair = _stripes(ExecContext.parallel(4), 1000)
    ref starts = pair[0]
    ref ends = pair[1]
    _assert_tiles(starts, ends, 1000)


def test_stripe_tiles_when_length_is_indivisible() raises:
    """A length that does not divide by the worker count still tiles exactly."""
    var pair = _stripes(ExecContext.parallel(4), 1001)
    ref starts = pair[0]
    ref ends = pair[1]
    _assert_tiles(starts, ends, 1001)


def test_stripe_tiles_with_alignment() raises:
    """`align` must not break the tiling — it only moves the boundaries."""
    var pair = _stripes(ExecContext.parallel(4), 1000, align=8)
    ref starts = pair[0]
    ref ends = pair[1]
    _assert_tiles(starts, ends, 1000)


def test_stripe_alignment_keeps_boundaries_on_multiples() raises:
    """Every stripe starts on a multiple of `align`, so a vectorized body runs
    its scalar tail once at the very end rather than once per stripe."""
    var pair = _stripes(ExecContext.parallel(4), 1000, align=8)
    ref starts = pair[0]
    var checked = 0
    for w in range(len(starts)):
        if starts[w] >= 0:
            assert_equal(starts[w] % 8, 0)
            checked += 1
    # Without `align` the chunk would be 250 and the starts 0/250/500/750 — two
    # of which are not multiples of 8 — so this genuinely discriminates.
    assert_equal(checked, 4)


def test_stripe_skips_empty_tail_stripes() raises:
    """With more workers than elements the trailing stripes have nothing to do
    and must not run — a body that assumes `start < end` would read garbage."""
    var pair = _stripes(ExecContext.parallel(8), 3)
    ref starts = pair[0]
    ref ends = pair[1]
    _assert_tiles(starts, ends, 3)
    var ran = 0
    for w in range(len(starts)):
        if starts[w] >= 0:
            ran += 1
    assert_true(ran <= 3)


def test_stripe_wid_indexes_within_stripe_workers() raises:
    """Every `wid` is a valid index into scratch sized by `stripe_workers` —
    the invariant that keeps a per-worker histogram in bounds."""
    var ctx = ExecContext.parallel(4)
    var workers = ctx.stripe_workers(1000)
    var seen = List[Int](length=workers, fill=0)

    @always_inline
    @parameter
    def mark(wid: Int, start: Int, end: Int):
        seen[wid] += 1

    ctx.stripe[mark](1000)
    # Exactly once each, not "at most once" — the weaker form would also pass if
    # no stripe ran at all.
    assert_equal(workers, 4)
    for w in range(workers):
        assert_equal(seen[w], 1)


def test_stripe_zero_length_visits_nothing() raises:
    """An empty input runs no stripe body over any element."""
    var total = List[Int](length=1, fill=0)

    @always_inline
    @parameter
    def count(wid: Int, start: Int, end: Int):
        total[0] += end - start

    ExecContext.serial().stripe[count](0)
    assert_equal(total[0], 0)
