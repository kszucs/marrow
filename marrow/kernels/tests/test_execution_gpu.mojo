"""The half of `ExecContext`'s contract that needs a real device.

Everything here is about one defect: a context that carries a GPU device being
rebuilt through a factory that sets `device=None`. `HashJoin` hit it at five
internal sites and fixed it by holding the context whole; the eager group-by
driver and `Aggregation` hit it through `num_threads: Int` API boundaries that
destructured the context and rebuilt it with `ExecContext.parallel(n)`.

A CPU-only test cannot see any of this — `is_gpu()` is False before and after,
so every assertion passes on a context that has already lost its device.
"""

from std.testing import assert_true, assert_false, assert_equal
from max.gpu.host import DeviceContext

from ...execution import ExecContext


def test_with_threads_preserves_the_device() raises:
    """The replacement for `ExecContext.parallel(n)` at every internal
    re-derivation site. `parallel(n)` is a *factory* — it builds a fresh CPU
    context — so using it to change a worker count silently drops the device."""
    var ctx = ExecContext.gpu(DeviceContext())
    assert_true(ctx.is_gpu())

    var rethreaded = ctx.with_threads(4)
    assert_true(rethreaded.is_gpu())
    assert_equal(rethreaded.resolved_num_threads(), 4)

    # And the serial direction, which a single-partition grouped aggregation
    # takes.
    var serialized = ctx.with_threads(1)
    assert_true(serialized.is_gpu())
    assert_equal(serialized.resolved_num_threads(), 1)


def test_parallel_factory_drops_the_device() raises:
    """Pins the trap itself, so nobody "simplifies" `with_threads` back into it.

    This is not a bug in `parallel` — a factory is entitled to build a fresh
    context. It is a bug at every call site that used it to *modify* one.
    """
    assert_false(ExecContext.parallel(4).is_gpu())
    assert_false(ExecContext.serial().is_gpu())


def test_worth_parallel_is_false_on_the_gpu() raises:
    """The device runs its own parallelism, so a CPU worker split is never the
    right answer on it — the same short-circuit `wants_parallel` already has.

    Its absence is what let `count_distinct`, `HashJoin.build` and the eager
    group-by driver's strategy choice take a CPU-parallel path on a GPU context:
    all three asked `resolved_num_threads()`, which knows nothing about the
    device.
    """
    var ctx = ExecContext(num_threads=8, device=DeviceContext())
    assert_true(ctx.is_gpu())
    assert_false(ctx.wants_parallel(1_000_000))
    assert_false(ctx.worth_parallel(1_000_000, 60_000))
