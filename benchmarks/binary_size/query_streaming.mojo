"""Binary-size demo: pull-based STREAMING fat-node relations, fused values.

Same query as the other variants (`SELECT a, name WHERE a > b`), built with
`marrow.expr.streaming` — the Phase-2 fat-node design where each relational op is
its own processor (`Scan`/`Filter`/`Project` with `pull()`), streamed
morsel-at-a-time with **no `Planner`**, over fused `AnyValue` values.

This proves the streaming model (which preserves `dyn`'s morsel protocol) does
*not* reintroduce the bloat: with no open per-kind `Planner` dispatch and
fused-only values, the pipeline should land near `query_erased_aot`/
`query_comptime` (~250 KB), not the `query_runtime` executor path (~7.7 MB).

    pixi run binary_size
"""

from marrow.builders import array
from marrow.dtypes import Int64Type, StringType, int64
from marrow.tabular import record_batch
from marrow.expr.relations import Table
from marrow.expr.values import Gt
from marrow.expr.erased import AnyValue
from marrow.expr.streaming import AnySource, Scan, Filter, Project


struct Orders:
    var a: Int64Type
    var b: Int64Type
    var name: StringType


def main() raises:
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var name = array(["p", "q", "r", "s", "t"])
    var batch = record_batch(
        [a.copy(), b.copy(), name.copy()], names=["a", "b", "name"]
    )

    var t = Table[Orders]()
    var scan = AnySource(Scan(batch^, morsel_size=1024))
    var filtered = AnySource(Filter(scan^, AnyValue(Gt(t.a, t.b))))
    var values = List[AnyValue]()
    values.append(AnyValue(t.a))
    values.append(AnyValue(t.name))
    var plan = AnySource(Project(filtered^, values^))
    var result = plan.collect()
    print(result)
