"""The physical layer: operators that own execution state and pull morsels.

A `Relation` describes a query and is immutable, shareable and rewritable. A
`Processor` is what that description becomes when it runs, and it owns
everything mutable — a scan's offset, a grouper's table, a sort's buffer.

The split is deliberate and it is the cleaner half of `expr/`'s design:
`DynProcessor` accreted **three** slots to `DynRelation`'s eight, because a
processor was only ever asked two things. It is kept here for that reason and
for one more — `to_processor(ctx)` is the seam where one logical operator could
become different physical ones (hash join or merge join, CPU or GPU). Fusing
the layers would weld that shut.

`pull` raises `Exhausted` rather than returning an `Optional[RecordBatch]`:
end-of-stream is not a value a caller should be able to ignore, and an operator
that forgets to propagate it hangs rather than silently truncating.
"""

from std.memory import ArcPointer

from ..arrays import DynArray
from ..builders import DynBuilder
from ..kernels.concat import concat
from ..execution import ExecContext
from ..kernels.filter import filter
from ..schema import Schema
from ..tabular import RecordBatch
from .core import DynValue, into_array


struct Exhausted(TrivialRegisterPassable, Writable):
    """Raised by `pull()` when a processor has no more morsels."""

    def __init__(out self):
        pass

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Exhausted")


trait Processor(Deinitable, Movable):
    """Something that yields morsels until it is empty.

    Two methods, and that is the whole physical contract. Every operator —
    scan, filter, sort, join — is one of these, and the erased box carries
    exactly these two plus a destructor.
    """

    def schema(self) -> Schema:
        ...

    def pull(mut self) raises -> RecordBatch:
        """The next morsel, or raise `Exhausted`."""
        ...


struct DynProcessor(Movable):
    """A `Processor` of any operator, erased.

    Move-only: a processor owns mutable state, so copying one would fork an
    execution mid-stream. `DynRelation` copies freely for the opposite reason —
    it owns nothing that runs.
    """

    var _data: ArcPointer[NoneType]
    var _virt_schema: def(ArcPointer[NoneType]) thin -> Schema
    var _virt_pull: def(ArcPointer[NoneType]) thin raises -> RecordBatch

    @staticmethod
    def _schema_tramp[P: Processor](ptr: ArcPointer[NoneType]) -> Schema:
        return rebind[ArcPointer[P]](ptr)[].schema()

    @staticmethod
    def _pull_tramp[
        P: Processor
    ](ptr: ArcPointer[NoneType]) raises -> RecordBatch:
        return rebind[ArcPointer[P]](ptr)[].pull()

    @implicit
    def __init__[P: Processor](out self, var value: P):
        var ptr = ArcPointer[P](value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_schema = Self._schema_tramp[P]
        self._virt_pull = Self._pull_tramp[P]

    def schema(self) -> Schema:
        return self._virt_schema(self._data)

    def pull(mut self) raises -> RecordBatch:
        return self._virt_pull(self._data)

    def collect(mut self) raises -> RecordBatch:
        """Drain every morsel into one batch.

        The only thing that catches `Exhausted`, so an operator never has to
        decide what end-of-stream means.
        """
        var batches = List[RecordBatch]()
        while True:
            try:
                batches.append(self.pull())
            except e:
                if String(e) == "Exhausted":
                    break
                raise e
        if len(batches) == 0:
            # An empty result must still be a *well-formed* batch: one
            # zero-length column per field. A schema naming fields beside an
            # empty column list leaves `num_columns()` at 0, so anything
            # walking columns by schema index runs off the end.
            var s = self.schema()
            var cols = List[DynArray](capacity=len(s.fields))
            for ref f in s.fields:
                var b = DynBuilder(f.dtype)
                cols.append(b.finish())
            return RecordBatch(schema=s^, columns=cols^)
        if len(batches) == 1:
            return batches.pop()

        # Concatenate per column: `concat` joins arrays, and a batch is its
        # columns plus a schema every morsel already agrees on.
        var s = batches[0].schema.copy()
        var cols = List[DynArray](capacity=len(s.fields))
        for i in range(len(s.fields)):
            var parts = List[DynArray]()
            for ref b in batches:
                parts.append(b.columns[i].copy())
            # Serial: this runs once at the end over already-materialised
            # morsels, so there is nothing for workers to overlap with.
            cols.append(concat(parts, ExecContext.serial()))
        return RecordBatch(schema=s^, columns=cols^)


struct BatchSource(Processor):
    """Yields one in-memory batch, once."""

    var _batch: RecordBatch
    var _done: Bool

    def __init__(out self, var batch: RecordBatch):
        self._batch = batch^
        self._done = False

    def schema(self) -> Schema:
        return self._batch.schema.copy()

    def pull(mut self) raises -> RecordBatch:
        if self._done:
            raise Exhausted()
        self._done = True
        return self._batch.copy()


struct FilterProcessor(Processor):
    """Keeps rows where the predicate is true."""

    var _input: DynProcessor
    var _predicate: DynValue
    var _ctx: ExecContext

    def __init__(
        out self,
        var input: DynProcessor,
        var predicate: DynValue,
        var ctx: ExecContext,
    ):
        self._input = input^
        self._predicate = predicate^
        self._ctx = ctx^

    def schema(self) -> Schema:
        return self._input.schema()

    def pull(mut self) raises -> RecordBatch:
        """Evaluate the predicate, then compact every column.

        A morsel that filters to nothing is skipped rather than yielded: an
        empty batch is a legitimate *result* but a useless *morsel*, and
        forwarding it makes every operator above handle a case that carries no
        rows.
        """
        while True:
            var batch = self._input.pull()
            var mask = into_array(
                self._predicate.evaluate(batch), batch.num_rows()
            )
            var cols = List[DynArray]()
            for i in range(batch.num_columns()):
                cols.append(
                    filter(batch.columns[i].copy(), mask.copy(), self._ctx)
                )
            var out = RecordBatch(schema=batch.schema.copy(), columns=cols^)
            if out.num_rows() > 0:
                return out^


struct ProjectProcessor(Processor):
    """Evaluates each projected value against every morsel."""

    var _input: DynProcessor
    var _values: List[DynValue]
    var _schema: Schema

    def __init__(
        out self,
        var input: DynProcessor,
        var values: List[DynValue],
        var schema: Schema,
    ):
        self._input = input^
        self._values = values^
        self._schema = schema^

    def schema(self) -> Schema:
        return self._schema.copy()

    def pull(mut self) raises -> RecordBatch:
        var batch = self._input.pull()
        var cols = List[DynArray](capacity=len(self._values))
        for ref v in self._values:
            # `into_array` is where a scalar-shaped value stops being lazy: a
            # projection of a constant materialises here and nowhere earlier.
            cols.append(into_array(v.evaluate(batch), batch.num_rows()))
        return RecordBatch(schema=self._schema.copy(), columns=cols^)
