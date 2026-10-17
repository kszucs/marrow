# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Decorrelation: every subquery a plan evaluates, planned as the joins,
aggregates and bindings that compute it.

A subquery is a value (`Subquery`, reported through `References`), so the
expression reading it is never rewritten and a fused predicate stays fused.
What changes is the plan *around* the node that reads it:

| subquery | where | becomes |
|---|---|---|
| `EXISTS`, `IN` | a filter conjunct | a semi join; the conjunct is dropped |
| `NOT EXISTS` | a filter conjunct | an anti join |
| `NOT IN` | a filter conjunct | an anti join, when neither key can be null |
| uncorrelated scalar or existence test | anywhere | a `ScalarBinding` at the root |
| correlated scalar over an aggregate | a filter or projection | a left join to the aggregate, grouped by the correlation keys |

A correlated conjunct equating an inner column with an outer one is a join key:
the subquery's filters lose it, and an aggregate between it and the subquery's
root gains the inner column as a group key. Anything else raises
`NotImplementedError` naming the shape, rather than running a nested loop.

**Idempotent.** A subquery whose slot is already a column of the node's input,
or already bound by a `ScalarBinding`, is left alone, so running this twice
plans nothing twice.
"""

from ..errors import InternalError, NotImplementedError
from ..execution import ExecContext
from ..kernels.join import BUILD_RIGHT, JOIN_ANTI, JOIN_LEFT, JOIN_SEMI
from ..scalars import BoolScalar, DynScalar
from ..tabular import RecordBatch
from .logical import (
    Aggregate,
    DynRelation,
    DynValue,
    EmptyRelation,
    FieldRef,
    Filter,
    JoinChain,
    Limit,
    Project,
    ResidualJoin,
    ScalarBinding,
    Sort,
    Subquery,
    SubqueryKind,
    Window,
)
from .runtime.values import (
    and_,
    column,
    exists,
    if_else,
    is_null,
    is_valid,
    literal,
    not_,
    or_,
)


@fieldwise_init
struct Correlation(Copyable, Movable):
    """A subquery's relation with its correlated conjuncts lifted out.

    `relation` no longer reads the enclosing query. It answers with every
    column it answered with before — an aggregate's new keys come first, so
    read them by name — plus every inner key and every column a residual
    reads.
    """

    var relation: DynRelation
    var inner: List[String]
    """The subquery's side of each equi-correlation, parallel to `outer`."""
    var outer: List[String]
    """The enclosing query's side."""
    var residuals: List[DynValue]
    """Correlated conjuncts that are not an equality of two columns."""
    var grouped: Bool
    """Whether a keyless aggregate took the inner keys as its group keys, so
    the subquery answers at most one row per key — what makes a correlated
    scalar subquery a left join."""


@fieldwise_init
struct Binding(Copyable, Movable):
    """An uncorrelated subquery's slot, the planned relation answering it and
    what it asks of that relation, wrapped around the root as a
    `ScalarBinding`."""

    var slot: String
    var relation: DynRelation
    var kind: SubqueryKind


struct Decorrelate:
    """The pass `AllRules.prepare` and `SubqueryRules.prepare` run first.

    **One function recurses, and only into itself**: `_planned` plans a node's
    children and every subquery body its values read, then hands the planned
    bodies to helpers that never recurse. A cycle through several functions —
    planning a body from inside the helper that joins it — deadlocks the
    compiler when the pass is linked into a shared library.
    """

    @staticmethod
    def apply(plan: DynRelation) raises -> DynRelation:
        """`plan` with every subquery its filters and projections evaluate
        planned, and every uncorrelated binding wrapped around its root; `plan`
        itself when it evaluates none."""
        if len(plan.subqueries()) == 0:
            return plan.copy()
        var bindings = List[Binding]()
        var out = Self._planned(plan, Self._bound_slots(plan), bindings)
        return Self._bound(out^, bindings)

    @staticmethod
    def _bound(
        var plan: DynRelation, bindings: List[Binding]
    ) raises -> DynRelation:
        """`plan` under a `ScalarBinding` per binding."""
        for ref b in bindings:
            plan = ScalarBinding(
                plan^, b.relation.copy(), b.slot.copy(), b.kind
            )
        return plan^

    @staticmethod
    def _bound_slots(plan: DynRelation) raises -> List[String]:
        """Every slot a `ScalarBinding` in `plan` already binds."""
        var out = List[String]()
        var pending: List[DynRelation] = [plan.copy()]
        while len(pending) > 0:
            var node = pending.pop()
            if node.isa[ScalarBinding]():
                out.append(node.get[ScalarBinding]().slot.copy())
            for ref child in node.children():
                pending.append(child.copy())
        return out^

    @staticmethod
    def _planned(
        node: DynRelation, known: List[String], mut bindings: List[Binding]
    ) raises -> DynRelation:
        """Children first, then this node's own subqueries, each body planned
        here — the pass's only recursion."""
        var children = List[DynRelation]()
        for ref child in node.children():
            children.append(Self._planned(child, known, bindings))
        var rebuilt = node.with_children(children)
        if not rebuilt.isa[Filter]() and not rebuilt.isa[Project]():
            return rebuilt^
        var subqueries = Self._read_by(rebuilt)
        var bodies = List[DynRelation](capacity=len(subqueries))
        for ref sq in subqueries:
            var inner = List[Binding]()
            var body = Self._planned(
                sq.relation[], Self._bound_slots(sq.relation[]), inner
            )
            bodies.append(Self._bound(body^, inner))
        if rebuilt.isa[Filter]():
            return Self._filter(rebuilt, subqueries, bodies, known, bindings)
        return Self._project(rebuilt, subqueries, bodies, known, bindings)

    @staticmethod
    def _read_by(node: DynRelation) -> List[Subquery]:
        """The subqueries a filter's or projection's own values evaluate."""
        if node.isa[Filter]():
            return node.get[Filter]().predicate.subqueries()
        var out = List[Subquery]()
        for ref v in node.get[Project]().values:
            for ref sq in v.subqueries():
                var seen = False
                for ref kept in out:
                    seen = seen or kept.id == sq.id
                if not seen:
                    out.append(sq.copy())
        return out^

    @staticmethod
    def _body(
        sq: Subquery, subqueries: List[Subquery], bodies: List[DynRelation]
    ) raises -> DynRelation:
        """`sq`'s planned body."""
        for i in range(len(subqueries)):
            if subqueries[i].id == sq.id:
                return bodies[i].copy()
        raise InternalError(t"subquery: {sq} was not planned with its node")

    # -- the nodes that read a subquery -------------------------------------

    @staticmethod
    def _filter(
        node: DynRelation,
        subqueries: List[Subquery],
        bodies: List[DynRelation],
        known: List[String],
        mut bindings: List[Binding],
    ) raises -> DynRelation:
        """A filter whose conjuncts may test or read subqueries.

        A conjunct that is nothing but a correlated (or `IN`) test becomes a
        join and is dropped; any other conjunct keeps reading its subqueries'
        slots, which are planned below it.
        """
        ref f = node.get[Filter]()
        var input = f.input[].copy()
        var names = input.schema().names()
        var current = input.copy()
        # A correlated filter in the run of filters below is lifted above the
        # joins this plans: it reads only columns they keep, so it commutes
        # with them and with its neighbours, and above them is where the
        # enclosing query's decorrelation looks for it.
        var lifted = List[DynValue]()
        var stayed = List[DynValue]()
        var base = input.copy()
        while base.isa[Filter]():
            ref below = base.get[Filter]()
            if len(below.predicate.outer_columns()) > 0:
                lifted.append(below.predicate.copy())
            else:
                stayed.append(below.predicate.copy())
            var next = below.input[].copy()
            base = next^
        if len(lifted) > 0:
            current = base^
            for i in range(len(stayed) - 1, -1, -1):
                current = current.filter(stayed[i].copy())
        var kept = List[DynValue]()
        var changed = False
        for ref c in f.terms():
            var test = c.subquery()
            if test and Self._joins(test.value()):
                ref sq = test.value()
                current = Self._test(
                    current, sq, Self._body(sq, subqueries, bodies), bindings
                )
                changed = True
                continue
            for ref sq in c.subqueries():
                current = Self._value(
                    current,
                    sq,
                    Self._body(sq, subqueries, bodies),
                    known,
                    bindings,
                    changed,
                )
            kept.append(c.copy())
        if not changed:
            return node.copy()
        var out = current^
        for i in range(len(lifted) - 1, -1, -1):
            out = out.filter(lifted[i].copy())
        for ref c in kept:
            out = out.filter(c.copy())
        if out.schema().names() != names:
            out = out.select(names)
        return out^

    @staticmethod
    def _project(
        node: DynRelation,
        subqueries: List[Subquery],
        bodies: List[DynRelation],
        known: List[String],
        mut bindings: List[Binding],
    ) raises -> DynRelation:
        """A projection reading subqueries: each planned below it, the
        projection itself unchanged — it names its outputs, so a column added
        below never reaches above it."""
        ref p = node.get[Project]()
        var current = p.input[].copy()
        var changed = False
        for i in range(len(subqueries)):
            current = Self._value(
                current, subqueries[i], bodies[i], known, bindings, changed
            )
        if not changed:
            return node.copy()
        return node.with_children([current^])

    # -- what each subquery becomes -----------------------------------------

    @staticmethod
    def _joins(sq: Subquery) -> Bool:
        """Whether a conjunct that is exactly this test becomes a join: every
        test except an uncorrelated existence test, which is a binding."""
        return sq.kind != SubqueryKind.scalar and not sq.bound()

    @staticmethod
    def _test(
        current: DynRelation,
        sq: Subquery,
        planned: DynRelation,
        mut bindings: List[Binding],
    ) raises -> DynRelation:
        """`current` semi- or anti-joined to the subquery: what a filter
        conjunct `[NOT] EXISTS` / `[NOT] IN` keeps."""
        var c = Self._pull(planned)
        var left = current.copy()
        var left_keys = List[String]()
        var right_keys = List[String]()
        if sq.kind.is_membership():
            ref operand = sq.operand.value()[]
            var read = operand.read_column()
            var name: String
            if read:
                name = read.value()
            else:
                name = sq.slot() + "_in"
                left = left.with_columns([name.copy()], [operand.copy()])
            left_keys.append(name^)
            right_keys.append(planned.schema().fields[0].name.copy())
        for i in range(len(c.outer)):
            left_keys.append(c.outer[i].copy())
            right_keys.append(c.inner[i].copy())
        if len(left_keys) == 0:
            raise InternalError(t"subquery: {sq} has no key to join on")

        var anti = sq.kind == SubqueryKind.not_exists
        var nullable_not_in = False
        if sq.kind == SubqueryKind.not_in:
            anti = True
            # Without a null on either side `NOT IN` is exactly "no row
            # matches", which is what an anti join keeps; a null key on the
            # correlation columns matches nothing, which is also right.
            nullable_not_in = not (
                Self._never_null(left, left_keys[0])
                and Self._never_null(c.relation, right_keys[0])
            )
        if len(c.residuals) > 0:
            if sq.kind.is_membership():
                raise Self._not_equality(sq, c.residuals[0])
            var out: DynRelation = ResidualJoin(
                left^,
                c.relation.copy(),
                left_keys^,
                right_keys^,
                c.residuals.copy(),
                anti,
            )
            return out^
        if nullable_not_in:
            return Self._not_in(left, c, left_keys, right_keys, bindings)
        return left.join(
            c.relation.copy(),
            left.schema().field_indices(left_keys),
            c.relation.schema().field_indices(right_keys),
            JOIN_ANTI if anti else JOIN_SEMI,
            BUILD_RIGHT,
        )

    @staticmethod
    def _not_in(
        left: DynRelation,
        c: Correlation,
        left_keys: List[String],
        right_keys: List[String],
        mut bindings: List[Binding],
    ) raises -> DynRelation:
        """`x NOT IN (SELECT y ...)` where a null may appear on either side.

        Three-valued: a row survives when no `y` equals its `x` *and* the
        answer is not unknown — unknown when `y` holds a null, or when `x` is
        null and there is any `y` at all. An anti join answers the first part;
        two uncorrelated bindings, "some `y` is null" and "there is no `y`",
        answer the rest. All three read the subquery's rows once, shared.
        Correlated, the bindings would be per outer row, which needs a mark
        join.
        """
        if len(c.outer) > 0:
            raise NotImplementedError(
                "subquery: a correlated NOT IN over columns that may hold "
                "nulls needs a mark join"
            )
        var rows = c.relation.shared()
        var nulls = rows.filter(is_null(column(right_keys[0])))
        var no_null = not_(exists(nulls^))
        var no_rows = not_(exists(rows.copy()))
        for ref test in [no_null.copy(), no_rows.copy()]:
            var sq = test.subquery().value().copy()
            bindings.append(Binding(sq.slot(), sq.relation[].limit(1), sq.kind))
        var keep = and_(no_null^, or_(is_valid(column(left_keys[0])), no_rows^))
        return left.join(
            rows.copy(),
            left.schema().field_indices(left_keys),
            rows.schema().field_indices(right_keys),
            JOIN_ANTI,
            BUILD_RIGHT,
        ).filter(keep^)

    @staticmethod
    def _value(
        current: DynRelation,
        sq: Subquery,
        planned: DynRelation,
        known: List[String],
        mut bindings: List[Binding],
        mut changed: Bool,
    ) raises -> DynRelation:
        """`current` with `sq`'s slot made available to the node above it: a
        binding recorded for the root, or a column joined on."""
        var slot = sq.slot()
        if sq.bound():
            if slot in known:
                return current.copy()
            for ref b in bindings:
                if b.slot == slot:
                    return current.copy()
            # An existence test needs one row to answer.
            var body = planned.copy()
            if sq.kind != SubqueryKind.scalar:
                body = body.limit(1)
            bindings.append(Binding(slot^, body^, sq.kind))
            return current.copy()
        if slot in current.schema().names():
            return current.copy()
        changed = True
        if sq.kind == SubqueryKind.scalar:
            return Self._scalar(current, sq, planned)
        raise NotImplementedError(
            t"subquery: {sq} is read as a value rather than tested in a "
            t"filter conjunct, which needs a mark join"
        )

    @staticmethod
    def _scalar(
        current: DynRelation, sq: Subquery, planned: DynRelation
    ) raises -> DynRelation:
        """`current` left-joined to the subquery's aggregate, grouped by its
        correlation keys, with the value under the slot.

        A row with no group gets what the aggregate answers over no rows —
        `count` is 0 there, not null. When that answer is not null the value
        is chosen by a marker column the left join leaves null exactly for
        such rows, so a group whose own answer is null keeps it.
        """
        var c = Self._pull(planned, current.copy())
        if len(c.residuals) > 0:
            raise Self._not_equality(sq, c.residuals[0])
        if not c.grouped:
            raise NotImplementedError(
                t"subquery: correlated scalar subquery {sq} must aggregate "
                t"without GROUP BY, so it has one row per outer row"
            )
        var slot = sq.slot()
        var empty = Self._over_no_rows(planned)
        var source = c.relation.schema()
        var raw = slot + "_raw" if empty else slot.copy()
        var marker = slot + "_m"
        # By name: an aggregate answers with its keys first, and the
        # correlation keys just became some.
        var value = planned.schema().fields[0].name.copy()
        var names: List[String] = [raw.copy()]
        var values: List[DynValue] = [FieldRef(source.field(name=value).copy())]
        var right_keys = List[Int]()
        for i in range(len(c.inner)):
            names.append(String(slot, "_k", i))
            values.append(FieldRef(source.field(name=c.inner[i]).copy()))
            right_keys.append(i + 1)
        if empty:
            names.append(marker.copy())
            values.append(literal(BoolScalar(True).to_dyn()))
        var right = c.relation.project(names^, values^)
        var joined = current.join(
            right^,
            current.schema().field_indices(c.outer),
            right_keys^,
            JOIN_LEFT,
            BUILD_RIGHT,
        )
        if empty:
            var chosen = List[DynValue]()
            chosen.append(
                if_else(
                    is_null(column(marker)),
                    literal(empty.value().copy()),
                    column(raw),
                )
            )
            joined = joined.with_columns([slot.copy()], chosen^)
        return joined^

    # -- correlation --------------------------------------------------------

    @staticmethod
    def _pull(
        rel: DynRelation, domain: Optional[DynRelation] = None
    ) raises -> Correlation:
        """`rel` with its correlated conjuncts lifted to the top.

        `domain`, when given, is the enclosing query's input: an aggregate the
        correlation keys become group keys of then aggregates only the groups
        that input can ask for, when that is much less than all of them.
        """
        if len(rel.outer_columns()) == 0:
            return Correlation(
                rel.copy(),
                List[String](),
                List[String](),
                List[DynValue](),
                False,
            )
        if rel.isa[Filter]():
            ref f = rel.get[Filter]()
            var below = Self._pull(f.input[], domain)
            var kept = List[DynValue]()
            for ref c in f.terms():
                if len(c.outer_columns()) == 0:
                    kept.append(c.copy())
                    continue
                var pair = c.correlation()
                if pair:
                    below.inner.append(pair.value()[0].copy())
                    below.outer.append(pair.value()[1].copy())
                else:
                    below.residuals.append(c.copy())
            for ref c in kept:
                below.relation = below.relation.filter(c.copy())
            return below^
        if rel.isa[Project]():
            ref p = rel.get[Project]()
            for ref v in p.values:
                Self._refuse_outer(v, "the select list")
            var below = Self._pull(p.input[], domain)
            var names = p.names.copy()
            var values = p.values.copy()
            var source = below.relation.schema()
            var carried = below.inner.copy()
            for ref r in below.residuals:
                for ref col in r.columns():
                    carried.append(col.copy())
            for ref k in carried:
                if k in names:
                    if not p.passes_through(k):
                        raise NotImplementedError(
                            t"subquery: the select list replaces '{k}', which "
                            t"a correlated condition reads"
                        )
                    continue
                names.append(k.copy())
                values.append(FieldRef(source.field(name=k).copy()))
            below.relation = Project(below.relation.copy(), names^, values^)
            return below^
        if rel.isa[Aggregate]():
            ref a = rel.get[Aggregate]()
            for ref v in a.keys:
                Self._refuse_outer(v, "GROUP BY")
            for ref v in a.aggs:
                Self._refuse_outer(v, "an aggregate")
            var below = Self._pull(a.input[])
            if len(below.residuals) > 0:
                raise NotImplementedError(
                    t"subquery: '{below.residuals[0]}' correlates below an "
                    t"aggregate and is not an equality between two columns"
                )
            var keys = a.keys.copy()
            var source = below.relation.schema()
            var seen = List[String]()
            var asked = List[String]()
            for i in range(len(below.inner)):
                ref k = below.inner[i]
                if k not in seen:
                    keys.append(FieldRef(source.field(name=k).copy()))
                    seen.append(k.copy())
                    asked.append(below.outer[i].copy())
            if domain and Self._worth_reducing(domain.value(), below.relation):
                # Magic set: only the groups some outer row asks for.
                var wanted = domain.value().select(asked)
                below.relation = below.relation.join(
                    wanted.copy(),
                    below.relation.schema().field_indices(seen),
                    wanted.schema().field_indices(asked),
                    JOIN_SEMI,
                    BUILD_RIGHT,
                )
            below.relation = Aggregate(
                below.relation.copy(), keys^, a.aggs.copy()
            )
            below.grouped = len(a.keys) == 0
            return below^
        if rel.isa[ScalarBinding]():
            ref b = rel.get[ScalarBinding]()
            var below = Self._pull(b.input[], domain)
            below.relation = rel.with_children(
                [below.relation.copy(), b.subquery[].copy()]
            )
            return below^
        raise NotImplementedError(
            t"subquery: a correlated condition below {Self._node_name(rel)} "
            t"is not supported; write it in the subquery's WHERE"
        )

    @staticmethod
    def _refuse_outer(value: DynValue, place: StringSlice) raises:
        var outer = value.outer_columns()
        if len(outer) > 0:
            raise NotImplementedError(
                t"subquery: {place} reads the enclosing query's "
                t"'{outer[0]}', which is not supported"
            )

    @staticmethod
    def _not_equality(sq: Subquery, residual: DynValue) -> NotImplementedError:
        return NotImplementedError(
            t"subquery: {sq} correlates on '{residual}', which is not an "
            t"equality between two columns"
        )

    @staticmethod
    def _node_name(rel: DynRelation) -> String:
        if rel.isa[JoinChain]():
            return "a join"
        if rel.isa[Sort]():
            return "an ORDER BY"
        if rel.isa[Limit]():
            return "a LIMIT"
        if rel.isa[Window]():
            return "a window"
        return "a set operation"

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _over_no_rows(planned: DynRelation) raises -> Optional[DynScalar]:
        """What a keyless aggregate subquery answers over no rows, when that is
        not null: its aggregate run over an empty input, the projections and
        `HAVING` above it kept."""
        var empty = Self._emptied(planned)
        var batch = empty.execute(ExecContext.serial())
        if batch.num_rows() == 0 or batch.column(0).is_null(0):
            return None
        return batch.column(0)[0]

    @staticmethod
    def _emptied(rel: DynRelation) raises -> DynRelation:
        """`rel` with the input of its aggregate replaced by no rows."""
        if rel.isa[Aggregate]():
            ref a = rel.get[Aggregate]()
            var none: DynRelation = EmptyRelation(
                RecordBatch.empty(a.input[].schema())
            )
            return a.with_input(none^)
        if rel.isa[Project]() or rel.isa[Filter]() or rel.isa[ScalarBinding]():
            # Each reads its input first; a binding's subquery is left whole.
            var children = rel.children()
            children[0] = Self._emptied(children[0])
            return rel.with_children(children)
        raise InternalError("subquery: no aggregate to evaluate over no rows")

    @staticmethod
    def _worth_reducing(domain: DynRelation, input: DynRelation) -> Bool:
        """Whether semi-joining an aggregate's input with the keys the outer
        query asks for pays for computing that query twice: only when both
        sizes are estimated and the outer side is a tenth of the input or
        less. Unknown sizes keep the plain plan."""
        try:
            var outer = domain.estimate().rows.known()
            var inner = input.estimate().rows.known()
            if outer and inner:
                return outer.value() * 10 <= inner.value()
        except:
            pass
        return False

    @staticmethod
    def _never_null(rel: DynRelation, name: String) raises -> Bool:
        return not rel.schema().field(name=name).nullable
