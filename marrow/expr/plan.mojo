"""Logical relational plan nodes.

``Relation``    — the trait every relational plan node must implement.
``AnyRelation`` — the type-erased, ArcPointer-backed container.

Concrete plan nodes
-------------------
``Scan``    — reads from a named source (leaf node, no child plans).
``Filter``  — applies a boolean predicate to its input.
``Project`` — evaluates a list of expressions to produce output columns.

Example
-------
    var schema = Schema(fields=[Field("x", int64), Field("y", float64)])
    var src = Scan(name="t", schema_=schema)
    var pred = Expr.greater(Expr.input(0), Expr.literal[int64](0))
    var filtered = Filter(input=src, predicate=pred)
"""

from std.memory import ArcPointer
from marrow.schema import Schema
from marrow.expr.expr import Expr


# ---------------------------------------------------------------------------
# Relation trait — interface every relational plan node must implement
# ---------------------------------------------------------------------------


trait Relation(ImplicitlyDestructible, Movable):
    """Interface for immutable relational plan nodes."""

    fn output_schema(self) -> Schema:
        """Return the output schema produced by this plan node."""
        ...

    fn inputs(self) -> List[AnyRelation]:
        """Return child plan nodes (empty for leaf nodes such as Scan)."""
        ...

    fn exprs(self) -> List[Expr]:
        """Return scalar expressions attached to this node."""
        ...

    fn write_to[W: Writer](self, mut writer: W):
        """Format this node for display."""
        ...


# ---------------------------------------------------------------------------
# AnyRelation — type-erased, ArcPointer-backed relational plan container
# ---------------------------------------------------------------------------


struct AnyRelation(ImplicitlyCopyable, Movable, Writable):
    """Type-erased relational plan node.

    Wraps any ``Relation``-conforming type on the heap behind an
    ``ArcPointer`` so copies are O(1) ref-count bumps.
    """

    var _data: ArcPointer[NoneType]
    var _virt_output_schema: fn(ArcPointer[NoneType]) -> Schema
    var _virt_inputs: fn(ArcPointer[NoneType]) -> List[AnyRelation]
    var _virt_exprs: fn(ArcPointer[NoneType]) -> List[Expr]
    var _virt_write_to_string: fn(ArcPointer[NoneType]) -> String
    var _virt_drop: fn(var ArcPointer[NoneType])

    # --- trampolines ---

    @staticmethod
    fn _tramp_output_schema[T: Relation](ptr: ArcPointer[NoneType]) -> Schema:
        return rebind[ArcPointer[T]](ptr)[].output_schema()

    @staticmethod
    fn _tramp_inputs[
        T: Relation
    ](ptr: ArcPointer[NoneType]) -> List[AnyRelation]:
        return rebind[ArcPointer[T]](ptr)[].inputs()

    @staticmethod
    fn _tramp_exprs[T: Relation](ptr: ArcPointer[NoneType]) -> List[Expr]:
        return rebind[ArcPointer[T]](ptr)[].exprs()

    @staticmethod
    fn _tramp_write_to_string[
        T: Relation
    ](ptr: ArcPointer[NoneType]) -> String:
        var s = String()
        rebind[ArcPointer[T]](ptr)[].write_to(s)
        return s^

    @staticmethod
    fn _tramp_drop[T: Relation](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    # --- construction ---

    @implicit
    fn __init__[T: Relation](out self, var value: T):
        var ptr = ArcPointer(value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_output_schema = Self._tramp_output_schema[T]
        self._virt_inputs = Self._tramp_inputs[T]
        self._virt_exprs = Self._tramp_exprs[T]
        self._virt_write_to_string = Self._tramp_write_to_string[T]
        self._virt_drop = Self._tramp_drop[T]

    fn __copyinit__(out self, read copy: Self):
        self._data = copy._data.copy()
        self._virt_output_schema = copy._virt_output_schema
        self._virt_inputs = copy._virt_inputs
        self._virt_exprs = copy._virt_exprs
        self._virt_write_to_string = copy._virt_write_to_string
        self._virt_drop = copy._virt_drop

    # --- public API ---

    fn output_schema(self) -> Schema:
        return self._virt_output_schema(self._data)

    fn inputs(self) -> List[AnyRelation]:
        return self._virt_inputs(self._data)

    fn exprs(self) -> List[Expr]:
        return self._virt_exprs(self._data)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self._virt_write_to_string(self._data))

    # --- downcast ---

    fn downcast[T: Relation](self) -> ArcPointer[T]:
        return rebind[ArcPointer[T]](self._data.copy())

    @always_inline
    fn as_scan(self) -> ArcPointer[Scan]:
        return self.downcast[Scan]()

    @always_inline
    fn as_filter(self) -> ArcPointer[Filter]:
        return self.downcast[Filter]()

    @always_inline
    fn as_project(self) -> ArcPointer[Project]:
        return self.downcast[Project]()

    fn __del__(deinit self):
        self._virt_drop(self._data^)


# ---------------------------------------------------------------------------
# Concrete relation nodes
# ---------------------------------------------------------------------------


struct Scan(Relation):
    """Table / array scan — leaf node with no child plans.

    ``name``    — identifier for the data source.
    ``schema_`` — output schema of this scan.
    """

    var name: String
    var schema_: Schema

    fn __init__(out self, *, var name: String, var schema_: Schema):
        self.name = name^
        self.schema_ = schema_^

    fn output_schema(self) -> Schema:
        return Schema(copy=self.schema_)

    fn inputs(self) -> List[AnyRelation]:
        return List[AnyRelation]()

    fn exprs(self) -> List[Expr]:
        return List[Expr]()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(t"Scan({self.name})")


struct Filter(Relation):
    """Filter — apply a boolean predicate to the input relation.

    ``input``     — child relation.
    ``predicate`` — boolean expression; rows where True are kept.

    Output schema equals the input schema.
    """

    var input: AnyRelation
    var predicate: Expr

    fn __init__(out self, *, var input: AnyRelation, var predicate: Expr):
        self.input = input^
        self.predicate = predicate^

    fn output_schema(self) -> Schema:
        return self.input.output_schema()

    fn inputs(self) -> List[AnyRelation]:
        var result = List[AnyRelation](capacity=1)
        result.append(self.input)
        return result^

    fn exprs(self) -> List[Expr]:
        var result = List[Expr](capacity=1)
        result.append(self.predicate)
        return result^

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(t"Filter(predicate=")
        self.predicate.write_to(writer)
        writer.write(t")")


struct Project(Relation):
    """Projection — evaluate a list of named expressions.

    ``input``   — child relation.
    ``names``   — output column names (parallel to ``exprs_``).
    ``exprs_``  — scalar expressions to evaluate.
    ``schema_`` — pre-computed output schema.
    """

    var input: AnyRelation
    var names: List[String]
    var exprs_: List[Expr]
    var schema_: Schema

    fn __init__(
        out self,
        *,
        var input: AnyRelation,
        var names: List[String],
        var exprs_: List[Expr],
        var schema_: Schema,
    ):
        self.input = input^
        self.names = names^
        self.exprs_ = exprs_^
        self.schema_ = schema_^

    fn output_schema(self) -> Schema:
        return Schema(copy=self.schema_)

    fn inputs(self) -> List[AnyRelation]:
        var result = List[AnyRelation](capacity=1)
        result.append(self.input)
        return result^

    fn exprs(self) -> List[Expr]:
        var result = List[Expr](capacity=len(self.exprs_))
        for ref e in self.exprs_:
            result.append(e)
        return result^

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(t"Project([")
        for i in range(len(self.names)):
            if i > 0:
                writer.write(t", ")
            writer.write(self.names[i])
            writer.write(t"=")
            self.exprs_[i].write_to(writer)
        writer.write(t"])")
