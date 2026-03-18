"""marrow.expr — expression and logical plan system.

Scalar expressions
------------------
``AnyValue``    — type-erased scalar expression node (ArcPointer-backed)
``Value``       — trait every scalar expression node must implement

Concrete scalar nodes: ``Column``, ``Literal``, ``Binary``, ``Unary``,
``IsNull``, ``IfElse``, ``Cast``

Factory functions: ``col()``, ``lit()``, ``if_else()``
Utilities: ``rebuild()``, ``resolve_columns()``
Operator overloads: ``+``, ``-``, ``*``, ``/``, ``>``, ``<``, ``>=``,
``<=``, ``==``, ``!=``, ``&``, ``|``, ``~``, unary ``-``

Relational plans
----------------
``AnyRelation`` — type-erased relational plan node
``Relation``    — trait every relational plan node must implement

Concrete plan nodes: ``Scan``, ``Filter``, ``Project``, ``InMemoryTable``
Plan-building: ``AnyRelation.select()``, ``AnyRelation.filter()``
Factory: ``in_memory_table()``

Rewriting
---------
``Rewrite``    — trait for non-destructive rewrite rules
``AnyRewrite`` — type-erased rule container
``Rewriter``   — bottom-up fixed-point rewrite driver
"""

from marrow.expr.values import (
    # Traits
    Value,
    # Type-erased container
    AnyValue,
    # Concrete nodes
    Column,
    Literal,
    Binary,
    Unary,
    IsNull,
    IfElse,
    Cast,
    # Free-standing factory functions
    col,
    lit,
    if_else,
    # Utilities
    rebuild,
    resolve_columns,
    # Leaf-node kinds
    LOAD,
    LITERAL,
    # BinaryOp constants
    ADD,
    SUB,
    MUL,
    DIV,
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
    AND,
    OR,
    # UnaryOp constants
    NEG,
    ABS,
    NOT,
    # Other node kinds
    IS_NULL,
    IF_ELSE,
    CAST,
    # Dispatch hints
    DISPATCH_AUTO,
    DISPATCH_CPU,
    DISPATCH_GPU,
)
from marrow.expr.relations import (
    Relation,
    AnyRelation,
    Scan,
    Filter,
    Project,
    InMemoryTable,
    in_memory_table,
    # Plan node kind constants
    SCAN_NODE,
    FILTER_NODE,
    PROJECT_NODE,
    IN_MEMORY_TABLE_NODE,
)
from marrow.expr.rewrite import (
    Rewrite,
    AnyRewrite,
    Rewriter,
)
from marrow.expr.executor import (
    ExecutionContext,
    # Value processors
    ValueProcessor,
    AnyValueProcessor,
    ColumnProcessor,
    LiteralProcessor,
    BinaryProcessor,
    UnaryProcessor,
    IsNullProcessor,
    IfElseProcessor,
    build,
    # Relation processors
    RelationProcessor,
    AnyRelationProcessor,
    ScanProcessor,
    FilterProcessor,
    ProjectProcessor,
    plan,
    execute,
)
