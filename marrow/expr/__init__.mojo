"""marrow.expr — expression and logical plan system.

Scalar expressions
------------------
``Expr``     — type-erased scalar expression node (ArcPointer-backed)
``Value``    — trait every scalar expression node must implement

Concrete scalar nodes: ``Column``, ``Literal``, ``Binary``, ``Unary``,
``IsNull``, ``IfElse``, ``Cast``

Node-kind / op constants: ``LOAD``, ``LITERAL``, ``ADD``, ``SUB``,
``MUL``, ``DIV``, ``EQ``, ``NE``, ``LT``, ``LE``, ``GT``, ``GE``,
``AND``, ``OR``, ``NEG``, ``ABS``, ``NOT``, ``IS_NULL``, ``IF_ELSE``,
``CAST``

Dispatch hints: ``DISPATCH_AUTO``, ``DISPATCH_CPU``, ``DISPATCH_GPU``

Relational plans
----------------
``AnyRelation`` — type-erased relational plan node
``Relation``    — trait every relational plan node must implement

Concrete plan nodes: ``Scan``, ``Filter``, ``Project``

Rewriting
---------
``Rewrite``    — trait for non-destructive rewrite rules
``AnyRewrite`` — type-erased rule container
``Rewriter``   — bottom-up fixed-point rewrite driver
"""

from marrow.expr.expr import (
    # Trait
    Value,
    # Type-erased container
    Expr,
    # Concrete nodes
    Column,
    Literal,
    Binary,
    Unary,
    IsNull,
    IfElse,
    Cast,
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
from marrow.expr.plan import (
    Relation,
    AnyRelation,
    Scan,
    Filter,
    Project,
)
from marrow.expr.rewrite import (
    Rewrite,
    AnyRewrite,
    Rewriter,
)
from marrow.expr.executor import (
    ExecutionContext,
    PipelineExecutor,
)
