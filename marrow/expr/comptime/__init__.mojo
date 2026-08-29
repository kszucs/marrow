"""The comptime lane: nodes whose structure lives in their type.

Relative imports *inside* this package need no backtick escaping, since the
package name never appears in them. Consumers reach these names through
`marrow.expr`, which re-exports them, so the reserved word is spelled once — in
`marrow/expr/__init__.mojo` — rather than at every call site. That was this
docstring's claim while that file was still empty and 33 sites escaped for
themselves; it is true now.
"""


from .leaves import Column, Literal
from .numeric import (
    Add,
    Gt,
    Lt,
    Mul,
    NumericBinary,
    NumericCompare,
    Sub,
)
from .rules import promote, widest_shape
from .core import ComptimeValue, NumericValue
