# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""What a JSON read is asked to do — the options of `pyarrow.json`.

`ReadOptions` and `ParseOptions` carry pyarrow's names and defaults so its
muscle memory carries over. Two of pyarrow's fields are absent on purpose:
`use_threads`, because the reader is single-threaded, and `newlines_in_values`,
because a block is always cut at a newline.
"""

from ..schema import Schema


@fieldwise_init
struct UnexpectedFieldBehavior(Equatable, ImplicitlyCopyable, Movable):
    """What to do with a key the explicit schema does not name.

    Without an explicit schema every key is unexpected, so the behaviour is
    forced to `INFER`, as in Arrow C++.
    """

    var code: Int

    comptime IGNORE = Self(0)
    """Skip the value."""
    comptime ERROR = Self(1)
    """Raise."""
    comptime INFER = Self(2)
    """Infer a type for it and add the column after the schema's own."""

    @staticmethod
    def parse(name: String) raises -> Self:
        """`"ignore"`, `"error"` or `"infer"`, as pyarrow spells them."""
        if name == "ignore":
            return Self.IGNORE
        elif name == "error":
            return Self.ERROR
        elif name == "infer":
            return Self.INFER
        raise Error(
            (
                "unexpected_field_behavior must be 'ignore', 'error' or"
                " 'infer', got '"
            ),
            name,
            "'",
        )

    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code


@fieldwise_init
struct ReadOptions(Copyable, Movable):
    """How the input is read."""

    var block_size: Int
    """Bytes per block. A block ends at the last newline inside it, and a row
    longer than a block grows the read rather than failing, where Arrow C++
    raises "straddling object"."""

    def __init__(out self):
        self.block_size = 1 << 20


struct ParseOptions(Copyable, Movable):
    """How each value is turned into a column."""

    var explicit_schema: Optional[Schema]
    """Columns and types to read, in this order; `None` infers everything."""
    var unexpected_field_behavior: UnexpectedFieldBehavior
    """What a key outside `explicit_schema` does."""

    def __init__(out self):
        self.explicit_schema = None
        self.unexpected_field_behavior = UnexpectedFieldBehavior.INFER

    def __init__(
        out self,
        explicit_schema: Schema,
        unexpected_field_behavior: UnexpectedFieldBehavior = (
            UnexpectedFieldBehavior.INFER
        ),
    ):
        self.explicit_schema = explicit_schema.copy()
        self.unexpected_field_behavior = unexpected_field_behavior

    def behavior(self) -> UnexpectedFieldBehavior:
        """The behaviour in force: `INFER` whenever there is no schema."""
        if not self.explicit_schema:
            return UnexpectedFieldBehavior.INFER
        return self.unexpected_field_behavior
