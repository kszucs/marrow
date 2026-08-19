"""The `.exp` expectation format — rendering and parsing.

Split out of `regenerate.py` so that running the corpus does not import
`duckdb`. Regeneration needs a reference engine; comparison does not, and the
`dev` environment has no duckdb.

The format is one block per case:

    == test_golden_filter_gt
    k:string\tv:int64\tw:int64
    c\t4\t40

A type per column is what lets the text be read back into a *typed* Arrow
table. sqllogictest's `query IIR` letters coerce results before comparing,
which hides exactly the type bugs an Arrow engine should be asserting.
"""

import ast

import pyarrow as pa

NULL = "NULL"

# Only the types the corpus uses are mapped; an unmapped type is an error
# rather than a silent cast.
TYPES = {
    "string": pa.string(),
    "int64": pa.int64(),
    "int32": pa.int32(),
    "double": pa.float64(),
    "bool": pa.bool_(),
}
_NAMES = {str(v): k for k, v in TYPES.items()}


def type_name(dtype):
    try:
        return _NAMES[str(dtype)]
    except KeyError:
        raise SystemExit(f"golden: unmapped arrow type {dtype!r}; add it to TYPES")


def render_value(value):
    if value is None:
        return NULL
    if isinstance(value, float):
        return repr(value)
    return str(value)


def parse_value(text, dtype):
    if text == NULL:
        return None
    if dtype == pa.string():
        return text
    if dtype == pa.bool_():
        return text == "True"
    return ast.literal_eval(text)


def render(table):
    header = "\t".join(f"{f.name}:{type_name(f.type)}" for f in table.schema)
    rows = [
        "\t".join(render_value(v) for v in row.values()) for row in table.to_pylist()
    ]
    return "\n".join([header, *rows])


def parse(text):
    """A whole `.exp` file -> {case name: pyarrow.Table}."""
    tables = {}
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        name = lines[0].removeprefix("== ").strip()
        columns, types = [], []
        for spec in lines[1].split("\t"):
            column, _, tname = spec.partition(":")
            columns.append(column)
            types.append(TYPES[tname])
        rows = [line.split("\t") for line in lines[2:]]
        tables[name] = pa.table(
            {
                column: pa.array(
                    [parse_value(row[i], types[i]) for row in rows], types[i]
                )
                for i, column in enumerate(columns)
            }
        )
    return tables
