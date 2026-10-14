# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Batch concatenation on a context: in parallel when it is large.

`StructBuilder.concat` is the concatenation every array type has, on the
calling thread. `concat` here takes a context, and a concatenation of 65,536
rows or more on one with several threads allocates the output once and copies
each input to its own offset as a task of its own: fixed-width values, the
bytes and rebased offsets of strings and binaries, and a struct's children.
The copy is of bytes, shared by every type, and out of line; temporal,
decimal and interval columns keep the builder, and validity bits are joined
on the calling thread since two inputs can share an output byte.
`-D MARROW_DRIVER=serial` compiles the parallel path out.
"""

from std.sys import size_of

from ..arrays import BinaryLikeArray, DynArray, PrimitiveArray, StructArray
from ..buffers import Bitmap, Buffer
from ..builders import DynBuilder, StructBuilder
from ..dtypes import BinaryLikeType, NumericType, PrimitiveType
from ..execution import PARALLEL_ENABLED, ExecContext
from ..schema import Schema
from ..tabular import RecordBatch


comptime _PARALLEL_CONCAT_ROWS = 65_536
"""Rows below which concatenating is a short serial copy, not worth a
dispatch."""


def concat(
    batches: List[StructArray],
    schema: Schema,
    ctx: ExecContext = ExecContext.serial(),
) raises -> StructArray:
    """Concatenate batches into one, on `ctx`'s threads when they are many
    rows.

    `schema` describes the batches, and is read only when there are none: the
    one case with no array to take a type from, which answers an empty batch
    with a zero-length column per field.

    Args:
        batches: Batches with the same dtype, possibly none.
        schema: Their fields.
        ctx: Execution context; its threads copy a large concatenation.
    """
    if len(batches) == 0:
        return RecordBatch.empty(schema).to_struct_array()
    comptime if PARALLEL_ENABLED:
        var total = 0
        for ref b in batches:
            total += len(b)
        if (
            len(batches) > 1
            and total >= _PARALLEL_CONCAT_ROWS
            and ctx.resolved_num_threads() > 1
            and not ctx.is_gpu()
        ):
            var parts = List[DynArray](capacity=len(batches))
            for ref b in batches:
                parts.append(b.copy().to_dyn())
            var parallel = _concat_parallel(parts, total, ctx)
            if parallel:
                return parallel.take().as_struct().copy()
    return StructBuilder.concat(batches)


def _concat_column(
    arrays: List[DynArray], total: Int, ctx: ExecContext
) raises -> DynArray:
    """One column of a parallel concatenation: copied at once when its type
    is covered, built otherwise."""
    var parallel = _concat_parallel(arrays, total, ctx)
    if parallel:
        return parallel.take()
    return DynBuilder.concat(arrays)


@fieldwise_init
struct _Piece(Copyable, Movable):
    """One input's part of a parallel concatenation, its type erased: the
    bytes to copy and where they land, and its validity bits."""

    var source: Buffer[mut=False]
    var start: Int
    """First byte of `source` to copy."""
    var bytes: Int
    var at: Int
    """Byte of the output it lands at."""
    var bitmap: Optional[Bitmap[mut=False]]
    var bit_offset: Int
    """First bit of `bitmap` that is the input's first row."""
    var rows: Int
    var row_at: Int
    """Row of the output its first row lands at."""


def _copy_pieces(
    pieces: List[_Piece], mut out: Buffer[mut=True], ctx: ExecContext
) raises:
    """Every piece's bytes into `out`, one task per piece. One instantiation
    for every type: the copy is of bytes."""

    def copy(w: Int, i: Int) raises {mut out, imm}:
        ref piece = pieces[i]
        if piece.bytes > 0:
            out.extend(
                piece.source.view[DType.uint8](piece.start, piece.bytes),
                piece.at,
                piece.bytes,
            )

    ctx.fan_out(len(pieces), copy, ctx.resolved_num_threads())


def _validity(
    pieces: List[_Piece], total: Int, nulls: Int
) raises -> Optional[Bitmap[mut=False]]:
    """The pieces' validity bits joined, or `None` when every row is valid.
    Serial: two pieces may share an output byte."""
    if nulls == 0:
        return None
    var bits = Bitmap.alloc_zeroed(total)
    for ref piece in pieces:
        if piece.bitmap:
            bits.extend(
                piece.bitmap.value().view(piece.bit_offset, piece.rows),
                piece.row_at,
                piece.rows,
            )
        else:
            bits.set_range(piece.row_at, piece.rows, True)
    return Optional(bits^.to_immutable())


@no_inline
def _concat_parallel(
    arrays: List[DynArray], total: Int, ctx: ExecContext
) raises -> Optional[DynArray]:
    """`arrays` copied into one output at once, or `None` for a type this
    does not cover, which the caller hands to the builder."""
    var dtype = arrays[0].dtype()
    if dtype.is_struct():
        return _concat_struct(arrays, total, ctx)
    # Numeric only: every type covered is an arm of its own in every binary
    # that collects, so temporal, decimal and interval columns keep the
    # builder.
    if dtype.is_numeric():

        def numeric[T: NumericType](d: T) raises {imm} -> Optional[DynArray]:
            return _concat_primitive[T](arrays, total, ctx)

        return dtype.dispatch_numeric(numeric)
    if (
        dtype.is_string()
        or dtype.is_large_string()
        or dtype.is_binary()
        or dtype.is_large_binary()
    ):

        def binary[T: BinaryLikeType](d: T) raises {imm} -> Optional[DynArray]:
            return _concat_binary[T](arrays, total, ctx)

        return dtype.dispatch_binarylike(binary)
    return None


def _concat_struct(
    arrays: List[DynArray], total: Int, ctx: ExecContext
) raises -> Optional[DynArray]:
    """Each child concatenated on its own; a struct with null rows of its own
    goes to the builder."""
    for ref a in arrays:
        if a.null_count() > 0:
            return None
    ref first = arrays[0].as_struct()
    var fields = len(first.dtype.as_struct().fields)
    var children = List[DynArray](capacity=fields)
    for f in range(fields):
        var parts = List[DynArray](capacity=len(arrays))
        for ref a in arrays:
            parts.append(a.as_struct().field(f))
        children.append(_concat_column(parts, total, ctx))
    return Optional[DynArray](
        StructArray(
            dtype=first.dtype.copy(),
            length=total,
            nulls=0,
            offset=0,
            bitmap=None,
            children=children^,
        ).to_dyn()
    )


def _concat_primitive[
    T: PrimitiveType
](arrays: List[DynArray], total: Int, ctx: ExecContext) raises -> Optional[
    DynArray
]:
    """The typed part is only gathering each input's buffer and building the
    output; the copy is `_copy_pieces`, shared by every type."""
    comptime width = size_of[Scalar[T.native]]()
    var pieces = List[_Piece](capacity=len(arrays))
    var at = 0
    var nulls = 0
    for ref d in arrays:
        ref a = d.as_type[PrimitiveArray[T]]()
        pieces.append(
            _Piece(
                a.buffer,
                a.offset * width,
                a.length * width,
                at * width,
                a.bitmap,
                a.offset,
                a.length,
                at,
            )
        )
        at += a.length
        nulls += a.nulls
    var values = Buffer.alloc_uninit[DType.uint8](max(total * width, 1))
    _copy_pieces(pieces, values, ctx)
    return Optional[DynArray](
        PrimitiveArray[T](
            arrays[0].as_type[PrimitiveArray[T]]().dtype,
            length=total,
            nulls=nulls,
            offset=0,
            bitmap=_validity(pieces, total, nulls),
            buffer=values^.to_immutable(),
        ).to_dyn()
    )


def _rebased_offsets[
    O: DType
](
    offsets: List[Buffer[mut=False]],
    pieces: List[_Piece],
    total: Int,
    bytes: Int,
    ctx: ExecContext,
) raises -> Buffer[mut=False]:
    """The output's offsets: each input's own, shifted to where its bytes
    land. Generic over the offset width only, so two instantiations serve
    every string and binary type."""
    var out = Buffer.alloc_uninit[O](total + 1)

    def rebase(w: Int, i: Int) raises {mut out, imm}:
        ref piece = pieces[i]
        var view = offsets[i].view[O]()
        var first = Int(view[piece.bit_offset])
        for j in range(piece.rows):
            out.unsafe_set[O](
                piece.row_at + j,
                Scalar[O](piece.at + Int(view[piece.bit_offset + j]) - first),
            )

    ctx.fan_out(len(pieces), rebase, ctx.resolved_num_threads())
    out.unsafe_set[O](total, Scalar[O](bytes))
    return out^.to_immutable()


def _concat_binary[
    T: BinaryLikeType
](arrays: List[DynArray], total: Int, ctx: ExecContext) raises -> Optional[
    DynArray
]:
    """Each input's bytes copied to where the inputs before it end, and its
    offsets rebased onto that point."""
    comptime O = T.offset
    var pieces = List[_Piece](capacity=len(arrays))
    var offsets = List[Buffer[mut=False]](capacity=len(arrays))
    var at = 0
    var bytes = 0
    var nulls = 0
    for ref d in arrays:
        ref s = d.as_type[BinaryLikeArray[T]]()
        var view = s.offsets.view[O]()
        var first = Int(view[s.offset])
        var n = Int(view[s.offset + s.length]) - first
        pieces.append(
            _Piece(s.values, first, n, bytes, s.bitmap, s.offset, s.length, at)
        )
        offsets.append(s.offsets)
        at += s.length
        bytes += n
        nulls += s.nulls
    var values = Buffer.alloc_uninit[DType.uint8](max(bytes, 1))
    _copy_pieces(pieces, values, ctx)
    var out_offsets = _rebased_offsets[O](offsets, pieces, total, bytes, ctx)
    return Optional[DynArray](
        BinaryLikeArray[T](
            length=total,
            nulls=nulls,
            offset=0,
            bitmap=_validity(pieces, total, nulls),
            offsets=out_offsets^,
            values=values^.to_immutable(),
        ).to_dyn()
    )
