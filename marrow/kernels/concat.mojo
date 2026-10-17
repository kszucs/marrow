# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Batch concatenation on a context: in parallel when it is large.

`StructBuilder.concat` is the concatenation every array type has, on the
calling thread. `concat` here takes a context, and a concatenation large
enough to share out (`ExecContext.worth_parallel`) allocates the output once
and copies each input to its own offset as a task of its own: fixed-width
values, the bytes and rebased offsets of strings and binaries, and a struct's
children. The copy is of bytes, shared by every type; temporal, decimal and
interval columns keep the builder, and validity bits are joined on the
calling thread since two inputs can share an output byte.
"""

from std.sys import size_of

from ..arrays import BinaryLikeArray, DynArray, PrimitiveArray, StructArray
from ..buffers import Bitmap, Buffer
from ..builders import DynBuilder, StructBuilder, array
from ..dtypes import BinaryLikeType, NumericType, PrimitiveType, struct_
from ..execution import ExecContext
from ..schema import Schema


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
        var columns = List[DynArray](capacity=len(schema.fields))
        for ref f in schema.fields:
            columns.append(array(f.dtype))
        return StructArray(
            dtype=struct_(schema.fields.copy()),
            length=0,
            nulls=0,
            offset=0,
            bitmap=None,
            children=columns^,
        )
    var total = 0
    for ref b in batches:
        total += len(b)
    if len(batches) > 1 and ctx.worth_parallel(total, _PARALLEL_CONCAT_ROWS):
        var parts = List[DynArray](capacity=len(batches))
        for ref b in batches:
            parts.append(b.copy().to_dyn())
        var parallel = _concat_parallel(parts, total, ctx)
        if parallel:
            return parallel.take().as_struct().copy()
    return StructBuilder.concat(batches)


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
    var offset: Int
    """The input's first row, in its own bitmap and offsets."""
    var rows: Int
    var row_at: Int
    """Row of the output its first row lands at."""


struct _Pieces(Movable):
    """Every input's part of one parallel concatenation, in order: where each
    one's bytes and rows land, and the output's totals. Not generic: the copy
    is of bytes, so one instantiation serves every type."""

    var _pieces: List[_Piece]
    var _rows: Int
    var _bytes: Int
    var _nulls: Int

    def __init__(out self, capacity: Int):
        self._pieces = List[_Piece](capacity=capacity)
        self._rows = 0
        self._bytes = 0
        self._nulls = 0

    def add(
        mut self,
        source: Buffer[mut=False],
        start: Int,
        bytes: Int,
        bitmap: Optional[Bitmap[mut=False]],
        offset: Int,
        rows: Int,
        nulls: Int,
    ):
        """The next input: `bytes` of `source` from `start`, and `rows` rows
        from `offset`, `nulls` of them null."""
        self._pieces.append(
            _Piece(
                source,
                start,
                bytes,
                self._bytes,
                bitmap,
                offset,
                rows,
                self._rows,
            )
        )
        self._rows += rows
        self._bytes += bytes
        self._nulls += nulls

    def nulls(self) -> Int:
        return self._nulls

    def gather(self, ctx: ExecContext) raises -> Buffer[mut=False]:
        """Every input's bytes, one task per input, in one buffer."""
        var out = Buffer.alloc_uninit[DType.uint8](max(self._bytes, 1))

        def copy(w: Int, i: Int) raises {mut out, imm}:
            ref piece = self._pieces[i]
            if piece.bytes > 0:
                out.extend(
                    piece.source.view[DType.uint8](piece.start, piece.bytes),
                    piece.at,
                    piece.bytes,
                )

        ctx.fan_out(len(self._pieces), copy, len(self._pieces))
        return out^.to_immutable()

    def validity(self) raises -> Optional[Bitmap[mut=False]]:
        """Every input's validity bits joined, or `None` when every row is
        valid. Serial: two inputs may share an output byte."""
        if self._nulls == 0:
            return None
        var bits = Bitmap.alloc_zeroed(self._rows)
        for ref piece in self._pieces:
            if piece.bitmap:
                bits.extend(
                    piece.bitmap.value().view(piece.offset, piece.rows),
                    piece.row_at,
                    piece.rows,
                )
            else:
                bits.set_range(piece.row_at, piece.rows, True)
        return Optional(bits^.to_immutable())

    def rebased_offsets[
        O: DType
    ](
        self, offsets: List[Buffer[mut=False]], ctx: ExecContext
    ) raises -> Buffer[mut=False]:
        """The output's offsets: each input's own, `offsets[i]`, shifted to
        where its bytes land. Generic over the offset width only, so two
        instantiations serve every string and binary type."""
        var out = Buffer.alloc_uninit[O](self._rows + 1)

        def rebase(w: Int, i: Int) raises {mut out, imm}:
            ref piece = self._pieces[i]
            var view = offsets[i].view[O]()
            var first = Int(view[piece.offset])
            for j in range(piece.rows):
                out.unsafe_set[O](
                    piece.row_at + j,
                    Scalar[O](piece.at + Int(view[piece.offset + j]) - first),
                )

        ctx.fan_out(len(self._pieces), rebase, len(self._pieces))
        out.unsafe_set[O](self._rows, Scalar[O](self._bytes))
        return out^.to_immutable()


@no_inline
def _concat_parallel(
    arrays: List[DynArray], total: Int, ctx: ExecContext
) raises -> Optional[DynArray]:
    """`arrays` copied into one output at once, or `None` for a type this
    does not cover, which the caller hands to the builder."""
    var dtype = arrays[0].dtype()
    if dtype.is_struct():
        # Each child on its own; a struct with null rows of its own goes to
        # the builder.
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
            var child = _concat_parallel(parts, total, ctx)
            if child:
                children.append(child.take())
            else:
                children.append(DynBuilder.concat(parts))
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
    # Numeric only: every type covered is an arm of its own in every binary
    # that collects, so temporal, decimal and interval columns keep the
    # builder.
    if dtype.is_numeric():

        def numeric[T: NumericType](d: T) raises {imm} -> Optional[DynArray]:
            return _concat_primitive[T](arrays, total, ctx)

        return dtype.try_visit[Optional[DynArray], NumericType](
            numeric, "dtype is not numeric"
        )
    if (
        dtype.is_string()
        or dtype.is_large_string()
        or dtype.is_binary()
        or dtype.is_large_binary()
    ):

        def binary[T: BinaryLikeType](d: T) raises {imm} -> Optional[DynArray]:
            return _concat_binary[T](arrays, total, ctx)

        return dtype.try_visit[Optional[DynArray], BinaryLikeType](
            binary, "dtype is not binarylike"
        )
    return None


def _concat_primitive[
    T: PrimitiveType
](arrays: List[DynArray], total: Int, ctx: ExecContext) raises -> Optional[
    DynArray
]:
    """The typed part is only gathering each input's buffer and building the
    output; the copy is `_Pieces.gather`, shared by every type."""
    comptime width = size_of[Scalar[T.native]]()
    var pieces = _Pieces(len(arrays))
    for ref d in arrays:
        ref a = d.get[PrimitiveArray[T]]()
        pieces.add(
            a.buffer,
            a.offset * width,
            a.length * width,
            a.bitmap,
            a.offset,
            a.length,
            a.nulls,
        )
    return Optional[DynArray](
        PrimitiveArray[T](
            arrays[0].get[PrimitiveArray[T]]().dtype,
            length=total,
            nulls=pieces.nulls(),
            offset=0,
            bitmap=pieces.validity(),
            buffer=pieces.gather(ctx),
        ).to_dyn()
    )


def _concat_binary[
    T: BinaryLikeType
](arrays: List[DynArray], total: Int, ctx: ExecContext) raises -> Optional[
    DynArray
]:
    """Each input's bytes copied to where the inputs before it end, and its
    offsets rebased onto that point."""
    comptime O = T.offset
    var pieces = _Pieces(len(arrays))
    var offsets = List[Buffer[mut=False]](capacity=len(arrays))
    for ref d in arrays:
        ref s = d.get[BinaryLikeArray[T]]()
        var view = s.offsets.view[O]()
        var first = Int(view[s.offset])
        pieces.add(
            s.values,
            first,
            Int(view[s.offset + s.length]) - first,
            s.bitmap,
            s.offset,
            s.length,
            s.nulls,
        )
        offsets.append(s.offsets)
    return Optional[DynArray](
        BinaryLikeArray[T](
            length=total,
            nulls=pieces.nulls(),
            offset=0,
            bitmap=pieces.validity(),
            offsets=pieces.rebased_offsets[O](offsets, ctx),
            values=pieces.gather(ctx),
        ).to_dyn()
    )
