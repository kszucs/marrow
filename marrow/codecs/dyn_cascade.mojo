# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Dynamic cascades: a chain of codecs chosen at run time, the codec box it
holds, and its streaming writer and reader. The stream is `Cascade`'s; it
does not name its chain, so a reader is given the cascade that wrote it.
"""

from std.builtin.rebind import downcast
from std.memory import bitcast
from std.os import abort
from std.sys import size_of
from std.utils import Variant

from ..errors import CorruptError, InvalidError
from .bitpack import BitPack
from .bits import Bits
from .byte_stream_split import ByteStreamSplit
from .byteorder import Leb128, LittleEndian
from .cascade import Cascade
from .constant import Constant
from .core import Append, BLOCK, Codec, Decoder, Elementwise, Values
from .delta import Delta
from .delta_binary_packed import DeltaBinaryPacked
from .dictionary import Dictionary
from .frequency import Frequency
from .hybrid import Hybrid
from .plain import Plain
from .rle import Rle
from .varint import Varint
from .xor import Xor
from .zigzag import Zigzag


struct DynCodec(Copyable, Movable, Writable):
    """Any one codec, chosen at run time: a `Variant` over every codec,
    dispatched as `DynArray` is -- a walk over the members at compile time
    that runs the active one."""

    comptime VariantType = Variant[
        Delta,
        Zigzag,
        Xor,
        BitPack,
        Varint,
        ByteStreamSplit,
        Constant,
        Rle,
        Dictionary,
        Frequency,
        Hybrid,
        DeltaBinaryPacked,
        Plain,
    ]

    var _v: Self.VariantType

    @implicit
    def __init__[C: Codec](out self, var codec: C):
        self._v = Self.VariantType(codec^)

    def _dispatch[
        R: Movable, //, Func: def[C: Codec](C) -> R
    ](self, func: Func) -> R:
        """Run `func` on the active codec."""
        comptime for i in range(len(Self.VariantType.Ts)):
            comptime T = Self.VariantType.Ts[i]
            if self._v.isa[T]():
                return func(rebind[downcast[T, Codec]](self._v[T]))
        abort("DynCodec._dispatch: no arm matched")

    def _dispatch[
        R: Movable, //, Func: def[C: Codec](C) raises -> R
    ](self, func: Func) raises -> R:
        """Raising counterpart of `_dispatch`."""
        comptime for i in range(len(Self.VariantType.Ts)):
            comptime T = Self.VariantType.Ts[i]
            if self._v.isa[T]():
                return func(rebind[downcast[T, Codec]](self._v[T]))
        abort("DynCodec._dispatch: no arm matched")

    def _elementwise[
        R: Movable, //, Func: def[C: Elementwise](C) -> R
    ](self, func: Func) -> R:
        """Run `func` on the active codec, which must be elementwise."""
        comptime for i in range(len(Self.VariantType.Ts)):
            comptime T = Self.VariantType.Ts[i]
            comptime if conforms_to(T, Elementwise):
                if self._v.isa[T]():
                    return func(rebind[downcast[T, Elementwise]](self._v[T]))
        abort("DynCodec._elementwise: not an elementwise codec")

    def name(self) -> StaticString:
        def f[C: Codec](c: C) {} -> StaticString:
            return C.name()

        return self._dispatch(f)

    def is_elementwise(self) -> Bool:
        def f[C: Codec](c: C) {} -> Bool:
            return conforms_to(C, Elementwise)

        return self._dispatch(f)

    def takes[V: DType](self) -> Bool:
        def f[C: Codec](c: C) {} -> Bool:
            return C.takes[V]()

        return self._dispatch(f)

    def out[V: DType](self) -> DType:
        """What the codec hands on for values of `V`."""

        def f[C: Codec](c: C) {} -> DType:
            return C.Out[V]

        return self._dispatch(f)

    def forward[
        V: DType, W: DType
    ](self, mut values: List[Scalar[W]]) -> UInt64:
        """Map `values` -- of `V`, held as their bits in `W`, a type of the
        same width -- through the active elementwise codec, in place; return
        the state it started from."""

        def f[C: Elementwise](c: C) {mut values} -> UInt64:
            comptime if C.takes[V]():
                comptime assert (
                    size_of[Scalar[C.Out[V]]]() == size_of[Scalar[W]]()
                ), "an elementwise codec keeps the width"
                var first = bitcast[V](values[0]) if len(
                    values
                ) > 0 else Scalar[V](0)
                var second = bitcast[V](values[1]) if len(values) > 1 else first
                var start = C.init(first, second)
                var state = start
                for ref v in values:
                    v = bitcast[W](C.forward[V](state, bitcast[V](v)))
                return start
            else:
                abort("DynCodec.forward: the codec does not take the values")

        return self._elementwise(f)

    def inverse[
        V: DType, W: DType
    ](self, state: UInt64, mut values: List[Scalar[W]], start: Int):
        """Map `values[start:]` back through the active elementwise codec, in
        place, from the state `forward` started at, to values of `V` held as
        their bits in `W`."""

        def f[C: Elementwise](c: C) {mut values, imm} -> None:
            comptime if C.takes[V]():
                var s = state
                for ref v in Span(values)[start:]:
                    v = bitcast[W](C.inverse[V](s, bitcast[C.Out[V]](v)))
            else:
                abort("DynCodec.inverse: the codec does not take the values")

        self._elementwise(f)

    def encode[
        V: DType, W: DType
    ](self, values: Span[Scalar[W], _], mut out: List[UInt8]) raises:
        """The active codec's `encode` over `values`, read as `V`."""

        def f[C: Codec](c: C) raises {mut out, imm} -> None:
            comptime if C.takes[V]():
                Self._encode[C, V](values, out)
            else:
                abort("DynCodec.encode: the codec does not take the values")

        self._dispatch(f)

    def decode[
        V: DType, W: DType
    ](self, mut src: Decoder[_], count: Int, mut out: List[Scalar[W]]) raises:
        """The active codec's `decode`, appending values of `V` to `out` as
        their bits in `W`."""

        def f[C: Codec](c: C) raises {mut src, mut out, imm} -> None:
            comptime if C.takes[V]():
                Self._decode[C, V](src, count, out)
            else:
                abort("DynCodec.decode: the codec does not take the values")

        self._dispatch(f)

    @staticmethod
    @no_inline
    def _encode[
        C: Codec, V: DType, W: DType
    ](values: Span[Scalar[W], _], mut out: List[UInt8]) raises:
        """`C.encode` reading `values`, compiled on its own rather than into
        `encode`'s every arm."""
        C.encode[V](Values(values), out)

    @staticmethod
    @no_inline
    def _decode[
        C: Codec, V: DType, W: DType
    ](mut src: Decoder[_], count: Int, mut out: List[Scalar[W]]) raises:
        """`C.decode` appending to `out`, compiled on its own rather than
        into `decode`'s every arm. The emitter owns `out` while the codec
        decodes -- a list it points at would be appended to through memory."""
        var read = Append[W](List[Scalar[W]]())
        swap(read.out, out)
        read = C.decode[V](src, count, read^)
        out = read^.take()

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.name())

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("DynCodec(", self.name(), ")")


struct DynCascade(Copyable, Movable, Writable):
    """A chain of codecs chosen at run time, as in
    `DynCascade(Delta(), Zigzag(), BitPack())` -- printed as
    `Delta -> Zigzag -> BitPack`. Each codec runs over what the one before it
    handed on, one at a time over the whole block; a chain of no codecs
    writes each value's little-endian bytes."""

    var codecs: List[DynCodec]

    def __init__[*Ts: Codec](out self, *codecs: *Ts):
        self.codecs = List[DynCodec](capacity=len(Ts))
        comptime for k in range(len(Ts)):
            self.codecs.append(DynCodec(Ts[k]()))

    @implicit
    def __init__[*Ts: Codec](out self, cascade: Cascade[*Ts]):
        """The chain `cascade` fixes at compile time, chosen at run time."""
        self.codecs = List[DynCodec](capacity=len(Ts))
        comptime for k in range(len(Ts)):
            self.codecs.append(DynCodec(Ts[k]()))

    def __init__(out self, var codecs: List[DynCodec]):
        self.codecs = codecs^

    @staticmethod
    def _out[T: DType](c: DynCodec, cur: DType) raises InvalidError -> DType:
        """What `c` hands on, given values of `cur` in a column of `T`: of
        `T`, of its unsigned twin, or bytes -- an elementwise codec keeps the
        width, any other writes bytes."""
        var takes: Bool
        var out: DType
        if cur == T:
            takes, out = c.takes[T](), c.out[T]()
        elif cur == Bits.unsigned[T]:
            takes, out = c.takes[Bits.unsigned[T]](), c.out[Bits.unsigned[T]]()
        elif cur == DType.uint8:
            takes, out = c.takes[DType.uint8](), c.out[DType.uint8]()
        else:
            raise InvalidError(t"codecs: a chain over {T} reaches {cur}")
        if not takes:
            raise InvalidError(t"codecs: {c.name()} does not take {cur}")
        return out

    def _kinds[T: DType](self) raises InvalidError -> List[DType]:
        """The type each codec is handed, and what the last hands on."""
        var kinds: List[DType] = [T]
        for c in self.codecs:
            kinds.append(Self._out[T](c, kinds[len(kinds) - 1]))
        return kinds^

    def check[T: DType](self) raises InvalidError:
        _ = self._kinds[T]()

    def encode[
        T: DType
    ](self, values: Span[Scalar[T], _], block: Int = BLOCK) raises -> List[
        UInt8
    ]:
        """`values` as one stream."""
        var writer = DynCascadeWriter[T](self.copy(), block)
        var out = List[UInt8]()
        writer.write(values, out)
        writer.finish(out)
        return out^

    def decode[T: DType](self, data: Span[UInt8, _]) raises -> List[Scalar[T]]:
        """Every value of a stream of this chain."""
        var reader = DynCascadeReader[T](self.copy(), data)
        var out = List[Scalar[T]](capacity=reader.count())
        while reader.read(out):
            pass
        return out^

    def write_block[
        T: DType
    ](self, mut values: List[Scalar[T]], mut out: List[UInt8]) raises:
        """One block: `values`, through each codec in turn -- in place while
        they are values, then as the bytes a codec wrote."""
        Leb128.write(out, UInt64(len(values)))
        var at = len(out)
        LittleEndian.append[DType.uint32](out, 0)
        var kinds = self._kinds[T]()
        var bytes = List[UInt8]()
        var in_bytes = False
        var stored = False
        var n = len(self.codecs)
        for k in range(n):
            ref c = self.codecs[k]
            if c.is_elementwise():
                var state: UInt64
                if in_bytes:
                    state = c.forward[DType.uint8](bytes)
                elif kinds[k] == T:
                    state = c.forward[T](values)
                else:
                    state = c.forward[Bits.unsigned[T]](values)
                Leb128.write(out, state)
            elif k == n - 1:
                Self._encode_stage[T](c, kinds[k], in_bytes, values, bytes, out)
                stored = True
            else:
                var inner = List[UInt8]()
                Self._encode_stage[T](
                    c, kinds[k], in_bytes, values, bytes, inner
                )
                Leb128.write(out, UInt64(len(inner)))
                bytes = inner^
                in_bytes = True
        if not stored:
            Self._encode_stage[T](
                Plain(), kinds[n], in_bytes, values, bytes, out
            )
        LittleEndian.write[DType.uint32](out, at, UInt32(len(out) - at - 4))

    @staticmethod
    def _encode_stage[
        T: DType
    ](
        c: DynCodec,
        kind: DType,
        in_bytes: Bool,
        values: List[Scalar[T]],
        bytes: List[UInt8],
        mut out: List[UInt8],
    ) raises:
        """`c.encode` over the current stream: `bytes`, or `values` as
        `kind`."""
        if in_bytes:
            c.encode[DType.uint8](Span(bytes), out)
        elif kind == T:
            c.encode[T](Span(values), out)
        else:
            c.encode[Bits.unsigned[T]](Span(values), out)

    def read_block[
        T: DType
    ](self, mut src: Decoder[_], count: Int, mut out: List[Scalar[T]]) raises:
        """Append one block's `count` values."""
        var kinds = self._kinds[T]()
        var start = len(out)
        var bytes = List[UInt8]()
        self._read[T](0, src, count, kinds, False, out, bytes)
        if len(out) - start != count:
            raise CorruptError(
                t"codecs: a block of {len(out) - start} values, not {count}"
            )

    def _read[
        T: DType
    ](
        self,
        k: Int,
        mut src: Decoder[_],
        count: Int,
        kinds: List[DType],
        in_bytes: Bool,
        mut values: List[Scalar[T]],
        mut bytes: List[UInt8],
    ) raises:
        """Append the `count` values codec `k` was handed -- to `bytes` once
        an earlier codec wrote bytes, else to `values` as `kinds[k]`."""
        var n = len(self.codecs)
        if k == n:
            Self._decode_stage[T](
                Plain(), kinds[k], in_bytes, src, count, values, bytes
            )
        elif self.codecs[k].is_elementwise():
            ref c = self.codecs[k]
            var state = src.varint()
            var start = len(bytes) if in_bytes else len(values)
            self._read[T](k + 1, src, count, kinds, in_bytes, values, bytes)
            # Mapped back in place, while the block is still in cache.
            if in_bytes:
                c.inverse[DType.uint8](state, bytes, start)
            elif kinds[k] == T:
                c.inverse[T](state, values, start)
            else:
                c.inverse[Bits.unsigned[T]](state, values, start)
        elif k == n - 1:
            Self._decode_stage[T](
                self.codecs[k], kinds[k], in_bytes, src, count, values, bytes
            )
        else:
            var m = src.length()
            var inner = List[UInt8]()
            var unused = List[Scalar[T]]()
            self._read[T](k + 1, src, m, kinds, True, unused, inner)
            var from_inner = Decoder(Span(inner))
            Self._decode_stage[T](
                self.codecs[k],
                kinds[k],
                in_bytes,
                from_inner,
                count,
                values,
                bytes,
            )
            from_inner.expect_end()

    @staticmethod
    def _decode_stage[
        T: DType
    ](
        c: DynCodec,
        kind: DType,
        in_bytes: Bool,
        mut src: Decoder[_],
        count: Int,
        mut values: List[Scalar[T]],
        mut bytes: List[UInt8],
    ) raises:
        """`c.decode` onto the current stream: `bytes`, or `values` as
        `kind`."""
        if in_bytes:
            c.decode[DType.uint8](src, count, bytes)
        elif kind == T:
            c.decode[T](src, count, values)
        else:
            c.decode[Bits.unsigned[T]](src, count, values)

    def write_to[W: Writer](self, mut writer: W):
        if len(self.codecs) == 0:
            writer.write("Plain")
        for i in range(len(self.codecs)):
            if i > 0:
                writer.write(" -> ")
            writer.write(self.codecs[i])

    def write_repr_to[W: Writer](self, mut writer: W):
        writer.write("DynCascade(", self, ")")


struct DynCascadeWriter[T: DType](Movable):
    """Writes a column through a `DynCascade` as it arrives: values in chunks of
    any size, a block written whenever one fills."""

    var cascade: DynCascade
    var block: Int
    var buf: List[Scalar[Self.T]]

    def __init__(out self, var cascade: DynCascade, block: Int = BLOCK) raises:
        cascade.check[Self.T]()
        self.cascade = cascade^
        self.block = max(block, 1)
        self.buf = List[Scalar[Self.T]](capacity=self.block)

    def write(
        mut self, values: Span[Scalar[Self.T], _], mut out: List[UInt8]
    ) raises:
        """Take `values`, appending to `out` every block they fill."""
        var i = 0
        while i < len(values):
            var take = min(self.block - len(self.buf), len(values) - i)
            self.buf.extend(values[i : i + take])
            i += take
            if len(self.buf) == self.block:
                self.cascade.write_block(self.buf, out)
                self.buf.clear()

    def finish(mut self, mut out: List[UInt8]) raises:
        """Write the last, partial block and the end of the stream."""
        if len(self.buf) > 0:
            self.cascade.write_block(self.buf, out)
            self.buf.clear()
        out.append(0)


struct DynCascadeReader[o: ImmOrigin, //, T: DType](Movable):
    """Reads a stream of a `DynCascade` a block at a time."""

    var src: Decoder[Self.o]
    var cascade: DynCascade

    def __init__(
        out self, var cascade: DynCascade, data: Span[UInt8, Self.o]
    ) raises:
        cascade.check[Self.T]()
        self.cascade = cascade^
        self.src = Decoder(data)

    def count(self) raises -> Int:
        """The values left to read, counted without decoding them."""
        return self.src.count()

    def read(mut self, mut out: List[Scalar[Self.T]]) raises -> Bool:
        """Append the next block's values; False, having appended none, once
        the stream has ended."""
        var count, end = self.src.block()
        if count == 0:
            self.src.expect_end()
            return False
        self.cascade.read_block(self.src, count, out)
        self.src.end_block(end)
        return True
