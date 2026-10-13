# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`BodyCompression`: how a record batch's body buffers are compressed."""

from ...errors import CorruptError, DynError, InvalidError
from ...buffers import Buffer
from ...views import BufferView
from ...compress import Method
from ...utils.byteorder import LittleEndian


@fieldwise_init
struct BodyCompression(Equatable, ImplicitlyCopyable, Movable):
    """How a record batch's body buffers are compressed (`Message.fbs`'s
    `BodyCompression`, method `BUFFER`): each on its own, with one of two
    codecs, prefixed by its uncompressed length as a little-endian i64 -- `-1`
    for a buffer stored as is. An empty buffer carries no prefix."""

    var codec: Int

    comptime LZ4_FRAME = Self(0)
    comptime ZSTD = Self(1)

    @staticmethod
    def from_code(code: Int) raises CorruptError -> Self:
        """The codec a `BodyCompression` table names."""
        if code != 0 and code != 1:
            raise CorruptError(t"ipc: unknown body compression codec {code}")
        return Self(code)

    @staticmethod
    def from_name(name: String) raises InvalidError -> Self:
        """`lz4` (or `lz4_frame`) or `zstd`, in any case, as pyarrow's
        `IpcWriteOptions` names them."""
        var lower = name.lower()
        if lower == "lz4" or lower == "lz4_frame":
            return Self.LZ4_FRAME
        elif lower == "zstd":
            return Self.ZSTD
        raise InvalidError(t"ipc: unknown body compression '{name}'")

    def method(self) -> Method:
        """The method this codec's buffers are compressed with."""
        if self == Self.LZ4_FRAME:
            return Method.LZ4_FRAME
        return Method.ZSTD

    def compress(
        self,
        src: Span[mut=False, UInt8, _],
        mut out: List[UInt8],
        native: Bool = True,
    ) raises:
        """Append `src`, compressed behind its length, to `out` -- always
        compressed, as Arrow C++ writes it unless asked for a minimum saving
        -- or nothing for an empty `src`. Without `native`, liblz4 or
        libzstd compresses it."""
        if len(src) > 0:
            LittleEndian.append[DType.int64](out, Int64(len(src)))
            self.method().compress[Method.LZ4_FRAME, Method.ZSTD](
                native, src, out
            )

    def decompress(
        self, src: Span[UInt8, _], native: Bool = True
    ) raises DynError -> Buffer[mut=False]:
        """The buffer whose compressed form is `src` -- empty for an empty
        `src`, as `compress` writes an empty buffer. Without `native`,
        liblz4 or libzstd decompresses it."""
        var n = 0
        var payload = src
        var raw = False
        if len(src) > 0:
            n = Int(LittleEndian.checked[DType.int64](src, 0))
            payload = src[8:]
            raw = n == -1
        if raw:
            n = len(payload)
        else:
            # Checked before allocating: the length comes from the file.
            var most = self.method().max_decompressed_length(len(payload))
            if n < 0 or n > most:
                raise CorruptError(
                    t"ipc: a compressed buffer of {n} bytes in {len(payload)}"
                )
        var buf = Buffer.alloc_uninit[DType.uint8](n)
        var dst = buf.view(0, n)
        if raw:
            dst.copy_from(BufferView(payload), n)
        elif n > 0:
            # A length of 0 is an empty buffer, whatever follows it: Arrow
            # Java writes one with nothing after it.
            self.method().decompress[Method.LZ4_FRAME, Method.ZSTD](
                native, payload, dst.as_span()
            )
        return buf^.to_immutable()
