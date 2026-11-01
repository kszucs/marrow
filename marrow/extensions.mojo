# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Arrow's canonical extension types.

Each is an `ExtensionType`: carry it on a field with `Field.with_extension` and
read it back with `Field.extension[E]`. The names are the ones PyArrow gives
them (`pa.uuid()`, `pa.json_()`, `pa.bool8()`); see
https://arrow.apache.org/docs/format/CanonicalExtensions.html.

```mojo
var f = field("id", fixed_size_binary_(16)).with_extension(UuidType())
assert f.extension[UuidType]()
```
"""

from .dtypes import DynType, ExtensionType, fixed_size_binary_, int8, string
from .errors import InvalidError


struct UuidType(ExtensionType):
    """`arrow.uuid`: a 128-bit identifier stored as `fixed_size_binary(16)`."""

    def __init__(out self):
        pass

    @staticmethod
    def extension_name() -> String:
        return "arrow.uuid"

    def storage_type(self) -> DynType:
        return fixed_size_binary_(16)

    def serialize(self) -> String:
        return ""

    @staticmethod
    def deserialize(
        storage: DynType, serialized: String
    ) raises InvalidError -> Self:
        if storage != fixed_size_binary_(16):
            raise InvalidError(
                t"arrow.uuid: storage must be fixed_size_binary[16], got"
                t" {storage}"
            )
        elif serialized != "":
            raise InvalidError(
                t"arrow.uuid: expected empty metadata, got '{serialized}'"
            )
        return UuidType()


struct JsonType(ExtensionType):
    """`arrow.json`: JSON text stored as `string`, `large_string` or
    `string_view`, whichever `storage` holds."""

    var storage: DynType

    def __init__(out self, storage: DynType = string) raises InvalidError:
        if not _is_json_storage(storage):
            raise InvalidError(
                t"arrow.json: storage must be string, large_string or"
                t" string_view, got {storage}"
            )
        self.storage = storage.copy()

    @staticmethod
    def extension_name() -> String:
        return "arrow.json"

    def storage_type(self) -> DynType:
        return self.storage.copy()

    def serialize(self) -> String:
        return ""

    @staticmethod
    def deserialize(
        storage: DynType, serialized: String
    ) raises InvalidError -> Self:
        # The parameters are ignored, as Arrow C++ ignores them.
        return JsonType(storage)


def _is_json_storage(storage: DynType) -> Bool:
    return (
        storage.is_string()
        or storage.is_large_string()
        or storage.is_string_view()
    )


struct Bool8Type(ExtensionType):
    """`arrow.bool8`: a boolean stored as one `int8` byte, zero for false."""

    def __init__(out self):
        pass

    @staticmethod
    def extension_name() -> String:
        return "arrow.bool8"

    def storage_type(self) -> DynType:
        return int8

    def serialize(self) -> String:
        return ""

    @staticmethod
    def deserialize(
        storage: DynType, serialized: String
    ) raises InvalidError -> Self:
        if not storage.is_int8():
            raise InvalidError(
                t"arrow.bool8: storage must be int8, got {storage}"
            )
        elif serialized != "":
            raise InvalidError(
                t"arrow.bool8: expected empty metadata, got '{serialized}'"
            )
        return Bool8Type()
