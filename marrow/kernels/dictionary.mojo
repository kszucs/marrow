# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Dictionary encoding — every distinct key to a dense code, exactly.

`DictionaryEncoder` gives each distinct key of a stream of batches a code —
0, 1, 2, ... in first-seen order — and keeps the key of every code. Codes and
keys together are a dictionary; the same map is what a group-by calls its
group ids and what ``is_in`` looks values up in. It serves a ``GROUP BY``,
``DISTINCT``, the set operations, the join, ``is_in`` and ``count_distinct``.

It is three pieces from the layers below plus what only it knows:

- **placement** — ``HashIndex`` (``hashtable.mojo``) turns hashes into codes,
  on one table or on radix-partitioned ones;
- **key identity** — ``HashKernel`` hashes a key, and a key equal to another
  is one by ``KeyCompare``'s rules (``hashing.mojo``);
- **its own**: the placement of rows whose hash collided.

**A key is a struct of its columns.** A batch of keys is one ``StructArray``
whatever the number and types of its columns, so it hashes and compares in one
call, and the key of every code is a ``ChunkedArray`` of such structs.

**A hash only nominates a code.** Which nominations need checking depends on
the key and is decided once, from its type:

- **One fixed-width column of at most 64 bits** — integers, floats, bool,
  temporal, 32/64-bit decimals — is hashed with ``Fmix64``, a bijection of the
  value's 64-bit word: equal hashes are equal keys, so the nomination is the
  answer. NULL has no word and hashes to the sentinel one value shares, so once
  a NULL has been seen, the rows carrying that hash are checked.
- **Any other key** is checked a column at a time against the key of its
  nominated code, and the rows that differ — true collisions — are placed one
  at a time by walking every code their hash holds.

**A single dictionary-encoded column is encoded by its entries**: the entries
a batch uses are encoded as plain values, and each row takes its entry's code
by index. Hashing and comparison then run once per distinct value rather than
once per row. A dictionary column among several key columns is decoded.

Keys stay columnar throughout; nothing is serialised into rows.
"""

from ..arrays import (
    ChunkedArray,
    DictionaryArray,
    DynArray,
    Int32Array,
    PrimitiveArray,
    StructArray,
    UInt64Array,
)
from ..buffers import Bitmap, Buffer
from ..builders import DynBuilder, Int32Builder, arange, array
from ..dtypes import (
    DynType,
    Field,
    IntegerType,
    Int32Type,
    int32,
    struct_,
    uint64,
)
from ..errors import InvalidError, TypeError
from ..execution import ExecContext
from ..views import apply
from .cast import decode_dictionary, normalized_indices
from .filter import FilterKernel, TakeKernel
from .hashing import HashKernel, KeyCompare, NULL_HASH_SENTINEL
from .hashtable import HashIndex, Placement
from ..utils import Fmix64, Hasher, KeyHash


# ---------------------------------------------------------------------------
# DictionaryEncoder — distinct keys to dense codes
# ---------------------------------------------------------------------------


def has_code[W: Int](code: SIMD[int32.native, W]) -> SIMD[DType.bool, W]:
    """Whether a key has a code — ``lookup`` answers ``-1`` for one that has
    none."""
    return code.ge(0)


struct DictionaryEncoder[Hash: Hasher = KeyHash](Movable, Sized):
    """Distinct keys to dense codes, stable across batches, over key columns
    of any types — see the module docstring.

    The key of every code is kept as chunks of key structs, one per batch that
    brought new codes. The last two merge whenever the older is less than
    twice the newer, so there are O(log n) chunks and each key is copied
    O(log n) times over a stream.

    ``Hash`` hashes every key that is not injective — tests pass
    ``TruncatedHash64`` to make those keys collide. An injective key always
    hashes with ``Fmix64``: its exactness is the bijection, not the
    comparison. The default ``KeyHash`` is ``Fmix64`` as well.
    """

    var _types: List[DynType]
    """The key columns' declared types; ``values`` emits in them."""
    var _by_entries: Bool
    """One dictionary-encoded column, encoded by its entries."""
    var _keys: ChunkedArray
    """The key of every code, by code, as structs of the plain key columns."""
    var _index: HashIndex
    var _seen_null: Bool
    """Whether a NULL key has been encoded — until then no injective hash is
    ambiguous."""
    var _ctx: ExecContext

    def __init__(
        out self, var types: List[DynType], var ctx: ExecContext = ExecContext()
    ):
        """No keys yet, of ``types``, one per key column. ``ctx`` selects
        radix placement over the single table, and stripes the hashing."""
        var plain = List[DynType](capacity=len(types))
        for t in types:
            plain.append(
                t.as_dictionary()
                .value_type()
                .copy() if t.is_dictionary() else t.copy()
            )
        self._by_entries = len(types) == 1 and types[0].is_dictionary()
        self._types = types^
        self._keys = ChunkedArray(Self._struct_of(plain), List[DynArray]())
        self._index = HashIndex(ctx.copy())
        self._seen_null = False
        self._ctx = ctx^

    def __len__(self) -> Int:
        """How many distinct keys have been encoded."""
        return len(self._index)

    def is_partitioned(self) -> Bool:
        """Whether codes are placed on radix-partitioned tables."""
        return self._index.is_partitioned()

    def reserve(mut self, n: Int):
        """Expect ``n`` new keys from the next ``encode``, so its tables are
        sized once instead of grown — a join's build side, whose rows are
        usually its keys."""
        self._index.reserve(n)

    def encode(mut self, columns: List[DynArray]) raises -> Placement:
        """The code of every row's key (``ids``), giving a new code to each key
        not seen before. Codes are dense and stable — a key seen in an earlier
        batch keeps its code — and ``firsts[j]`` is the row that introduced
        code ``len_before + j``, in code order.

        ``columns`` are the key columns, one per declared type, checked
        against them first."""
        self._check_types(columns)
        if self._by_entries and len(columns[0]) > 0:
            return self._encode_entries(columns[0].as_dictionary())
        return self._encode(self._decoded(columns))

    def lookup(self, columns: List[DynArray]) raises -> Int32Array:
        """The code of every row's key, or ``-1`` where the key has none.
        Encodes nothing; ``columns`` are checked as ``encode`` checks them."""
        self._check_types(columns)
        if self._by_entries and len(columns[0]) > 0:
            return self._lookup_entries(columns[0].as_dictionary())
        return self._lookup(self._decoded(columns))

    def _decoded(self, columns: List[DynArray]) raises -> StructArray:
        """``columns`` as a key batch, every dictionary column replaced by
        its values."""
        # The list is borrowed and copied here, not owned by `encode`: owning
        # or copying an erased list there inlines `DynArray`'s copy and
        # destructor for every array type into each caller.
        var out = List[DynArray](capacity=len(columns))
        for ref column in columns:
            if column.dtype().is_dictionary():
                out.append(
                    decode_dictionary(
                        column.as_dictionary(), self._ctx.for_batch(len(column))
                    )
                )
            else:
                out.append(column.copy())
        return Self._batch(out^)

    def values(mut self) raises -> List[DynArray]:
        """The key of every code, one array per key column, in the columns'
        declared types — a dictionary column as a dictionary array whose entry
        ``i`` is code ``i``'s value."""
        var columns = self._plain_values()
        var out = List[DynArray](capacity=len(columns))
        for k in range(len(columns)):
            if self._types[k].is_dictionary():
                out.append(
                    Self._as_dictionary(columns[k].copy(), self._types[k])
                )
            else:
                out.append(columns[k].copy())
        return out^

    @staticmethod
    def _as_dictionary(var values: DynArray, dtype: DynType) raises -> DynArray:
        """``values`` as a dictionary array of type ``dtype`` whose entry ``i``
        is at index ``i``, under a NULL index where the value is NULL — a key
        column held decoded, emitted in the type it was declared with. The
        values are distinct keys already, so nothing is deduplicated.

        Raises when there are more keys than the index type numbers: batches
        that each fit `int8` indices can bring more distinct keys between them
        than one `int8` dictionary holds, and a wrapped index names the wrong
        key."""
        ref dt = dtype.as_dictionary()
        var index_type = dt.index_type().copy()
        var n = len(values)
        var nulls = values.null_count()
        # Read off the values once, here: inside `indices` every query of the
        # erased column is inlined into each of the eight index-type arms.
        var validity: Optional[Bitmap[]] = None
        if nulls != 0:
            var valid = Bitmap.alloc_zeroed(n)
            for i in range(n):
                if values.is_valid(i):
                    valid.set(i)
            validity = valid^.to_immutable(length=n)

        def indices[T: IntegerType](d: T) raises {imm} -> DynArray:
            if n > 0 and Int(Scalar[T.native](n - 1)) != n - 1:
                raise InvalidError(
                    t"dictionary encoder: {n} distinct keys do not fit"
                    t" {index_type} indices"
                )
            var buf = Buffer.alloc_uninit[T.native](n)
            var out = buf.view[T.native](0, n)
            for i in range(n):
                out.store[1](i, Scalar[T.native](i))
            return PrimitiveArray[T](
                dtype=d,
                length=n,
                nulls=nulls,
                offset=0,
                bitmap=validity.copy(),
                buffer=buf^.to_immutable(),
            )

        var index = index_type.dispatch_integer(indices)
        return DictionaryArray.from_arrays(index^, values^, ordered=dt.ordered)

    def _check_types(self, columns: List[DynArray]) raises:
        """Refuse keys that are not the declared columns — a hash cannot tell
        an ``int32`` 5 from an ``int64`` 5, so a mismatch would not fail on its
        own."""
        if len(columns) == 0 or len(columns) != len(self._types):
            raise InvalidError(
                t"dictionary encoder: {len(columns)} key columns for"
                t" {len(self._types)} declared"
            )
        for k in range(len(columns)):
            if columns[k].dtype() != self._types[k]:
                raise TypeError(
                    t"dictionary encoder: key column {k} is"
                    t" {columns[k].dtype()}, declared {self._types[k]}"
                )

    def _encode_entries(mut self, key: DictionaryArray) raises -> Placement:
        """Encode the entries ``key``'s rows use, then give every row its
        entry's code. An unused entry is never encoded — it would become a key
        no row has — and a NULL index is encoded as one NULL entry."""
        var used = _UsedEntries(key)
        var entries = self._encode(Self._batch(used.values(key)))
        return Placement(
            ids=used.codes_of_rows(entries.ids),
            firsts=used.first_rows(entries.firsts),
        )

    def _lookup_entries(self, key: DictionaryArray) raises -> Int32Array:
        var used = _UsedEntries(key)
        var entry_codes = self._lookup(Self._batch(used.values(key)))
        return used.codes_of_rows(entry_codes)

    @staticmethod
    def _batch(var columns: List[DynArray]) -> StructArray:
        """``columns``, one per key column, as a batch of keys: a struct whose
        fields are named by position, since a key's identity is its values
        alone."""
        var types = List[DynType](capacity=len(columns))
        for ref column in columns:
            types.append(column.dtype())
        var n = len(columns[0]) if len(columns) > 0 else 0
        return StructArray(
            dtype=Self._struct_of(types),
            length=n,
            nulls=0,
            offset=0,
            bitmap=None,
            children=columns^,
        )

    @staticmethod
    def _struct_of(types: List[DynType]) -> DynType:
        var fields = List[Field](capacity=len(types))
        for k in range(len(types)):
            fields.append(Field(String(k), types[k].copy()))
        return struct_(fields^)

    # --- exact encoding of plain key structs --------------------------------

    def _encode(mut self, keys: StructArray) raises -> Placement:
        """``encode`` over plain keys."""
        if len(keys) == 0:
            return Placement(
                ids=Int32Array.empty(int32), firsts=Int32Array.empty(int32)
            )
        var hashes = self._hash(keys)
        var placed = self._index.insert(hashes)
        var codes = placed.ids.copy()
        var firsts = placed.firsts.copy()
        if len(firsts) > 0:
            self._append(keys, firsts, not self._index.is_partitioned())
        if self._injective() and Self._has_nulls(keys):
            self._seen_null = True
        var wrong = self._wrong_nominations(codes, keys, hashes)
        if len(wrong) == 0:
            return Placement(ids=codes^, firsts=firsts^)
        # Place each true collision among every code its hash holds; a code it
        # creates is kept before the next row is placed, and comes after every
        # code the batch's hashes created.
        var right = List[Int32](capacity=len(wrong))
        var created = List[Int32]()
        for j in range(len(wrong)):
            var row = Int(wrong.unsafe_get(j))
            var h = UInt64(hashes.unsafe_get(row))
            var code = self._code_of(keys, row, h)
            if code < 0:
                code = self._index.insert_new(h)
                self._append(keys, array([row], int32), False)
                created.append(Int32(row))
            right.append(Int32(code))
        if len(created) > 0:
            var all = Buffer.alloc_uninit[int32.native](
                len(firsts) + len(created)
            )
            all.extend(firsts.values(), 0, len(firsts))
            for j in range(len(created)):
                all.unsafe_set[int32.native](len(firsts) + j, created[j])
            firsts = Int32Array(
                length=len(firsts) + len(created),
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=all^.to_immutable(),
            )
        return Placement(
            ids=Self._replaced(codes, wrong, right), firsts=firsts^
        )

    def _lookup(self, keys: StructArray) raises -> Int32Array:
        """``lookup`` over plain keys."""
        if len(keys) == 0:
            return Int32Array.empty(int32)
        var hashes = self._hash(keys)
        var codes = self._index.find(hashes)
        var wrong = self._wrong_nominations(codes, keys, hashes)
        if len(wrong) == 0:
            return codes^
        var right = List[Int32](capacity=len(wrong))
        for j in range(len(wrong)):
            var row = Int(wrong.unsafe_get(j))
            right.append(
                Int32(self._code_of(keys, row, UInt64(hashes.unsafe_get(row))))
            )
        return Self._replaced(codes, wrong, right)

    def _plain_values(mut self) raises -> List[DynArray]:
        """The key of every code, one plain array per key column, by code."""
        if len(self._keys.chunks) == 0:
            var empty = DynBuilder(self._keys.dtype)
            return empty.finish().as_struct().flatten()
        while len(self._keys.chunks) >= 2:
            self._merge_last_two()
        return self._keys.chunks[0].as_struct().flatten()

    # --- the keys -----------------------------------------------------------

    def _injective(self) -> Bool:
        """Whether equal hashes are equal keys — one fixed-width column of at
        most 64 bits — but for NULL, which hashes to the sentinel one value
        shares (see the module docstring)."""
        ref fields = self._keys.dtype.as_struct().fields
        if len(fields) != 1:
            return False
        ref t = fields[0].dtype
        return t.is_bool() or (t.is_primitive() and t.byte_width() <= 8)

    @staticmethod
    def _has_nulls(keys: StructArray) raises -> Bool:
        """Whether a key of the batch is NULL. Asked only of an injective
        key, which is one column; the struct around it is never NULL."""
        return keys.field(0).null_count() != 0

    def _hash(self, keys: StructArray) raises -> UInt64Array:
        """Every row's hash: ``Fmix64``'s for an injective key, ``Hash``'s
        for any other."""
        var ctx = self._ctx.for_batch(len(keys))
        if self._injective():
            return HashKernel[Fmix64].apply(keys, ctx)
        return HashKernel[Self.Hash].apply(keys, ctx)

    def _append(
        mut self, keys: StructArray, rows: Int32Array, in_row_order: Bool
    ) raises:
        """Keep the keys at ``rows`` as the keys of the next codes.

        ``in_row_order`` when the codes were handed out as rows introduced
        them — one table's numbering. When those rows are then the batch's
        first ones, the batch itself, sliced, is the new chunk, copied not at
        all: a value set and a join's build side on one table take this path.
        """
        var m = len(rows)
        if (
            in_row_order
            and Int(rows.unsafe_get(0)) == 0
            and Int(rows.unsafe_get(m - 1)) == m - 1
        ):
            self._keys.append(keys.slice(0, m))
        else:
            self._keys.append(TakeKernel.apply(keys, rows))
        while len(self._keys.chunks) >= 2:
            var c = len(self._keys.chunks)
            if len(self._keys.chunks[c - 2]) >= 2 * len(
                self._keys.chunks[c - 1]
            ):
                break
            self._merge_last_two()

    def _merge_last_two(mut self) raises:
        """The last two chunks as one, in place."""
        ref chunks = self._keys.chunks
        var c = len(chunks)
        var merged = DynBuilder(self._keys.dtype)
        merged.extend(chunks[c - 2])
        merged.extend(chunks[c - 1])
        chunks[c - 2] = merged.finish()
        chunks.shrink(c - 1)

    def _compare(
        self,
        codes: Int32Array,
        keys: StructArray,
        rows: Int32Array,
        mut equal: Bitmap[mut=True],
        ctx: ExecContext,
    ) raises:
        """Clear ``equal[i]`` where the key at ``rows[i]`` is not the key of
        ``codes[i]``. Rows already clear are skipped.

        Each row is compared against the chunk holding its code: the rows are
        sorted by chunk in one counting pass, and each chunk's run is one
        ``KeyCompare``."""
        ref chunks = self._keys.chunks
        var num_chunks = len(chunks)
        if num_chunks == 1:
            KeyCompare.apply(
                chunks[0].as_struct(), codes, keys, rows, equal, ctx
            )
            return
        # The code of each chunk's first key.
        var starts = List[Int](capacity=num_chunks)
        var start = 0
        for ref chunk in chunks:
            starts.append(start)
            start += len(chunk)
        var n = len(codes)
        var run_start = List[Int](length=num_chunks + 1, fill=0)
        for i in range(n):
            if equal.test(i):
                var c = Self._chunk_of(starts, Int(codes.unsafe_get(i)))
                run_start[c + 1] += 1
        for c in range(num_chunks):
            run_start[c + 1] += run_start[c]
        var total = run_start[num_chunks]
        var local = Buffer.alloc_uninit[int32.native](max(total, 1))
        var at = Buffer.alloc_uninit[int32.native](max(total, 1))
        var position = Buffer.alloc_uninit[int32.native](max(total, 1))
        var next = run_start.copy()
        for i in range(n):
            if equal.test(i):
                var code = Int(codes.unsafe_get(i))
                var c = Self._chunk_of(starts, code)
                var p = next[c]
                next[c] += 1
                local.unsafe_set[int32.native](p, Int32(code - starts[c]))
                at.unsafe_set[int32.native](p, rows.unsafe_get(i))
                position.unsafe_set[int32.native](p, Int32(i))
        var local_codes = Int32Array(
            length=total,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=local^.to_immutable(),
        )
        var at_rows = Int32Array(
            length=total,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=at^.to_immutable(),
        )
        for c in range(num_chunks):
            var first = run_start[c]
            var m = run_start[c + 1] - first
            if m == 0:
                continue
            var run = Bitmap.alloc_zeroed(m)
            run.set_range(0, m, True)
            KeyCompare.apply(
                chunks[c].as_struct(),
                local_codes.slice(first, m),
                keys,
                at_rows.slice(first, m),
                run,
                ctx,
            )
            if run.unset_count() != 0:
                for j in range(m):
                    if not run.test(j):
                        equal.clear(
                            Int(position.unsafe_get[int32.native](first + j))
                        )

    @staticmethod
    def _chunk_of(starts: List[Int], code: Int) -> Int:
        """The chunk holding ``code``, by the code of each chunk's first
        key."""
        var lo = 0
        var hi = len(starts) - 1
        while lo < hi:
            var mid = (lo + hi + 1) // 2
            if starts[mid] <= code:
                lo = mid
            else:
                hi = mid - 1
        return lo

    # --- verification -------------------------------------------------------

    def _wrong_nominations(
        self, codes: Int32Array, keys: StructArray, hashes: UInt64Array
    ) raises -> Int32Array:
        """The rows whose nominated code is not their key's — true hash
        collisions. A row nominated nothing (``-1``) is not among them.

        An injective key can be wrong only where it holds the NULL sentinel,
        and only once a NULL has been seen, in this batch or an earlier one.
        Any other key is compared with the key of its nominated code."""
        var n = len(codes)
        var injective = self._injective()
        if injective and not self._seen_null and not Self._has_nulls(keys):
            return Int32Array.empty(int32)
        var check = Bitmap.alloc_uninit(n)
        apply[int32.native, has_code](codes.values(), check.view())
        if injective:
            var sentinel = Bitmap.alloc_uninit(n)
            apply[uint64.native, Self._is_null_sentinel](
                hashes.values(), sentinel.view()
            )
            check = check.view() & sentinel.view()
        var rows: Int32Array
        var at: Int32Array
        if check.unset_count() == 0:
            rows = arange[Int32Type](0, n)
            at = codes.copy()
        else:
            rows = FilterKernel.apply(arange[Int32Type](0, n), check.view())
            at = FilterKernel.apply(codes, check.view())
        if len(rows) == 0:
            return rows^
        var equal = Bitmap.alloc_uninit(len(rows))
        equal.set_range(0, len(rows), True)
        self._compare(at, keys, rows, equal, self._ctx.for_batch(len(rows)))
        if equal.unset_count() == 0:
            return Int32Array.empty(int32)
        return FilterKernel.apply(rows, (~equal.view()).view())

    @staticmethod
    def _is_null_sentinel[
        W: Int
    ](h: SIMD[uint64.native, W]) -> SIMD[DType.bool, W]:
        """Whether a hash is the one a NULL key hashes to."""
        return h.eq(NULL_HASH_SENTINEL)

    @staticmethod
    def _replaced(
        codes: Int32Array, rows: Int32Array, right: List[Int32]
    ) raises -> Int32Array:
        """``codes`` with ``right[j]`` at ``rows[j]``."""
        var n = len(codes)
        var buf = Buffer.alloc_uninit[int32.native](max(n, 1))
        buf.extend(codes.values(), 0, n)
        for j in range(len(rows)):
            buf.unsafe_set[int32.native](Int(rows.unsafe_get(j)), right[j])
        return Int32Array(
            length=n, nulls=0, offset=0, bitmap=None, buffer=buf^.to_immutable()
        )

    def _code_of(self, keys: StructArray, row: Int, h: UInt64) raises -> Int:
        """The code whose key is the one at ``row``, among every code ``h``
        holds — or ``-1``. How a true collision is resolved, one row at a
        time, each candidate through the batch ``_compare`` with one row."""
        var at = array([row], int32)

        def same_key(code: Int) raises {imm} -> Bool:
            var equal = Bitmap.alloc_zeroed(1)
            equal.set(0)
            self._compare(
                array([code], int32), keys, at, equal, ExecContext.serial()
            )
            return equal.test(0)

        return self._index.find_one(h, same_key)


struct _UsedEntries(Movable):
    """The entries of one dictionary-encoded batch its rows actually use, and
    which of them each row uses — the two halves of encoding a dictionary
    column by its entries."""

    var _indices: Int32Array
    """The batch's indices, widened."""
    var _slot: List[Int32]
    """Entry -> its position among the used entries, or -1."""
    var _used: Int32Array
    """The used entries, in first-use order; a null for a NULL index."""
    var _null_slot: Int
    """The position of the NULL entry, or -1."""
    var _first_row: List[Int32]
    """The row that first used each used entry, by position."""

    def __init__(out self, key: DictionaryArray) raises:
        self._indices = normalized_indices(key)
        self._slot = List[Int32](length=len(key.dictionary()), fill=-1)
        self._null_slot = -1
        self._first_row = List[Int32]()
        var used = Int32Builder()
        for i in range(len(key)):
            if self._indices.is_valid(i):
                var e = Int(self._indices.unsafe_get(i))
                if self._slot[e] < 0:
                    self._slot[e] = Int32(len(used))
                    self._first_row.append(Int32(i))
                    used.append(Int32(e))
            elif self._null_slot < 0:
                self._null_slot = len(used)
                self._first_row.append(Int32(i))
                used.append_null()
        self._used = used.finish()

    def values(self, key: DictionaryArray) raises -> List[DynArray]:
        """The used entries' values, as the one key column."""
        var column = List[DynArray](capacity=1)
        column.append(TakeKernel.dispatch(key.dictionary(), self._used))
        return column^

    def first_rows(self, positions: Int32Array) raises -> Int32Array:
        """The row that first used each used entry at ``positions``."""
        var n = len(positions)
        var buf = Buffer.alloc_uninit[int32.native](max(n, 1))
        for j in range(n):
            buf.unsafe_set[int32.native](
                j, self._first_row[Int(positions.unsafe_get(j))]
            )
        return Int32Array(
            length=n, nulls=0, offset=0, bitmap=None, buffer=buf^.to_immutable()
        )

    def codes_of_rows(self, entry_codes: Int32Array) raises -> Int32Array:
        """Every row's code, from its entry's."""
        var n = len(self._indices)
        var buf = Buffer.alloc_uninit[int32.native](n)
        var out = buf.view[int32.native](0, n)
        for i in range(n):
            var s = Int(
                self._slot[Int(self._indices.unsafe_get(i))]
            ) if self._indices.is_valid(i) else self._null_slot
            out.store[1](i, entry_codes.unsafe_get(s))
        return Int32Array(
            length=n, nulls=0, offset=0, bitmap=None, buffer=buf^.to_immutable()
        )
