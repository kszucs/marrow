"""Naming a location, and configuring the service that holds it.

`Uri` is the one place that knows `s3://bucket/key` means "the `s3` service,
bucket `bucket`, object `key`" -- so adding a backend is a row in the table
below rather than a branch in every reader.

Bare paths stay bare. `data.parquet` and `/tmp/data.parquet` have no scheme and
resolve to the local, zero-copy `BufferSource` / `FileSink`; nothing about them
touches OpenDAL, which is what lets marrow work with no `libopendal_c` present.
"""

from std.collections.string import Codepoint
from std.os import getenv


struct Uri(Copyable, Movable, Writable):
    """A parsed location: `scheme://authority/path?query`.

    An empty `scheme` means a bare filesystem path, which is not the same thing
    as `file://` -- both are local, but only the bare form is guaranteed to
    round-trip a path containing characters a URI would escape.
    """

    var scheme: String
    """Lowercased, empty for a bare path."""
    var authority: String
    """Bucket, container, or host -- whatever sits between `//` and the next
    `/`."""
    var path: String
    """The object key. No leading `/`, because that is how every object store
    names things and how OpenDAL expects to be asked."""
    var query: Dict[String, String]
    """`?region=us-east-1` -- service options carried in the location itself."""

    def __init__(
        out self,
        var scheme: String,
        var authority: String,
        var path: String,
        var query: Dict[String, String] = {},
    ):
        self.scheme = scheme^
        self.authority = authority^
        self.path = path^
        self.query = query^

    def __init__(out self, *, copy: Self):
        self.scheme = copy.scheme
        self.authority = copy.authority
        self.path = copy.path
        self.query = copy.query.copy()

    @staticmethod
    def parse(uri: String) raises -> Self:
        """Split `uri` into its parts.

        A scheme is recognised only before `://`, never a bare `:`. That is
        what keeps `report:2026.parquet` a filename instead of a `report`
        service, and it is the reason this does not accept `mailto:`-style
        opaque URIs -- no storage backend uses one.
        """
        if uri == "":
            raise Error("uri: empty location")

        var rest = uri
        var scheme = String()
        var authority = String()

        var sep = uri.find("://")
        if sep > 0 and _is_scheme(uri[byte=:sep]):
            scheme = String(uri[byte=:sep]).lower()
            rest = String(uri[byte = sep + 3 :])
            var slash = rest.find("/")
            if slash < 0:
                authority = rest
                rest = String()
            else:
                authority = String(rest[byte=:slash])
                var tail = String(rest[byte = slash + 1 :])
                rest = tail^

        var query = Dict[String, String]()
        var q = rest.find("?")
        if q >= 0:
            var qs = String(rest[byte = q + 1 :])
            var head = String(rest[byte=:q])
            rest = head^
            for pair in qs.split("&"):
                if pair == "":
                    continue
                var eq = pair.find("=")
                if eq < 0:
                    query[String(pair)] = String()
                else:
                    query[String(pair[byte=:eq])] = String(
                        pair[byte = eq + 1 :]
                    )

        # The path is decoded here and nowhere else -- see `_percent_decode`.
        # A bare path (no scheme) is a filename, not a URI, so it is taken
        # literally: a file really can be called `a%20b.parquet`.
        var path = String(rest) if scheme == "" else _percent_decode(rest)
        return Self(scheme^, authority^, path^, query^)

    def is_local(self) -> Bool:
        """Whether this resolves to the local filesystem without OpenDAL.

        `file://` counts: it names the same bytes a bare path does, and routing
        it through an object-store binding would make a local read depend on a
        library that may not be installed."""
        return self.scheme == "" or self.scheme == "file"

    def local_path(self) -> String:
        """The filesystem path, for a local URI.

        `file:///tmp/x` loses the empty authority and keeps the leading slash;
        a bare path is returned unchanged.
        """
        if self.scheme == "":
            return self.path
        return String("/", self.path)

    def object_key(self) raises -> String:
        """The path to hand the storage service, which is not always `path`.

        For most schemes the two are the same: `s3://bucket/a/b.parquet` has
        bucket `bucket` and key `a/b.parquet`. `hf://` is the exception --
        the repo and revision are *configuration*, so they come out of the
        path and into the operator's options, and what is left is the key.
        """
        if self.scheme == "hf":
            return _hf_split(self).path
        return self.path

    def service(self) raises -> String:
        """The OpenDAL service name for this scheme.

        Aliases collapse here -- `gs` and `gcs` are one service, `az`/`abfs`/
        `azblob` another -- so the table lives in one place rather than in
        every caller.
        """
        var s = self.scheme
        if s == "fs":
            # Deliberately *not* `is_local`: `fs://` is how a caller opts into
            # reading a local file through OpenDAL rather than a memory map --
            # useful for exercising the binding, and the only way to apply a
            # service option such as `root` to a local path. A bare path and
            # `file://` stay on the zero-copy route.
            return String("fs")
        elif s == "s3":
            return String("s3")
        elif s == "gs" or s == "gcs":
            return String("gcs")
        elif s == "az" or s == "abfs" or s == "azblob":
            return String("azblob")
        elif s == "http" or s == "https":
            return String("http")
        elif s == "hf":
            return String("hf")
        elif self.is_local():
            # Reachable only by asking a local URI for a remote service, which
            # is a caller bug rather than an unknown scheme -- say which.
            raise Error(
                "uri: '",
                self,
                (
                    "' is local; it is read through a memory map, not a storage"
                    " service"
                ),
            )
        raise Error(
            "uri: no storage backend for scheme '",
            s,
            "://' (known: file, fs, s3, gs, gcs, az, abfs, azblob, hf, https)",
        )

    def write_to[W: Writer](self, mut writer: W):
        if self.scheme != "":
            writer.write(self.scheme, "://", self.authority, "/")
        writer.write(self.path)


def _percent_decode(s: StringSlice) raises -> String:
    """Undo the percent-encoding of a URI path component.

    **OpenDAL percent-encodes the path itself**, so what it takes is the
    *decoded* object key. Handing it the raw URI text double-encodes every
    escape: a `%20` becomes `%2520` and the read targets an object whose name
    contains a literal "%20". This is the one place that can know the
    difference, because after this the path is just a key.

    An incomplete or non-hex escape is left alone rather than rejected --
    a bare `%` is legal in an object key on every service marrow speaks to,
    and refusing it would make a nameable object unreadable.
    """
    var b = s.as_bytes()
    var out = List[UInt8](capacity=len(b))
    var i = 0
    while i < len(b):
        # Bytes, not `chr()`. `chr(0xC3)` is the *codepoint* U+00C3, which
        # encodes as two UTF-8 bytes, so decoding `%C3%A9` through it yields
        # four bytes of mojibake instead of the two that spell "é" -- and the
        # read goes to a different object, or nowhere.
        var hi = _hex(b[i + 1]) if i + 2 < len(b) else -1
        var lo = _hex(b[i + 2]) if i + 2 < len(b) else -1
        if b[i] == UInt8(ord("%")) and hi >= 0 and lo >= 0:
            out.append(UInt8(hi * 16 + lo))
            i += 3
        else:
            out.append(b[i])
            i += 1
    # Non-validating: an object key is a byte string on every service marrow
    # speaks to, and refusing one that is not UTF-8 would make a nameable
    # object unreadable.
    return String(StringSlice(unsafe_from_utf8=Span(out)))


def _hex(c: UInt8) -> Int:
    """The value of one hex digit, or -1 if it is not one."""
    var v = Int(c)
    if v >= ord("0") and v <= ord("9"):
        return v - ord("0")
    if v >= ord("a") and v <= ord("f"):
        return v - ord("a") + 10
    if v >= ord("A") and v <= ord("F"):
        return v - ord("A") + 10
    return -1


@fieldwise_init
struct _HfParts(Movable):
    """A Hugging Face URI split into what the `hf` service wants."""

    var repo_type: String
    var repo_id: String
    var revision: String
    var path: String


def _hf_split(uri: Uri) raises -> _HfParts:
    """Split a Hugging Face URI into repo type, repo id, revision and key.

    The spelling is `huggingface_hub`'s `HfFileSystem`, which is what users
    already have in their fingers:

        hf://datasets/openai/gsm8k/main/train-00000-of-00001.parquet
        hf://datasets/openai/gsm8k@v1.0/main/train-00000-of-00001.parquet
        hf://meta-llama/Llama-3-8B/config.json          (a model, no prefix)

    The authority is `datasets`, `spaces`, or the owner half of a model's
    `owner/name`; a repo id is always two segments. A `@` starts the revision
    and ends at the next `/`.

    **A revision containing slashes is therefore not expressible**, so the
    Hub's auto-converted `refs/convert/parquet` branch cannot be named this
    way: `@refs/convert/parquet/...` reads as revision `refs` and the rest as
    a path, which fails at `stat` rather than silently. `huggingface_hub`
    writes that revision `%2F`-escaped, and marrow cannot: the path is
    percent-decoded before it gets here, because every other backend needs
    the decoded key. Use a branch name or a commit SHA, which is what a
    reproducible reference wants anyway.

    One function rather than two because the options and the object key come
    out of the same split, and computing it twice is how they drift apart.
    """
    var rest = uri.path
    var owner = uri.authority
    if owner == "":
        raise Error(
            (
                "uri: hf:// needs a repo, as hf://datasets/owner/name/path or"
                " hf://owner/name/path, got '"
            ),
            uri,
            "'",
        )
    var repo_type = String("model")
    if owner == "datasets" or owner == "spaces":
        repo_type = "dataset" if owner == "datasets" else "space"
        var cut = rest.find("/")
        if cut < 0:
            raise Error(
                "uri: hf://", owner, "/ needs owner/name, got '", uri, "'"
            )
        owner = String(rest[byte=:cut])
        var after_owner = String(rest[byte = cut + 1 :])
        rest = after_owner^

    var cut = rest.find("/")
    if cut < 0:
        raise Error(
            "uri: hf:// needs owner/name and a file path, got '", uri, "'"
        )
    var name = String(rest[byte=:cut])
    var after_name = String(rest[byte = cut + 1 :])
    rest = after_name^

    # Only the repo-name segment can carry a revision. Searching the rest of
    # the path for `@` has no true-positive case -- a revision always sits in
    # the second segment -- and it silently mangles a filename that contains
    # one: `hf://datasets/o/n/data@2/f.parquet` would read as revision "2" and
    # key "dataf.parquet", a file that does not exist, against a revision that
    # does not exist. OpenDAL's own hf service guards this the same way, by
    # looking for `@` only in the first two segments (services/hf/src/core.rs).
    var revision = String()
    var at = name.find("@")
    if at >= 0:
        revision = String(name[byte = at + 1 :])
        var bare = String(name[byte=:at])
        name = bare^

    if rest == "":
        raise Error(
            "uri: hf:// needs a file path after the repo, got '", uri, "'"
        )

    return _HfParts(repo_type, String(owner, "/", name), revision^, rest^)


def _is_scheme(s: StringSlice) -> Bool:
    """A scheme is a letter followed by letters, digits, `+`, `-` or `.`.

    Checked rather than assumed so a Windows-style or colon-bearing path is not
    mistaken for one.
    """
    var b = s.as_bytes()
    if len(b) == 0 or not _alpha(b[0]):
        return False
    for i in range(1, len(b)):
        var c = Codepoint(b[i])
        # RFC 3986: ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )
        if not (
            _alpha(b[i])
            or c.is_ascii_digit()
            or c == Codepoint("+")
            or c == Codepoint("-")
            or c == Codepoint(".")
        ):
            return False
    return True


def _alpha(b: UInt8) -> Bool:
    var c = Codepoint(b)
    return c.is_ascii_upper() or c.is_ascii_lower()


def _default(mut kv: Dict[String, String], key: String, value: String):
    """Set `key` only if unset -- how a lower-precedence source applies."""
    if value != "" and key not in kv:
        kv[key] = value


def _fill_from_env(mut kv: Dict[String, String], service: String):
    """Fill unset keys from the environment, per service.

    Only the variables a user would expect to already have exported. This
    deliberately does *not* reproduce every provider's credential chain --
    OpenDAL does that itself once the operator exists; these are the few keys
    it has to be told.

    A free function rather than a method: filling in place is what keeps
    `resolve` from having to move a `Dict` out of the middle of a live
    `StorageOptions`, which the compiler rejects.
    """
    if service == "s3":
        var region = getenv("AWS_REGION")
        if region == "":
            region = getenv("AWS_DEFAULT_REGION")
        _default(kv, "region", region)
        _default(kv, "access_key_id", getenv("AWS_ACCESS_KEY_ID"))
        _default(kv, "secret_access_key", getenv("AWS_SECRET_ACCESS_KEY"))
        _default(kv, "session_token", getenv("AWS_SESSION_TOKEN"))
        _default(kv, "endpoint", getenv("AWS_ENDPOINT_URL"))
    elif service == "gcs":
        _default(
            kv, "credential_path", getenv("GOOGLE_APPLICATION_CREDENTIALS")
        )
    elif service == "azblob":
        _default(kv, "account_name", getenv("AZURE_STORAGE_ACCOUNT_NAME"))
        _default(kv, "account_key", getenv("AZURE_STORAGE_ACCOUNT_KEY"))


struct StorageOptions(Copyable, Movable):
    """Service configuration as a string map.

    A map rather than one struct per backend because that is exactly what
    `opendal_operator_new(scheme, options)` takes: adding a service adds no
    type here. The keys are OpenDAL's own, documented per service at
    <https://opendal.apache.org/docs/category/services/>.

    Precedence, highest first: what the caller passed, then the URI's query
    string, then the environment. A credential in the environment is therefore
    a default a call can override, which is the order every other tool uses.
    """

    var _kv: Dict[String, String]

    def __init__(out self, var kv: Dict[String, String] = {}):
        self._kv = kv^

    def __init__(out self, *, copy: Self):
        self._kv = copy._kv.copy()

    def resolve(self, uri: Uri) raises -> Dict[String, String]:
        """The full option map for `uri`: the caller's keys, then the query
        string, then the environment, then whatever the scheme itself implies.

        The scheme-implied keys come last because they are facts rather than
        defaults: the bucket in `s3://bucket/key` *is* the bucket, and a caller
        who also passed one meant the URI's.
        """
        var out = self._kv.copy()
        for entry in uri.query.items():
            if entry.key not in out:
                out[entry.key] = entry.value

        var service = uri.service()
        _fill_from_env(out, service)

        if service == "fs":
            # `fs:///tmp/x` -> root `/`, path `tmp/x`, so it needs no options.
            # An explicit `root` still wins.
            if uri.authority != "":
                # `fs://tmp/x` parses as authority `tmp`, path `x`, and a
                # filesystem has nothing to do with an authority -- so
                # ignoring it would silently read `/x`, a different file.
                # The spelling that means what it looks like is the
                # three-slash one.
                raise Error(
                    "fs:// takes no host, and '",
                    uri.authority,
                    "' would be dropped rather than treated as a directory;",
                    " write fs:///",
                    uri.authority,
                    "/... instead",
                )
            _default(out, "root", "/")
        elif service == "s3" or service == "gcs":
            out["bucket"] = uri.authority
        elif service == "azblob":
            out["container"] = uri.authority
        elif service == "http":
            out["endpoint"] = String(uri.scheme, "://", uri.authority)
        elif service == "hf":
            var hf = _hf_split(uri)
            out["repo_type"] = hf.repo_type
            out["repo_id"] = hf.repo_id
            if hf.revision != "":
                _default(out, "revision", hf.revision)
            _default(out, "token", getenv("HF_TOKEN"))
        return out^
