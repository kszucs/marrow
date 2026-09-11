"""URI parsing and option resolution.

A parse table rather than prose: this is the one place that decides
`s3://bucket/key` means bucket `bucket`, and getting it wrong reads the wrong
object rather than failing.
"""

from std.os import getenv, setenv
from std.testing import assert_equal, assert_false, assert_true

from ..uri import StorageOptions, Uri


def test_utils_uri_bare_paths_have_no_scheme() raises:
    """A path is a path. `report:2026.parquet` must not become a `report`
    service -- a scheme is only recognised before `://`."""
    for p in [
        "data.parquet",
        "/tmp/data.parquet",
        "./a/b.arrow",
        "report:2026.parquet",
    ]:
        var u = Uri.parse(p)
        assert_equal(u.scheme, "")
        assert_true(u.is_local())
        assert_equal(u.local_path(), String(p))


def test_utils_uri_file_scheme_is_local() raises:
    """`file://` names the same bytes a bare path does, so it must not route
    through an object-store binding that may not be installed."""
    var u = Uri.parse("file:///tmp/data.parquet")
    assert_equal(u.scheme, "file")
    assert_true(u.is_local())
    assert_equal(u.local_path(), "/tmp/data.parquet")


def test_utils_uri_s3_splits_bucket_from_key() raises:
    var u = Uri.parse("s3://bucket/nested/key.parquet")
    assert_equal(u.scheme, "s3")
    assert_equal(u.authority, "bucket")
    assert_equal(u.path, "nested/key.parquet")
    assert_false(u.is_local())
    assert_equal(u.service(), "s3")


def test_utils_uri_scheme_aliases_collapse() raises:
    """`gs` and `gcs` are one service; so are `az`, `abfs` and `azblob`."""
    assert_equal(Uri.parse("gs://b/k").service(), "gcs")
    assert_equal(Uri.parse("gcs://b/k").service(), "gcs")
    assert_equal(Uri.parse("az://c/k").service(), "azblob")
    assert_equal(Uri.parse("abfs://c/k").service(), "azblob")
    assert_equal(Uri.parse("azblob://c/k").service(), "azblob")
    assert_equal(Uri.parse("https://host/k").service(), "http")


def test_utils_uri_query_string_is_parsed() raises:
    var u = Uri.parse("s3://bucket/key.parquet?region=us-east-1&anonymous=")
    assert_equal(u.path, "key.parquet")
    assert_equal(u.query["region"], "us-east-1")
    assert_equal(u.query["anonymous"], "")
    assert_true("region" not in Uri.parse("s3://bucket/key.parquet").query)


def test_utils_uri_uppercase_scheme_is_normalised() raises:
    assert_equal(Uri.parse("S3://bucket/key").scheme, "s3")


def test_utils_uri_unknown_scheme_raises_naming_it() raises:
    var msg = String()
    try:
        _ = Uri.parse("zzz://host/key").service()
    except e:
        msg = String(e)
    assert_true("zzz" in msg, String("should name the scheme, got: ", msg))


def test_utils_uri_empty_location_raises() raises:
    var raised = False
    try:
        _ = Uri.parse("")
    except:
        raised = True
    assert_true(raised)


def test_utils_uri_options_precedence() raises:
    """Caller beats query string beats environment; the URI's bucket wins
    outright, because it is a fact rather than a default."""
    _ = setenv("AWS_REGION", "env-region", True)
    _ = setenv("AWS_ACCESS_KEY_ID", "env-key", True)

    var u = Uri.parse("s3://from-uri/key?region=query-region")
    var opts = StorageOptions({"region": "caller-region", "bucket": "ignored"})
    var got = opts.resolve(u)

    assert_equal(got["region"], "caller-region", "caller should win")
    assert_equal(got["access_key_id"], "env-key", "env fills what is unset")
    assert_equal(got["bucket"], "from-uri", "the URI's bucket is the bucket")

    # With no caller key, the query string wins over the environment.
    var got2 = StorageOptions().resolve(u)
    assert_equal(got2["region"], "query-region")

    _ = setenv("AWS_REGION", "", True)
    _ = setenv("AWS_ACCESS_KEY_ID", "", True)


def test_utils_uri_http_endpoint_is_derived() raises:
    var got = StorageOptions().resolve(
        Uri.parse("https://example.com/a/b.parquet")
    )
    assert_equal(got["endpoint"], "https://example.com")


def test_utils_uri_fs_three_slash_roots_at_the_filesystem() raises:
    """`fs:///tmp/x` is the spelling that means what it looks like."""
    var u = Uri.parse("fs:///tmp/x.parquet")
    assert_equal(u.authority, "")
    assert_equal(u.path, "tmp/x.parquet")
    assert_equal(StorageOptions().resolve(u)["root"], "/")


def test_utils_uri_fs_with_a_host_raises_instead_of_dropping_it() raises:
    """`fs://tmp/x` parses `tmp` as an authority, and a filesystem has no use
    for one.

    Ignoring it would silently read `/x` — a different file, with no error —
    so this is the rare case where rejecting a plausible spelling beats
    accepting it. The message has to name the working form, or the user has
    no way to tell what it wanted.
    """
    var msg = String()
    try:
        _ = StorageOptions().resolve(Uri.parse("fs://tmp/x.parquet"))
    except e:
        msg = String(e)
    assert_true("fs://" in msg, String("should name the scheme, got: ", msg))
    assert_true("fs:///tmp" in msg, String("should show the fix, got: ", msg))


def test_utils_uri_path_is_percent_decoded_once() raises:
    """OpenDAL encodes the path itself, so what it takes is the decoded key.

    Leaving `%20` in would double-encode it to `%2520` and read an object
    whose name contains a literal "%20" — a miss, or worse, the wrong file.
    """
    assert_equal(Uri.parse("s3://b/my%20file.parquet").path, "my file.parquet")
    assert_equal(
        Uri.parse("s3://b/a%2Bb/c%3Dd.parquet").path, "a+b/c=d.parquet"
    )


def test_utils_uri_bare_path_is_not_decoded() raises:
    """A bare path is a filename, not a URI — a file really can be called
    `a%20b.parquet`, and decoding it would open a different one."""
    assert_equal(Uri.parse("/tmp/a%20b.parquet").path, "/tmp/a%20b.parquet")


def test_utils_uri_lone_percent_survives() raises:
    """An incomplete escape is left alone rather than rejected: `%` is a legal
    character in an object key, and refusing it makes a nameable object
    unreadable."""
    assert_equal(Uri.parse("s3://b/100%.parquet").path, "100%.parquet")
    assert_equal(Uri.parse("s3://b/a%zz.parquet").path, "a%zz.parquet")


def test_utils_uri_hf_dataset_splits_repo_from_key() raises:
    """The repo and revision are *configuration*; only the rest is the key."""
    var u = Uri.parse(
        "hf://datasets/openai/gsm8k/main/train-00000-of-00001.parquet"
    )
    var got = StorageOptions().resolve(u)
    assert_equal(got["repo_type"], "dataset")
    assert_equal(got["repo_id"], "openai/gsm8k")
    assert_equal(u.object_key(), "main/train-00000-of-00001.parquet")


def test_utils_uri_hf_model_needs_no_prefix() raises:
    var u = Uri.parse("hf://meta-llama/Llama-3-8B/config.json")
    var got = StorageOptions().resolve(u)
    assert_equal(got["repo_type"], "model")
    assert_equal(got["repo_id"], "meta-llama/Llama-3-8B")
    assert_equal(u.object_key(), "config.json")


def test_utils_uri_hf_revision_after_at() raises:
    var u = Uri.parse("hf://datasets/openai/gsm8k@v1.0/main/train.parquet")
    assert_equal(StorageOptions().resolve(u)["revision"], "v1.0")
    assert_equal(u.object_key(), "main/train.parquet")


def test_utils_uri_hf_needs_owner_and_name() raises:
    var msg = String()
    try:
        _ = Uri.parse("hf://datasets/gsm8k").object_key()
    except e:
        msg = String(e)
    assert_true("owner/name" in msg, String("got: ", msg))


def test_utils_uri_percent_decode_is_bytes_not_codepoints() raises:
    """`%C3%A9` is the two UTF-8 bytes of "é", not two codepoints.

    Decoding through `chr()` would emit U+00C3 U+00A9 — four bytes — and the
    read would go to a different object. This is byte-level on purpose.
    """
    var got = Uri.parse("s3://b/caf%C3%A9.parquet").path
    assert_equal(got, "café.parquet")
    assert_equal(len(got.as_bytes()), len("café.parquet".as_bytes()))


def test_utils_uri_hf_at_in_a_filename_is_not_a_revision() raises:
    """Only the repo-name segment can carry `@`.

    Searching the whole path for one has no true-positive case and mangles a
    filename that contains it: `data@2/f.parquet` used to parse as revision
    "2" and key "dataf.parquet" — a file that does not exist, against a
    revision that does not exist, with no error.
    """
    var u = Uri.parse("hf://datasets/o/n/data@2/file.parquet")
    var got = StorageOptions().resolve(u)
    assert_equal(got["repo_id"], "o/n")
    assert_true("revision" not in got, "a path @ is not a revision")
    assert_equal(u.object_key(), "data@2/file.parquet")


def test_utils_uri_hf_without_a_repo_raises() raises:
    """`hf:///o/n/f` has an empty authority, which used to build the repo id
    `/o` and put `n/f` in the key — split across two places, failing later as
    an opaque 404."""
    var msg = String()
    try:
        _ = Uri.parse("hf:///o/n/file.parquet").object_key()
    except e:
        msg = String(e)
    assert_true("needs a repo" in msg, String("got: ", msg))


def test_utils_uri_hf_trailing_slash_raises() raises:
    """`hf://datasets/o/n/` would otherwise stat the empty key."""
    var msg = String()
    try:
        _ = Uri.parse("hf://datasets/o/n/").object_key()
    except e:
        msg = String(e)
    assert_true("file path" in msg, String("got: ", msg))
