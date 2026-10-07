# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Datasets on the Hugging Face Hub: configs, splits and the files behind them.

A Hub dataset is a repository of files. Which of them make up a config's split
is said by the dataset card's `configs:` block -- patterns per split -- or,
when the card has none, inferred from file and directory names as `datasets`
infers it. Data in a format marrow cannot read, CSV say, is read from the
Parquet copy the Hub converts every public dataset into.
"""

from std.builtin.sort import sort
from std.os import getenv

from emberjson import Value as Json

from ..errors import InvalidError, KeyError, NotImplementedError
from ..io.opendal import OpenDalStore
from ..io.uri import Uri
from ..io.glob import Glob
from .files import DataFiles
from .formats import FileFormat


struct _HubJson:
    """Checked reads of the Hub's JSON. EmberJson's accessors do not check
    the type they read, so an answer of an unexpected shape is refused here
    rather than read as another type."""

    @staticmethod
    def require_object(v: Json, what: StringSlice) raises:
        if not v.is_object():
            raise InvalidError(t"datasets: the Hub's {what} is not an object")

    @staticmethod
    def require_array(v: Json, what: StringSlice) raises:
        if not v.is_array():
            raise InvalidError(t"datasets: the Hub's {what} is not a list")

    @staticmethod
    def string(v: Json, what: StringSlice) raises -> String:
        if not v.is_string():
            raise InvalidError(t"datasets: the Hub's {what} is not a string")
        return v.string()


struct HubConfig(Copyable, Movable, Writable):
    """One config of a Hub dataset: its name and its splits' patterns."""

    var name: String
    var default: Bool
    var data_files: DataFiles

    def __init__(
        out self,
        var name: String,
        default: Bool = False,
        var data_files: DataFiles = DataFiles(),
    ):
        self.name = name^
        self.default = default
        self.data_files = data_files^

    @staticmethod
    def from_card(entry: Json) raises -> Self:
        """One entry of a card's `configs:` list.

        `data_files` takes every spelling `datasets` accepts: a pattern, a
        list of them (both `train`), a `{split: patterns}` mapping, or a list
        of `{split, path}` entries whose `path` is a pattern or a list.
        """
        _HubJson.require_object(entry, "card config")
        ref o = entry.object()
        var name = String("default")
        if "config_name" in o:
            name = _HubJson.string(o["config_name"], "config_name")
        var default = False
        if "default" in o:
            if not o["default"].is_bool():
                raise InvalidError(
                    "datasets: the Hub's default is not a boolean"
                )
            default = o["default"].bool()
        var config = Self(name^, default)
        if "data_files" not in o:
            return config^
        ref files = o["data_files"]
        if files.is_object():
            for item in files.object().items():
                config.data_files.add(
                    String(item.key), Self._patterns(item.value)
                )
        elif (
            files.is_array()
            and len(files.array())
            and files.array()[0].is_object()
        ):
            for ref item in files.array():
                _HubJson.require_object(item, "data_files entry")
                ref e = item.object()
                if "path" not in e:
                    raise InvalidError(
                        "datasets: a data_files entry of the Hub has no path"
                    )
                var split = String("train")
                if "split" in e:
                    split = _HubJson.string(e["split"], "split")
                config.data_files.add(split, Self._patterns(e["path"]))
        else:
            config.data_files.add("train", Self._patterns(files))
        return config^

    @staticmethod
    def infer(files: List[String]) raises -> Self:
        """The `default` config of a repository whose card names none.

        Its data files are the repository's files in its most common data
        format, hidden paths aside. Each belongs to the first split whose
        name it spells as a delimited word -- `train`, `validation` (or
        `valid`, `dev`, `val`), `test` (or `eval`) -- and all are `train`
        when none does.
        """
        var counts = Dict[String, Int]()
        for ref f in files:
            var format = FileFormat.of(f)
            if format != FileFormat.NONE and not Self._hidden(f):
                counts[String(format)] = counts.get(String(format), 0) + 1
        var config = Self("default", default=True)
        if len(counts) == 0:
            return config^
        var best = String()
        var most = 0
        for entry in counts.items():
            if entry.value > most:
                best = entry.key.copy()
                most = entry.value
        var splits: List[StaticString] = ["train", "validation", "test"]
        var by_split = List[List[String]](length=len(splits), fill=[])
        var unnamed = List[String]()
        for ref f in files:
            if String(FileFormat.of(f)) != best or Self._hidden(f):
                continue
            # A file name is a pattern here, so its wildcards are escaped.
            var split = Self._split_of(f)
            if split < 0:
                unnamed.append(Glob.escape(f))
            else:
                by_split[split].append(Glob.escape(f))
        var named = False
        for i in range(len(splits)):
            if len(by_split[i]):
                config.data_files.add(String(splits[i]), by_split[i].copy())
                named = True
        if not named:
            config.data_files.add("train", unnamed^)
        return config^

    def write_to[W: Writer](self, mut writer: W):
        writer.write("HubConfig(", self.name, ": ", self.data_files, ")")

    @staticmethod
    def _patterns(v: Json) raises -> List[String]:
        """A pattern or a list of them."""
        var out = List[String]()
        if v.is_string():
            out.append(v.string())
        else:
            _HubJson.require_array(v, "data_files")
            for ref p in v.array():
                out.append(_HubJson.string(p, "data_files pattern"))
        return out^

    @staticmethod
    def _split_of(path: String) -> Int:
        """The index into `train`, `validation`, `test` of the split `path`
        names as a delimited word, or -1."""
        var keywords: List[List[StaticString]] = [
            ["train", "training"],
            ["validation", "valid", "dev", "val"],
            ["test", "testing", "eval", "evaluation"],
        ]
        var lower = path.lower()
        var b = lower.as_bytes()
        for split in range(len(keywords)):
            for keyword in keywords[split]:
                var start = 0
                while True:
                    var at = lower.find(keyword, start)
                    if at < 0:
                        break
                    var end = at + keyword.byte_length()
                    if (at == 0 or Self._delimits(b[at - 1])) and (
                        end < len(b) and Self._delimits(b[end])
                    ):
                        return split
                    start = at + 1
        return -1

    @staticmethod
    def _delimits(c: UInt8) -> Bool:
        """`-`, `_`, `.`, a space, `/` or a digit."""
        return (
            c == UInt8(ord("-"))
            or c == UInt8(ord("_"))
            or c == UInt8(ord("."))
            or c == UInt8(ord(" "))
            or c == UInt8(ord("/"))
            or (c >= UInt8(ord("0")) and c <= UInt8(ord("9")))
        )

    @staticmethod
    def _hidden(path: String) -> Bool:
        """Whether a segment of `path` is hidden: `.git…`, `__pycache__`."""
        for segment in path.split("/"):
            if segment.startswith(".") or segment.startswith("__"):
                return True
        return False


struct HubDataset(Movable, Writable):
    """A Hub dataset repository: its files and its configs.

    ```mojo
    var ds = HubDataset.fetch("openai/gsm8k")
    ds.config_names()               # ["main", "socratic"]
    ds.config("main").data_files    # {train: [main/train-*], test: [...]}
    ```
    """

    var repo: String
    var revision: String
    """Where the files were listed; empty for the default branch."""
    var files: List[String]
    """Every file in the repository, relative to its root."""
    var configs: List[HubConfig]

    def __init__(
        out self,
        var repo: String,
        var revision: String,
        var files: List[String],
        var configs: List[HubConfig],
    ):
        self.repo = repo^
        self.revision = revision^
        self.files = files^
        self.configs = configs^

    comptime ENDPOINT = "https://huggingface.co"

    @staticmethod
    def fetch(repo: String, revision: String = "") raises -> Self:
        """Ask the Hub about dataset `repo` (`owner/name`) at `revision` -- a
        branch, tag or commit -- or at the default branch."""
        var parts = repo.split("/")
        if len(parts) != 2 or parts[0] == "" or parts[1] == "":
            raise InvalidError(
                t"datasets: '{repo}' is neither a format marrow reads "
                t"(parquet, json, arrow) nor a Hub dataset named owner/name"
            )
        if revision.find("/") >= 0:
            raise InvalidError(
                t"datasets: revision '{revision}' holds a '/', which an "
                t"hf:// location cannot carry; name its commit instead"
            )
        var path = String("api/datasets/", repo)
        if revision != "":
            path = String(path, "/revision/", revision)
        return Self.parse(repo, revision, Self._get(path))

    @staticmethod
    def parse(repo: String, revision: String, info: List[UInt8]) raises -> Self:
        """From the Hub API's JSON for the dataset: `siblings` lists its
        files and `cardData.configs` its configs, inferred when absent."""
        var json = Json(parse_bytes=Span(info))
        _HubJson.require_object(json, "dataset info")
        ref o = json.object()
        var files = List[String]()
        if "siblings" in o:
            _HubJson.require_array(o["siblings"], "siblings")
            for ref sibling in o["siblings"].array():
                _HubJson.require_object(sibling, "sibling")
                ref entry = sibling.object()
                if "rfilename" not in entry:
                    raise InvalidError(
                        "datasets: a file of the Hub's listing has no name"
                    )
                files.append(_HubJson.string(entry["rfilename"], "rfilename"))
        var configs = List[HubConfig]()
        if "cardData" in o and o["cardData"].is_object():
            ref card = o["cardData"].object()
            if "configs" in card:
                _HubJson.require_array(card["configs"], "configs")
                for ref entry in card["configs"].array():
                    configs.append(HubConfig.from_card(entry))
        if len(configs) == 0:
            configs.append(HubConfig.infer(files))
        return Self(repo, revision, files^, configs^)

    def config_names(self) -> List[String]:
        """The configs' names, in the card's order."""
        var out = List[String]()
        for ref c in self.configs:
            out.append(c.name.copy())
        return out^

    def config(self, name: String = "") raises -> HubConfig:
        """The config called `name`; when empty, the one the card marks
        `default`, else the only one, else the one called `default`."""
        for ref c in self.configs:
            if c.name == name or (name == "" and c.default):
                return c.copy()
        var names = String(", ").join(self.config_names())
        if name == "":
            if len(self.configs) == 1:
                return self.configs[0].copy()
            for ref c in self.configs:
                if c.name == "default":
                    return c.copy()
            raise KeyError(
                t"datasets: {self.repo} has several configs; name one of "
                t"[{names}]"
            )
        raise KeyError(
            t"datasets: {self.repo} has no config '{name}'; it has [{names}]"
        )

    def uri(self, file: String) -> String:
        """The `hf://` location of one of the repository's files."""
        var at = String("@", self.revision) if self.revision != "" else ""
        return String("hf://datasets/", self.repo, at, "/", Uri.quote(file))

    def locate(
        self, data_files: DataFiles, split: String
    ) raises -> List[String]:
        """The locations of the repository's files `data_files` names for
        `split`, each once, sorted."""
        var patterns = data_files.patterns(split)
        var out = List[String]()
        for ref f in self.files:
            for ref p in patterns:
                if p.matches(f):
                    out.append(self.uri(f))
                    break
        sort(out)
        return out^

    def parquet_conversion(
        self, config: String, split: String
    ) raises -> List[String]:
        """The URLs of the Hub's Parquet conversion of `config`'s `split`,
        which exists for every public dataset at its default branch."""
        if self.revision != "":
            raise NotImplementedError(
                t"datasets: the Hub converts {self.repo} to Parquet at its "
                t"default branch only, not at revision '{self.revision}'"
            )
        return Self.parse_urls(
            Self._get(
                String(
                    "api/datasets/", self.repo, "/parquet/", config, "/", split
                )
            )
        )

    @staticmethod
    def parse_urls(body: List[UInt8]) raises -> List[String]:
        """The Hub's answer listing URLs: a JSON array of them, or an object
        carrying its error."""
        var json = Json(parse_bytes=Span(body))
        if json.is_object() and "error" in json.object():
            raise InvalidError(
                t"datasets: the Hub answered: {json.object()['error']}"
            )
        _HubJson.require_array(json, "file list")
        var out = List[String]()
        for ref url in json.array():
            out.append(_HubJson.string(url, "file URL"))
        return out^

    @staticmethod
    def _get(path: String) raises -> List[UInt8]:
        """GET `path` from the Hub, with `HF_TOKEN` as a bearer token when it
        is set, which a private or gated dataset needs."""
        var options: Dict[String, String] = {"endpoint": String(Self.ENDPOINT)}
        var token = getenv("HF_TOKEN")
        if token != "":
            options["token"] = token
        return OpenDalStore("http", options).read(path)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("HubDataset(", self.repo)
        if self.revision != "":
            writer.write("@", self.revision)
        for ref c in self.configs:
            writer.write(", ", c)
        writer.write(")")
