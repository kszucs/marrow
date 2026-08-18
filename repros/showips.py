"""Print the faulting-thread backtrace of a macOS .ips crash report."""

import json
import sys


def show(path):
    raw = open(path).read()
    _head, body = raw.split("\n", 1)
    d = json.loads(body)
    print("##", path)
    exc = d.get("exception", {})
    print("  ", exc.get("type"), exc.get("signal"), exc.get("subtype"))
    faulting = d.get("faultingThread")
    imgs = d["usedImages"]
    for i, t in enumerate(d["threads"]):
        if i != faulting:
            continue
        for f in t["frames"][:12]:
            im = imgs[f["imageIndex"]]
            sym = f.get("symbol", "?")
            print("   ", sym[:110], "|", im.get("name"))


for p in sys.argv[1:]:
    try:
        show(p)
    except Exception as e:  # noqa: BLE001
        print("##", p, "unparsable", e)
