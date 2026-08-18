import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "python"))
import marrow as ma
from marrow import col, lit

HITS = os.path.expanduser("~/Workspace/ClickBench/data/hits_0.parquet")
STRING = ma.string()


def s(n):
    return col(n).cast(STRING)


case = sys.argv[1]
t = ma.read_parquet(HITS)

if case == "q11":
    r = (
        t.filter(s("MobilePhoneModel") != lit(""))
        .aggregate(by=[s("MobilePhoneModel")], u=("count_distinct", "UserID"))
        .order_by(("u", False))
        .limit(10)
    )
elif case == "q11_nofilter":
    r = (
        t.aggregate(by=[s("MobilePhoneModel")], u=("count_distinct", "UserID"))
        .order_by(("u", False))
        .limit(10)
    )
elif case == "q11_count":
    r = (
        t.filter(s("MobilePhoneModel") != lit(""))
        .aggregate(by=[s("MobilePhoneModel")], u=("count", "UserID"))
        .order_by(("u", False))
        .limit(10)
    )
elif case == "q11_nolimit":
    r = t.filter(s("MobilePhoneModel") != lit("")).aggregate(
        by=[s("MobilePhoneModel")], u=("count_distinct", "UserID")
    )
elif case == "q11_noorder":
    r = (
        t.filter(s("MobilePhoneModel") != lit(""))
        .aggregate(by=[s("MobilePhoneModel")], u=("count_distinct", "UserID"))
        .limit(10)
    )
elif case == "filter_only":
    r = t.filter(s("MobilePhoneModel") != lit("")).aggregate(by=[], c=lit(1).count())
elif case == "distinct_nogroup":
    r = t.filter(s("MobilePhoneModel") != lit("")).aggregate(
        by=[], u=("count_distinct", "UserID")
    )
elif case == "distinct_nogroup_nofilter":
    r = t.aggregate(by=[], u=("count_distinct", "UserID"))
elif case == "q12":
    r = (
        t.filter(s("MobilePhoneModel") != lit(""))
        .aggregate(
            by=["MobilePhone", s("MobilePhoneModel")], u=("count_distinct", "UserID")
        )
        .order_by(("u", False))
        .limit(10)
    )
elif case == "q14":
    r = (
        t.filter(s("SearchPhrase") != lit(""))
        .aggregate(by=[s("SearchPhrase")], u=("count_distinct", "UserID"))
        .order_by(("u", False))
        .limit(10)
    )
elif case == "intkey_distinct":
    r = t.filter(s("MobilePhoneModel") != lit("")).aggregate(
        by=["MobilePhone"], u=("count_distinct", "UserID")
    )
elif case == "intkey_distinct_nofilter":
    r = t.aggregate(by=["MobilePhone"], u=("count_distinct", "UserID"))
else:
    raise SystemExit("unknown case " + case)

b = r.collect()
print(case, "OK", b)
