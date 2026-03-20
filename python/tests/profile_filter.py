"""Profiling workload: filter kernel.

Run via:  pixi run profile python/tests/profile_filter.py
"""

import random
import marrow as ma


def main():
    n = 10_000_000
    iterations = 100

    arr = ma.array(list(range(n)), type=ma.int64())
    rng = random.Random(42)
    mask = ma.array([rng.random() < 0.5 for _ in range(n)], type=ma.bool_())

    for _ in range(iterations):
        ma.filter_(arr, mask)


if __name__ == "__main__":
    main()
