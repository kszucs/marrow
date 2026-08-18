#!/bin/bash
# Baseline / after measurement driver for the projection-pushdown work.
# Usage: bash .baseline.sh <suffix>
set -u
SUF="$1"
OUT="/tmp/p1"
mkdir -p "$OUT"
cd /Users/kszucs/Workspace/marrow/.claude/worktrees/agent-a5de174c4a527ce7a

pixi run python3 benchmarks/binary_size/check_gate.py > "$OUT/gate_$SUF.log" 2>&1
echo "GATE_EXIT=$?" >> "$OUT/gate_$SUF.log"

pixi run -e bench pytest python/marrow/tests/test_clickbench.py > "$OUT/correct_$SUF.log" 2>&1
echo "CORRECT_EXIT=$?" >> "$OUT/correct_$SUF.log"

pixi run -e bench python python/marrow/tests/bench_clickbench.py > "$OUT/bench_$SUF.log" 2>&1
echo "BENCH_EXIT=$?" >> "$OUT/bench_$SUF.log"

echo "ALLDONE_$SUF"
