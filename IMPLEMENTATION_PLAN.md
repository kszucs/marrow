# Implementation Plan: Port AOT Value Expressions from `faszom` to `marrow/expr/`

## Confirmed Decisions

| Question | Choice |
|----------|--------|
| **1. AOT Entry Point** | Option C: Auto-detect in `Planner.build()` + explicit `Planner.build_aot[Expr]()` |
| **2. Column Access** | Option A: `FusedValueProcessor` captures `BufferView`s, custom evaluator inlines loads |
| **3. Boolean Output** | Option A: Single processor, always returns `AnyArray` |
| **4. Schema/Filter** | Move to `marrow/expr/relations.mojo` |
| **5. faszom.mojo** | Check usage → delete or keep only `RuntimeExpr` |
| **6. SIMD Width** | Match `faszom`: `max(8, simd_byte_width() // size_of[InNative])` |

## Implementation Steps

| Step | File | Action |
|------|------|--------|
| **1** | `marrow/expr/values.mojo` | Add AOT traits (`AOTValue`, `AOTNumericValue`, `AOTBoolValue`) + 15 node types + factories `aot_col`, `aot_lit` |
| **2** | `marrow/expr/executor.mojo` | Add `FusedValueProcessor[Expr: AOTValue]` + `_is_fully_aot()` + `Planner.build_aot[Expr]()` + modify `Planner.build()` auto-detection |
| **3** | `marrow/expr/relations.mojo` | Move `FieldDescriptor`, `Field`, `field`, `_schema_find_idx`, `Schema`, `table`, `Filter` from `faszom` |
| **4** | `marrow/expr/__init__.mojo` | Export all new AOT types and factories |
| **5** | `marrow/tests/test_expr_aot.mojo` | **New** - correctness tests |
| **6** | `marrow/bench_expr_aot.mojo` | **New** - AOT vs runtime benchmarks |
| **7** | `marrow/faszom.mojo` | Check `grep -r "RuntimeExpr\|faszom"` → delete or keep only `RuntimeExpr` |

## Verification Commands

```bash
# Tests
pixi run -e dev pytest marrow/tests/test_expr_aot.mojo -v

# Benchmarks  
pixi run -e dev pytest marrow/bench_expr_aot.mojo --benchmark

# Full test suite
pixi run -e dev test
```

## Progress

- [x] Step 1a: Add imports to `values.mojo`
- [x] Step 1b: Add AOT traits (`AOTValue`, `AOTNumericValue`, `AOTBoolValue`) to `values.mojo`
- [ ] Step 1c: Add AOT leaf nodes (`AOTColumn`, `AOTLiteral`)
- [ ] Step 1d: Add AOT arithmetic nodes (`AOTNegate`, `AOTAdd`, `AOTSub`, `AOTMul`, `AOTDiv`)
- [ ] Step 1e: Add AOT comparison nodes (`AOTEqual`, `AOTNotEqual`, `AOTLess`, `AOTLessEq`, `AOTGreater`, `AOTGreaterEq`)
- [ ] Step 1f: Add AOT boolean nodes (`AOTAnd`, `AOTOr`, `AOTNot`)
- [ ] Step 1g: Add factory functions (`aot_col`, `aot_lit`)
- [ ] Step 2: Add `FusedValueProcessor` and modify `Planner` in `executor.mojo`
- [ ] Step 3: Move Schema/Filter to `relations.mojo`
- [ ] Step 4: Update exports in `__init__.mojo`
- [ ] Step 5: Create test file
- [ ] Step 6: Create benchmark file
- [ ] Step 7: Clean up `faszom.mojo`