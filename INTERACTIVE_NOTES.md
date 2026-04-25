# Interactive Notes

This file tracks the manual `runs/codex-interactive` work so the important state is visible from the project root.

## Current scope

- Interactive manual submissions now exist for Level 1 problems `1-80` under `runs/codex-interactive`.
- Interactive manual submissions now also exist for CUDA Level 2 problems `1-50`, plus PTX Level 2 problems `9`, `12`, `14`, `18`, `22`, `29`, `33`, `37`, `39`, `40`, and `45`, under `runs/codex-interactive`.
- Interactive manual submissions now also exist for CUDA Level 3 problems `1-10` under `runs/codex-interactive`.
- Exploratory hybrid PTX+torch submissions now also exist under `runs/codex-interactive/ptx-hybrid`.
- Timing summaries for the interactive run live under `results/timing/codex-interactive/{ptx,cuda}/level1/summary.json`.
- Level 2 timing summaries now live under `results/timing/codex-interactive/{ptx,cuda}/level2/summary.json`.

## Important findings

- CUDA interactive work is currently the stronger path.
- PTX interactive work is usable for some elementwise and simpler norm cases, but the handwritten PTX norm/reduction kernels become fragile quickly under real assembly/runtime.
- Nsight profiling is working on this machine after installing `nsight-python`, fixing the local wrapper, and enabling GPU performance counters.

## Current Level 1 status snapshot

- `1-10`: both backends exist; CUDA is mostly strong, PTX is mostly correct but slower.
- `11-20`: both backends exist; CUDA is broadly usable, PTX is correct but mostly slower.
- `21-30`: both backends exist; activation-heavy block, CUDA mostly works, PTX is mixed but correct on the simpler cases.
- `31-40`: both backends exist.
  - PTX: `31`, `32`, and `36` are correct; `33`, `37`, `39`, `40` still fail at PTX assembly; `34`, `35`, `38` still hit illegal memory access.
  - CUDA: `31`, `32`, `33`, `34`, `35`, `36`, `37`, `39`, `40` are correct; `38` still OOMs because the benchmark input is about `8 GiB` and the out-of-place output needs another `8 GiB`.
- `41-50`: both backends exist.
  - PTX: `43`, `47`, `48`, and `49` are now correct. Lightweight snapshot:
    `43 ~1.53x`, `47 ~1.11x`, `48 ~0.88x`, `49 ~1.07x`.
    `41`, `42`, `44`, `45`, `46`, and `50` still need deeper kernel debugging; the remaining failures are now assembly/runtime issues rather than static-check rejections.
  - CUDA: `41-50` are all correct. Lightweight snapshot:
    `41 ~0.15x`, `42 ~0.84x`, `43 ~1.28x`, `44 ~1.04x`, `45 ~1.18x`, `46 ~0.93x`, `47 ~1.00x`, `48 ~1.00x`, `49 ~0.78x`, `50 ~0.78x`.
- `51-60`: interactive coverage now exists.
  - PTX: `51`, `52`, and `53` are now correct after cleanup. Lightweight snapshot:
    `51 ~1.80x`, `52 ~0.94x`, `53 ~1.10x`.
    Final PTX pass status:
    `54`, `59`, and `60` now get past static validation and hit illegal GPU memory access at runtime.
    `55` still fails PTX assembly because the kernel uses too much shared memory (`55296` bytes).
    `56` is down to a `literal_ptx_sources` policy failure only.
    `57` is down to a `literal_ptx_sources` policy failure after fixing the missing source-key mismatch.
    `58` now runs end-to-end but is numerically incorrect.
  - CUDA: `51-60` are all correct. Lightweight snapshot:
    `51 ~1.00x`, `52 ~0.92x`, `53 ~0.80x`, `54 ~1.37x`, `55 ~1.14x`, `56 ~2.05x`, `57 ~1.04x`, `58 ~0.90x`, `59 ~0.98x`, `60 ~1.15x`.
- `61-70`: interactive coverage now exists.
  - PTX: final pass status:
    `61` and `66` are down to `literal_ptx_sources` policy failures only.
    `62`, `64`, `67`, and `69` now get past static validation and hit illegal GPU memory access at runtime.
    `63` still crashes badly enough to poison the CUDA context during cleanup.
    `65` still fails PTX assembly because the kernel uses too much shared memory (`99328` bytes).
    `68` is still structurally incompatible with the PTX track because it falls back to `torch.nn.functional.conv3d`.
    `70` still fails PTX assembly because the kernel uses too much shared memory (`62208` bytes).
  - CUDA: `61-70` are all correct. Lightweight snapshot:
    `61 ~1.13x`, `62 ~1.24x`, `63 ~0.59x`, `64 ~0.87x`, `65 ~0.88x`, `66 ~0.80x`, `67 ~0.84x`, `68 ~0.99x`, `69 ~1.03x`, `70 ~1.02x`.
- `71-80`: interactive coverage now exists.
  - PTX: `74` and `80` are correct. Lightweight snapshot:
    `74 ~0.12x`, `80 ~5.41x`.
    `71`, `72`, and `73` are now static-clean but still fail at PTX assembly.
    `75` is still structurally static-invalid, `77` is still PTX-track-incompatible because it falls back to torch compute, and `76`, `78`, `79` are now static-clean but hit illegal GPU memory access at runtime.
  - CUDA: `71`, `72`, `74`, `75`, `76`, `77`, `78`, `79`, and `80` are correct. Lightweight snapshot:
    `71 ~1.88x`, `72 ~0.97x`, `74 ~1.57x`, `75 ~1.08x`, `76 ~0.93x`, `77 ~1.03x`, `78 ~1.22x`, `79 ~0.77x`, `80 ~1.04x`.
    `73` still mismatches numerically even after aligning the wrapper signature more closely with the vendor problem.

## Current Level 2 status snapshot

- `1-10`: interactive coverage now exists.
  - CUDA: `1-10` are all correct on the light pass. Lightweight snapshot:
    `1 ~1.02x`, `2 ~0.96x`, `3 ~1.04x`, `4 ~1.00x`, `5 ~0.95x`, `6 ~1.01x`, `7 ~1.00x`, `8 ~1.25x`, `9 ~1.00x`, `10 ~0.99x`.
  - PTX: `9` is landed and correct, but very slow:
    `9 ~0.0043x`.
    The rest of `1-10` are conv/deconv-heavy and need real kernel authoring rather than the simple GEMM-style PTX path that worked for `9`.
- `11-20`: interactive coverage now exists.
  - CUDA: `11-20` are all correct on the light pass. Lightweight snapshot:
    `11 ~0.963x`, `12 ~1.001x`, `13 ~0.987x`, `14 ~0.999x`, `15 ~0.999x`, `16 ~1.017x`, `17 ~1.005x`, `18 ~1.001x`, `19 ~0.949x`, `20 ~1.002x`.
  - PTX: only the GEMM-shaped composites are landed so far.
    `12 ~0.0040x`, `18 ~0.0004x` are correct but extremely slow.
    `14` is close but still fails correctness on the light pass due to numerical mismatch.
    The remaining conv/deconv-heavy cases still need actual PTX kernels rather than wrapper promotion.
- `21-30`: interactive coverage now exists.
  - CUDA: `21-30` are all correct after isolated reruns cleared stale batch results for `29` and `30`. Lightweight snapshot:
    `21 ~1.12x`, `22 ~1.06x`, `23 ~1.00x`, `24 ~1.01x`, `25 ~1.05x`, `26 ~1.00x`, `27 ~0.92x`, `28 ~0.91x`, `29 ~1.00x`, `30 ~1.77x`.
  - PTX: the GEMM-shaped cases are partially landed.
    `22 ~0.0044x` is correct but extremely slow.
    `29` is also correct in isolated rerun at about `0.0043x`, though the aggregate summary still needs a clean refresh.
    `14` remains the closest wrong-answer case from the earlier block.
    The conv/deconv-heavy cases in `21-28` and `30` still need real PTX kernel authoring rather than wrapper promotion.
- `31-40`: interactive coverage now exists.
  - CUDA: `31-40` are all correct on the light pass. Lightweight snapshot:
    `31 ~0.97x`, `32 ~1.04x`, `33 ~1.00x`, `34 ~1.00x`, `35 ~0.95x`, `36 ~0.89x`, `37 ~0.97x`, `38 ~0.90x`, `39 ~1.11x`, `40 ~1.00x`.
  - PTX: the attempted GEMM/norm subset `33`, `37`, `39`, and `40` did not land.
    These are currently blocked at PTX static validation by disallowed torch-side expressions and calls in the normalization/residual tails, not by PTX assembly/runtime.
    The conv/deconv-heavy cases `31`, `32`, `34`, `35`, `36`, and `38` still need real PTX kernel authoring.
- `41-50`: interactive coverage now exists.
  - CUDA: `41-50` are all correct on the light pass. Lightweight snapshot:
    `41 ~1.00x`, `42 ~1.00x`, `43 ~0.96x`, `44 ~1.00x`, `45 ~1.03x`, `46 ~1.01x`, `47 ~1.00x`, `48 ~1.01x`, `49 ~1.01x`, `50 ~1.12x`.
  - PTX: attempted `45` did not land.
    It is currently blocked at PTX static validation by disallowed torch-side expressions/calls in the second GEMM and logsumexp tail.
    The conv/deconv-heavy cases `41-44` and `46-50` still need real PTX kernel authoring.

## Current Level 3 status snapshot

- `1-10`: interactive CUDA coverage now exists.
  - CUDA: `1-10` are all correct on the light pass. Lightweight snapshot:
    `1 ~1.01x`, `2 ~0.93x`, `3 ~1.00x`, `4 ~1.01x`, `5 ~0.96x`, `6 ~1.00x`, `7 ~1.00x`, `8 ~0.99x`, `9 ~0.99x`, `10 ~0.96x`.
  - PTX: not attempted yet.
    On this tier the first-pass strategy is CUDA-only because these full-model graphs are not realistic PTX candidates without much deeper kernel decomposition.

## Hybrid PTX notes

- `level3/1_MLP`: the exploratory hybrid PTX+torch version is correct and now near eager after switching to torch `nn.Linear` plus PTX ReLU for hidden layers:
  `~3.79 ms` vs `~3.68 ms`, about `0.97x`.
- `level3/3_DeepNarrowMLP`: the same combo is correct and now slightly faster than eager on the light pass:
  `~1.67 ms` vs `~1.69 ms`, about `1.01x`.
- Current takeaway:
  hybrid PTX+torch only became interesting once the naive PTX GEMM replacement was dropped. PTX ReLU plus cuBLAS-backed torch linears is competitive; naive PTX linears were not.

## Practical notes

- For CUDA, extension-backed wrappers around ATen CUDA ops compile cleanly via `load_inline` and are the fastest way to cover more benchmark problems interactively.
- CUDA static validation still requires a `kernel_impl` marker such as `load_inline`, `cpp_extension`, or `__global__`. Pure module wrappers can be correct but still be rejected by the harness.
- Parameterized CUDA modules that mirror randomly initialized reference layers need to reset to `torch.initial_seed()` before constructing weights, otherwise they can fail correctness just from seed drift versus the reference model.
- For PTX, static-check clean does not imply runtime-safe. Re-eval is required after every PTX rewrite.
- Promoted PTX samples from the matched run are useful for coverage, but not trustworthy without isolated reruns. The `41-50` cleanup showed that some failures were only static-check noise, while others were genuine assembly/runtime bugs.
- Across the `54-70` PTX block, one common promoted-sample bug was invalid shared-memory address conversion syntax: `cvta.to.shared.u64 ..., <symbol>` needed to be rewritten to `cvta.shared.u64 ..., <symbol>`. That change moved multiple kernels from assembly failure into real runtime/resource failures.
- The remaining builder-backed PTX files (`56`, `57`, `61`, `66`) are now separated cleanly from kernel issues: their main remaining blocker is the PTX track policy that requires `PTX_SOURCES` to be a literal string or literal string dict.
- The `71-80` PTX block follows the same pattern: promoted samples are useful for quick coverage, but only a subset survives static validation or real runtime. In this block the immediate winners are `74` and `80`; `71-73` still need kernel-level assembly fixes, `75` is still structurally static-invalid, and `76/78/79` are now confirmed kernel-side illegal-address failures rather than validator noise.
- Grouped evals sometimes become misleading once one kernel poisons the CUDA context. Per-problem reruns are often the reliable source of truth.
- Level 2 continues to show a clearer split than Level 1: CUDA wrappers around the exact vendor compositions are easy to land and mostly sit near eager or slightly above it, and CUDA `1-50` is now covered on the light pass. PTX only lands cleanly on the GEMM-shaped composites (`9`, `12`, `18`, `22`, and isolated `29`) and even there is far slower than eager. The next PTX hurdle after basic GEMM is the static checker itself: mixed torch/PTX tails like `33/37/39/40/45` need more fully-PTX forward paths to land at all.
- Level 3 starts even more CUDA-skewed than Level 2. The CUDA wrappers for `1-10` all landed cleanly and are roughly at eager. PTX was intentionally not attempted on this first pass because these are full-network graphs, not small composite kernels, so a serious PTX submission would need end-to-end graph decomposition rather than a lightweight interactive wrapper.
- The exploratory `ptx-hybrid` lane now answers the “should PTX be a combo?” question more usefully for Level 3 MLPs. Naive PTX linears were a dead end, but a mixed path with torch `nn.Linear` for the heavy GEMMs and PTX ReLU for the hidden activations is at least performance-credible on the light pass (`~0.97x` on `1_MLP`, `~1.01x` on `3_DeepNarrowMLP`).

## Files to check first

- `runs/codex-interactive/README.md`
- `results/timing/codex-interactive/ptx/level1/summary.json`
- `results/timing/codex-interactive/cuda/level1/summary.json`
- `results/timing/codex-interactive/cuda/level2/summary.json`
- `results/timing/codex-interactive/ptx/level2/summary.json`
- `results/timing/codex-interactive/cuda/level3/summary.json`
- `runs/codex-interactive/ptx-hybrid/level3/001_eval.json`
- `runs/codex-interactive/ptx-hybrid/level3/003_eval.json`
