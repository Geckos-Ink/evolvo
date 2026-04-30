# PIPELINES.md

Practical best practices for **high-throughput Evolvo research workloads**.
This document consolidates lessons learned in `pcpl-evolvo` to avoid repeated performance and stability mistakes.

## Scope

Use this guide when your workload has at least one of these properties:

- large populations and many generations,
- expensive fitness/evaluation loops,
- mixed CPU/GPU execution (Kompute/Torch),
- resumable multi-round campaigns,
- repeated evaluation of near-duplicate genomes.

The goal is simple: **maximize useful evaluations per second**, not just raw worker count.

## Golden Rules

1. Treat throughput as a systems problem, not a single-flag tuning problem.
2. Keep the evaluator fed with unique, evaluable work.
3. Keep timeout and validity states explicit; never collapse them into generic low fitness.
4. Reuse compute artifacts aggressively (executors, plans, dedup cache, checkpoints).
5. Prefer stable progress and resumability over short-lived peak speed.

## 1) Parallelization Strategy

### 1.1 Choose backend by runtime characteristics

- CPU-only heavy evaluations:
  - prefer `--parallel-backend process`.
- Kompute/native GPU path (`--executor-backend kompute|kompute-sim|auto` with kp available):
  - prefer threads for evaluator parallelism (`process` is auto-downgraded unless explicitly allowed).
  - reason: process forking with Vulkan can be unstable and often slower in practice.

Operational controls:

- `--parallel-backend {auto,process,thread,off}`
- `--workers 0` for auto-budgeting, otherwise set an explicit cap.
- `--max-cpu-utilization` and `--max-gpu-utilization` to avoid host/GPU saturation.
- `--kompute-allow-process-pool` only when you have validated it on your host.

### 1.2 Use two-level parallelism deliberately

For long continuous runs:

- Intra-round parallelism: evaluator workers per round.
- Inter-round parallelism: concurrent round lanes (`--round-parallelism`, `--minimum-parallel-rounds`).

Best practice:

- synchronize learned/archive state at batch boundaries (`--round-state-sync batch-start`),
- run lanes in parallel, then merge round outputs in strict round order,
- persist pending checkpoints before merge.

This avoids cross-lane state races and keeps runs resumable.

### 1.3 Prefer shared executors over per-generation executor creation

Create worker pools once per round and reuse them across generations/stages.
Destroy only at round completion or hard watchdog fallback.

Why:

- lower process/thread setup overhead,
- fewer allocator and import churn spikes,
- more stable latency distribution.

### 1.4 Keep process scheduling fine-grained

When using process pools:

- keep map chunks small enough to prevent long-tail stragglers,
- balance task chunks by estimated complexity (effective instruction count, role-dependent cost),
- use watchdog-driven fallback for overdue tasks.

Do not let one heavyweight task stall all workers.

## 2) Pipeline Design (Quick -> Mid -> Full)

### 2.1 Always stage expensive evaluation

Use three stages:

- `quick`: low-cost statistical screen,
- `mid`: stronger filter on survivors,
- `full`: final high-fidelity scoring.

Also use scenario complexity tiers per stage (`quick`, `mid`, `hard`) and adaptive cycle fractions.

### 2.2 Keep-rate floors must track worker count

Do not cut so aggressively that full-stage finalists are fewer than available workers.

Use parallel-aware keep floors (quick/mid) to maintain enough finalists for full-stage saturation.
When full-stage load falls below worker capacity, throughput collapses even if many workers exist.

### 2.3 Throttle quick-stage load when duplicate pressure is high

If previous generations show high cache reuse and low uniqueness:

- reduce quick-stage candidate volume,
- prioritize novelty and non-archive signatures,
- apply predictive cuts to skipped candidates.

This trades redundant evaluations for unique signal.

### 2.4 Probe false negatives and run idle random trials

After staged cuts:

- probe a small sample of discarded candidates,
- estimate false-negative rate (`probe_win_rate`),
- inject random challengers when workers are underutilized.

This prevents premature convergence and keeps lanes occupied with useful exploration.

### 2.5 Deep Dive: Heterogeneous Cut Pipeline (GPU -> CPU -> GPU)

This part is **not fully solved yet** in `pcpl-evolvo`, but it is a key direction for brute-force throughput.
The target is to stop treating unsupported Kompute instructions as a monolithic fallback, and instead run the evaluable graph in cut segments.

Core idea:

- execute contiguous GPU-compatible instruction blocks on GPU,
- execute unsupported blocks on CPU in parallel,
- synchronize only the boundary state,
- continue with next GPU-compatible block.

Think in terms of **execution segments**, not whole-genome backend selection.

#### 2.5.1 Planning phase (before execution)

For each effective algorithm, build a cut plan:

1. Extract effective instruction list (`extract_effective_algorithm`).
2. Classify each instruction:
   - `gpu-native`
   - `cpu-required` (unsupported op, control-flow heavy, or policy-forced CPU)
3. Partition into maximal contiguous segments.
4. Attach dependency metadata at segment boundaries:
   - required input slots/symbols
   - produced output slots/symbols
   - data type and shape info

Recommended plan key (for cacheability):

- operation sequence hash,
- slot/type signature,
- runtime backend options (Kompute mode, native enable flags, fallback policy),
- scenario fingerprint class (quick/mid/full family).

#### 2.5.2 Execution phase (runtime)

Use a segmented scheduler:

1. Run `GPU segment 0` across a batch of genomes with the same plan signature.
2. Copy only boundary outputs (dirty subset) to host.
3. Run `CPU segment 1` in parallel workers over the same batch.
4. Copy only boundary outputs back to device.
5. Run `GPU segment 2`.
6. Repeat until segment list completes.

Two non-negotiable rules:

- never full-sync the whole state unless strictly required,
- never switch backend without a minimal boundary manifest.

#### 2.5.3 Minimal pseudocode for scheduler design

```text
plan = get_or_build_plan(genome_signature, runtime_signature)
state = init_symbol_state()

for segment in plan.segments:
    if segment.backend == GPU:
        run_gpu_segment(segment, batch, state.device_view)
        boundary = segment.boundary_outputs
        sync_device_to_host(boundary, state)
    else:
        run_cpu_segment_parallel(segment, batch, state.host_view)
        boundary = segment.boundary_outputs
        sync_host_to_device(boundary, state)
```

#### 2.5.4 Parallelization layers that should coexist

- Layer A: inter-round lanes (`round_parallelism`).
- Layer B: inter-genome batch evaluation.
- Layer C: intra-genome segment backend switching (GPU/CPU cuts).

Do not try to maximize all three blindly. Tune in this order:

1. stable segment execution,
2. stable genome batching,
3. higher round-lane parallelism.

#### 2.5.5 Where this fails in practice (and how to avoid it)

Typical failure modes:

- fallback storm: too many tiny CPU segments between GPU segments,
- transfer thrash: frequent large host<->device sync at boundaries,
- straggler tails: one expensive CPU segment stalls the whole batch,
- planner overhead: cut-plan building cost dominates short runs.

Mandatory mitigations:

- merge adjacent CPU segments when boundary is trivial,
- enforce a minimum GPU segment size threshold before offloading,
- batch genomes by compatible cut plan to reduce branchy scheduling,
- cache plan objects aggressively (RAM + disk when useful).

#### 2.5.6 Incremental implementation path for `pcpl-evolvo`

Do not attempt full heterogeneous scheduling in one refactor. Ship in phases:

Phase A: deterministic segmentation only

- build and persist segment plans from compatibility reports,
- keep current execution semantics unchanged,
- collect plan-frequency and segment-size statistics.

Phase B: boundary-minimized hybrid execution

- execute `GPU -> CPU -> GPU` on one genome with strict boundary sync,
- verify numerical parity against current fallback path,
- add checks for stale/dirty symbol transfer bugs.

Phase C: batch-level hybrid scheduler

- group genomes by plan signature,
- run segment-by-segment over batch,
- parallelize CPU segments with existing worker infrastructure.

Phase D: adaptive scheduler policy

- dynamically choose:
  - full CPU,
  - hybrid segmented,
  - mostly GPU,
  based on recent timeout and native-share statistics.

Suggested acceptance criteria per phase:

- lower `batch_seconds` at equal scoring semantics,
- reduced `native_cpu_fallback_count / native_dispatch_total`,
- stable or better `eval_timeout_ratio`,
- no drop in `eval_valid_ratio`.

### 2.6 Bottleneck Taxonomy For Intensive Brute Force

Use this table as a diagnosis shortcut:

| Bottleneck | Main symptom | Primary metric signal | First mitigation |
|---|---|---|---|
| GPU underutilization | low native acceleration benefit | low `native_gpu_share` with high worker count | increase compatible op density and batch by plan signature |
| CPU fallback dominance | many cut scores/timeouts despite Kompute | high `native_cpu_fallback_count` and `eval_timeout_ratio` | reduce unsupported op pressure; widen CPU segment parallelism |
| Sync overhead | no throughput gain when enabling GPU | high elapsed time with moderate op counts | boundary-only sync, avoid global state sync |
| Planner overhead | short scenarios become slower with Kompute | high fixed latency per eval in quick stage | persist Kompute plan cache and cut-plan cache |
| Duplicate saturation | more workers but same progress | high `cache_hits + dup_reuse`, low `eval_unique` | stronger novelty pressure and quick-stage throttle |
| Selection collapse | many rounds skip archive promotion | high `full_timeout_ratio`, frequent timeout rescue | reduce selection complexity/variants, rerank with rescue profile |

## 3) Caching Architecture

### 3.1 Use layered caches (all are needed)

1. Evaluation dedup cache (generation/round level):
   - key should include role, scenario fingerprint, opponent signature, genome signature.
2. Executor reuse cache (thread-local):
   - reuse `GFSLExecutor` instances by runtime kwargs/role.
3. Algorithmic LRU caches:
   - cache repeated analytical profiles (period/rank style computations).
4. Kompute execution-plan cache:
   - RAM + disk cache for planner/profile reuse.

### 3.2 Dedup cache policy

- Keep LRU bounded (`--max-eval-cache-entries`).
- Cache score and metrics rows for reusable outcomes.
- Count both cache hits and duplicate reuse explicitly.
- Track uniqueness ratio: `eval_unique / total_stage_eval`.

A bigger cache is not always better: grow only when uniqueness remains high and memory allows.

### 3.3 Kompute plan cache policy

Keep these enabled for intensive runs:

- `EVOLVO_KOMPUTE_PLAN_CACHE_ENABLE=1`
- `EVOLVO_KOMPUTE_PLAN_DISK_CACHE_ENABLE=1`

Useful tuning knobs:

- `EVOLVO_KOMPUTE_PLAN_CACHE_MAX_ENTRIES`
- `EVOLVO_KOMPUTE_PLAN_DISK_CACHE_MAX_FILES`
- `EVOLVO_KOMPUTE_PLAN_DISK_CACHE_DIR`

Without plan caching, planner overhead can dominate short scenario evaluations.

### 3.4 Startup/warm caches before long campaigns

Run pre-compilation/warmup before multi-hour runs:

- `./demo/pcpl-evolvo/compile_perf.sh`

This reduces cold-start penalties from imports and bytecode generation.

### 3.5 Cache hierarchy for cut pipelines

For segmented GPU/CPU execution, use a strict cache hierarchy:

- L0 (thread-local): executor objects and tiny hot metadata.
- L1 (process memory): cut plans + boundary manifests + op coverage summaries.
- L2 (shared disk): Kompute execution plans and reusable planner artifacts.
- L3 (run artifacts): round summaries/statistics used for adaptive policy.

If any layer is missing, throughput regresses:

- missing L0/L1 -> scheduler/planner churn,
- missing L2 -> repeated cold planner work on restart,
- missing L3 -> adaptive logic relearns from zero every run.

### 3.6 What to include in cache keys (to avoid false reuse)

Cache keys should include all variables that can change semantics or speed:

- genome canonical/evaluation signature,
- scenario fingerprint (including stage complexity and variants),
- opponent signature (for attacker/defender coupling),
- backend policy (`auto/cpu/kompute/kompute-sim`),
- Kompute mode flags (native family toggles, unsupported thresholds),
- runtime environment traits when relevant (device index/queue family).

Do not key only by genome hash.
In co-evolution and staged evaluation, that produces incorrect reuse.

## 4) Timeout and Bottleneck Control

### 4.1 Timeout is an execution status, not a fitness result

Maintain explicit status buckets:

- `valid`
- `valid-no-metrics`
- `timeout-cut`
- `complexity-cut`
- `error-empty`

If timeout is mixed with true low fitness, selection pressure becomes misleading and search collapses.

### 4.2 Apply watchdogs in process mode

Use watchdogs for overdue tasks and force fallback results when needed.
Do not wait indefinitely for stragglers.

Debug tools:

- `--debug-eval-timeout-seconds`
- `--debug-eval-log-interval-seconds`

### 4.3 Use selection rescue paths

If attacker-panel ranking collapses into timeout cuts:

- rerank with reduced scenario cost,
- reduce key variants and cycle fraction temporarily,
- increase timeout budget for rescue pass,
- avoid promoting empty-metric winners.

### 4.4 Gate archive promotion on evaluability

Promote elites only when final defender metrics are present.
Persist timeout-only rounds for diagnostics, but skip archive insertion.

This protects future rounds from being seeded by unevaluable genomes.

## 5) Continuous Campaign Reliability

### 5.1 Persist frequent checkpoints

Keep these artifacts updated:

- `results.json` (running summary + active batch state),
- `archive.json` (elites + predictive profile),
- `round-progress.json` (live lane phase),
- `round-pending.json` (pre-merge completed lane),
- `round-results.json` / `round-report.md` (merged round output).

This allows restart without losing expensive completed work.

### 5.2 Keep resumability default

- run with stable `--out-dir`,
- keep `resume` enabled unless intentionally starting fresh (`--no-resume`).

### 5.3 Merge parallel lanes deterministically

Batch rounds can finish out of order; merge in round index order.
Deterministic merge order keeps archives reproducible and easier to analyze.

## 6) Observability KPIs (Track Every Generation)

Minimum KPI set:

- `batch_seconds` vs `target_batch_seconds`
- `quick_eval`, `mid_eval`, `full_eval`
- `eval_unique`, `cache_hits`, `dup_reuse`
- `probe_samples`, `probe_win_rate`
- `eval_timeout_ratio`, `full_timeout_ratio`
- `random_trials`, `random_injected`
- `parallel_rebalanced`, `underutilization_boost`

Interpretation shortcuts:

- High `cache_hits + dup_reuse` with low `eval_unique`: too repetitive; increase diversity pressure and throttle quick intake.
- High `full_timeout_ratio`: selection workload is too heavy; trigger rescue profile.
- Low `full_eval` relative to workers: keep-rates or quick throttling are too aggressive.

## 7) Anti-Patterns to Avoid

1. Maximizing worker count without checking uniqueness and timeout ratios.
2. Using process pools with Kompute on unvalidated hosts.
3. Running full-depth evaluation on all genomes every generation.
4. Promoting elites from timeout-only/empty-metric final evaluation.
5. Disabling checkpoints in long runs.
6. Interpreting generation-0 winners across many rounds as optimization success.

## 8) Recommended Run Playbooks

### Fast verification (pipeline correctness)

```bash
python3 demo/pcpl-evolvo/run_experiments.py \
  --profile fast \
  --mode dynamic \
  --rounds 1 \
  --print-effective-config
```

### Long CPU campaign (stable baseline)

```bash
python3 demo/pcpl-evolvo/run_experiments.py \
  --profile full \
  --mode paper \
  --out-dir demo/pcpl-evolvo/runs/mainline-cpu \
  --parallel-backend process \
  --workers 0
```

### Kompute campaign (safe default)

```bash
python3 demo/pcpl-evolvo/run_experiments.py --kompute-self-test

python3 demo/pcpl-evolvo/run_experiments.py \
  --profile full \
  --mode paper \
  --executor-backend kompute \
  --parallel-backend auto \
  --workers 0 \
  --out-dir demo/pcpl-evolvo/runs/mainline-kompute
```

### Continuous sweep with resumability

```bash
python3 demo/pcpl-evolvo/run_experiments.py \
  --continuous \
  --profile full \
  --mode dynamic \
  --out-dir demo/pcpl-evolvo/runs/continuous-mainline \
  --workers 0
```

## 9) Quick Checklist Before Every Intensive Run

- Kompute host validated (`--kompute-check-libs` / `--kompute-self-test`) when applicable.
- Output directory fixed and resumable.
- Parallel backend chosen according to runtime mode (CPU vs Kompute).
- Staged predictive pipeline enabled.
- Eval cache size set for expected population/round scale.
- Timeout watchdog/debug knobs ready for long runs.
- Checkpoint artifacts monitored (`results.json`, `round-progress.json`, `round-pending.json`).

---

If you must choose one principle above all:

> Optimize for **valid, diverse, full-stage-evaluable genomes per second**, not raw worker throughput.
