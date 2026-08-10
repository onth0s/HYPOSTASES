# Implementation Plan: NP-Hard Inference Mitigations

Operationalizes the four engineering strategies discussed in `misc/np-rock-hard.md` into the HYPOSTASES inference engine. Phases are strictly sequential — each phase is self-contained, testable, and non-breaking before the next begins.

---

## Phase 1 — Bounded History (Lag Window)

**Goal:** Add an optional `lag_window` parameter that truncates the observation trace to the last `k` steps before the filter runs. Eliminates O(T) full-trace overhead for long episodes.

**Theoretical grounding:** `c.memory_decay` already encodes that older observations become exponentially irrelevant. Bounded history makes this explicit in the inference call.

---

### Changes

#### [MODIFY] [`particle_filter.py`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/src/hypostases/inference/particle_filter.py)

- Add `lag_window: int | None = None` to `infer`, `infer_joint`, and `infer_mean_field` signatures.
- At the top of each function body, before the main loop:
  ```python
  if lag_window is not None:
      observed_actions = observed_actions[-lag_window:]
      observed_pool_trace = observed_pool_trace[-lag_window:]
  ```
- Docstrings updated with `lag_window` parameter docs referencing `c.memory_decay`.

#### [MODIFY] [`tests/test_inference.py`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/tests/test_inference.py)

- New test: `test_infer_lag_window_truncates_trace` — verifies that passing `lag_window=1` produces a valid particle set from a 5-step trace.
- New test: `test_infer_lag_window_none_unchanged` — verifies `lag_window=None` is identical to current behavior.

### Verification
```
pytest tests/test_inference.py -v
ruff check . && ruff format --check .
```

---

## Phase 2 — Constraint Pruning Gate

**Goal:** Short-circuit `action_likelihood` with a hard feasibility check before the Gaussian evaluation. Dead particles (physically impossible given the observed action) get `LIKELIHOOD_MIN` immediately, skipping float arithmetic.

**Rules per action type:**
- `REQUEST(amount)`: feasible if `pool_belief >= amount * 0.1` (pool can plausibly grant something).
- `SHARE(amount)`: feasible if `p.sigma.c.reserve >= amount` (agent has reserve to give).
- `WITHDRAW`: always feasible (zero-amount status signal by spec §5.8 declared simplification).

---

### Changes

#### [MODIFY] [`likelihood.py`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/src/hypostases/engine/likelihood.py)

- New private function `_is_infeasible(agent: AgentState, action: Action, pool_belief: float) -> bool`.
- `action_likelihood` calls `_is_infeasible` at entry; if `True`, returns `LIKELIHOOD_MIN` immediately.

#### [MODIFY] [`particle_filter.py`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/src/hypostases/inference/particle_filter.py)

- No changes needed — pruning is encapsulated inside `action_likelihood`.

#### [MODIFY] [`tests/test_likelihood.py`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/tests/test_likelihood.py)

- New tests: infeasible SHARE returns `LIKELIHOOD_MIN`, feasible SHARE proceeds to Gaussian, WITHDRAW always feasible.

### Verification
```
pytest tests/test_likelihood.py tests/test_inference.py -v
ruff check . && ruff format --check .
```

---

## Phase 3 — Hierarchical Decomposition (`infer_hierarchical`)

**Goal:** Two-pass filter. Macro-pass identifies the dominant `GoalCategory` cluster cheaply; micro-pass runs the full filter with a biased prior concentrated on that cluster.

**Architecture:**
1. **Macro-pass:** `infer(n_particles=50, ...)` → `goal_posterior(particles)` → identify top-1 (or top-2) `GoalCategory`.
2. **Biased prior:** modified `sample_prior` with `goal_bias: dict[GoalCategory, float] | None` that adds a bias offset to `u[k]` for dominant goals.
3. **Micro-pass:** `infer(n_particles=N, prior_type=..., goal_bias=..., ...)` → final posterior.

**Known caveat:** SURVIVAL and ACQUISITION both map to `REQUEST` — they share an action-type and are only distinguishable by amount. The macro-pass may not resolve them without amount-clustering. This is documented in the function's docstring; the macro-pass will still correctly distinguish REQUEST vs SHARE vs WITHDRAW goal clusters.

---

### Changes

#### [MODIFY] [`particle_filter.py`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/src/hypostases/inference/particle_filter.py)

- `sample_prior` gains optional `goal_bias: dict[str, float] | None = None` — adds bias to `u` vector at init.
- New public function `infer_hierarchical(...)` with signature matching `infer`, plus:
  - `macro_n_particles: int = 50`
  - `goal_bias_strength: float = 2.0`

#### [MODIFY] [`inference/__init__.py`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/src/hypostases/inference/__init__.py)

- Export `infer_hierarchical` in `__all__`.

#### [NEW] `tests/test_hierarchical_inference.py`

- Test: `test_hierarchical_execution` — runs on a 5-step SHARE trace, verifies particle count and weight sum.
- Test: `test_hierarchical_goal_bias_applied` — checks that biased prior shifts goal posterior toward RELATIONAL for SHARE-heavy traces.
- Test: `test_hierarchical_vs_flat_convergence` — both filters on the same trace; RELATIONAL posterior within 0.2 of each other.

### Verification
```
pytest tests/test_hierarchical_inference.py tests/test_inference.py -v
ruff check . && ruff format --check .
```

---

## Phase 4 — Rao-Blackwellization of World Model

**Goal:** Replace per-particle stochastic `w` propagation with closed-form Kalman updates conditioned on each particle's goal state. Only `(g.u, c.reserve, c.sociality, c.mood)` remain particle-tracked; `(w.mu, w.sigma2)` become per-particle Kalman state updated analytically.

**Why this is now unblocked:** AGENTS.md Rule 004 has been removed. `memory_decay` is no longer declared-but-inert.

**Mathematical basis:**
The world model update in `feedback` is:
```
delta_w["mu"]    = WORLD_MU_GAIN * surprise
delta_w["sigma2"] = WORLD_SIGMA2_UPDATE_GAIN * (|surprise| - w.sigma2)
```
Both are affine in `surprise = observed_delta - w.replenish_rate_est`. This is a linear observation model → exact Kalman step replaces the EMA approximation.

**Kalman equations (per particle):**
- Prediction: `mu_pred = mu + replenish_rate_est`; `sigma2_pred = sigma2 + Q` (process noise `Q` = new constant).
- Update: `K = sigma2_pred / (sigma2_pred + R)` (obs noise `R` = new constant); `mu_post = mu_pred + K * (obs - mu_pred)`; `sigma2_post = (1 - K) * sigma2_pred`.

---

### Changes

#### [MODIFY] [`constants.py`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/src/hypostases/engine/constants.py)

- New constants: `KALMAN_PROCESS_NOISE_Q: Final[float]` and `KALMAN_OBS_NOISE_R: Final[float]` (initial values calibrated from existing `WORLD_SIGMA2_UPDATE_GAIN` and `WORLD_MU_GAIN` ratios).

#### [MODIFY] [`dynamics.py`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/src/hypostases/engine/dynamics.py)

- New function `evolve_rb(agent: AgentState, phi: FeedbackDelta, surprise: float) -> None` — applies Kalman update to `w.mu` / `w.sigma2` instead of the EMA path; all other state fields updated identically to `evolve`.
- `evolve` is **not changed** — `evolve_rb` is an additive, opt-in variant.

#### [MODIFY] [`particle_filter.py`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/src/hypostases/inference/particle_filter.py)

- `infer` and `infer_joint` gain `use_rao_blackwell: bool = False` flag.
- When `True`, propagation calls `evolve_rb` instead of `evolve`; `surprise` is computed from `delta_log` before the per-particle loop (it's observation-shared, not particle-specific).

#### [MODIFY] [`inference/__init__.py`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/src/hypostases/inference/__init__.py)

- No new exports needed (flag on existing `infer` / `infer_joint`).

#### [NEW] `tests/test_rao_blackwell.py`

- Test: `test_rb_execution` — `infer(..., use_rao_blackwell=True)` runs without error.
- Test: `test_rb_sigma2_decreases_on_consistent_obs` — after 10 identical REQUEST observations, `sigma2` in the MAP particle is lower with RB than with flat filter (tighter belief).
- Test: `test_rb_vs_flat_reserve_map_close` — MAP reserve estimate from both paths within tolerance on a fixed-seed trace.

### Verification
```
pytest tests/ -v
ruff check . && ruff format --check .
```

---

## Verification Plan

### Automated tests (each phase)

Each phase runs its own focused tests before merging:

| Phase | Test command |
|---|---|
| 1 | `pytest tests/test_inference.py -v` |
| 2 | `pytest tests/test_likelihood.py tests/test_inference.py -v` |
| 3 | `pytest tests/test_hierarchical_inference.py tests/test_inference.py -v` |
| 4 | `pytest tests/ -v` |

Full suite after each phase:
```
pytest && ruff check . && ruff format --check .
```

### Behavioral regression guard

For phases 3 and 4, a fixed-seed trace run comparing MAP reserve/goal posterior between the new variant and the flat baseline will be added as an assertion to prevent silent behavioral drift.

---

## Out of scope

- `infer_mean_field` does not get `use_rao_blackwell` in Phase 4 (factorised mean-field already reduces coupling; RB benefit is marginal there).
- `memory_decay` activation on `c.memory_decay` as a *per-particle parameter* (vs. the Kalman path) remains a separate calibration exercise.
- Vectorization of the particle loop (NumPy/JAX batch) — separate performance effort, not part of this plan.
