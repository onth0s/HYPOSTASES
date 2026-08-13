# Front 02 Master Specification — Explicit Planning Layer

**Status**: RATIFIED SPECIFICATION (Ingested Literature Synthesized)  
**Wave**: Wave 2 (Structural Abstraction & Metacognitive Planning)  
**Front**: Front 02 — Explicit Planning Layer  
**Target Substrate**: HYPOSTASES Engine ($\sigma = (c, w, g, \rho_{\text{ext}})$)  

---

## 1. Ingested Literature Foundation (`docs/WAVE_2_FRONT_02/papers/`)

This specification synthesizes the theoretical mechanisms, mathematical formulations, and algorithmic structures from all 11 foundational papers ingested into `docs/WAVE_2_FRONT_02/papers/`:

| Source Paper | Key Architectural Synthesis |
|---|---|
| [`adaplanner_adaptive_planning_from_feedback.pdf`](../../WAVE_2_FRONT_02/papers/adaplanner_adaptive_planning_from_feedback.pdf) | Dual-role Planner/Refiner, code-style prompting, in-plan vs. out-of-plan assertion checks, and "refine-then-resume" (`start_from=k`) sub-plan repair. |
| [`goap_applying_goal_oriented_action_planning_2003.pdf`](../../WAVE_2_FRONT_02/papers/goap_applying_goal_oriented_action_planning_2003.pdf) | Decoupling goals from actions, state atom sets $W$, pre-condition $\phi_{\text{pre}}$ / effect $\phi_{\text{post}}$ matching, and A* backward chaining. |
| [`goap_three_states_and_a_plan_fear_2006.pdf`](../../WAVE_2_FRONT_02/papers/goap_three_states_and_a_plan_fear_2006.pdf) | Dynamic plan validation at execution tick $t$; instant plan invalidation when $\text{precond}(a) \nsubseteq W_t$. |
| [`htn_domain_repair_context_free_grammars_llm.pdf`](../../WAVE_2_FRONT_02/papers/htn_domain_repair_context_free_grammars_llm.pdf) | Grammar-based HTN domain model repair treating plan reduction trees as Context-Free Grammar (CFG) production rules. |
| [`htn_planning_complexity_expressivity_erol.pdf`](../../WAVE_2_FRONT_02/papers/htn_planning_complexity_expressivity_erol.pdf) | Task Networks $TN = (T, S, C)$, compound task decomposition methods $M = (d, tn, \phi)$, and structural constraint enforcement. |
| [`lats_language_agent_tree_search.pdf`](../../WAVE_2_FRONT_02/papers/lats_language_agent_tree_search.pdf) | Unifying tree search, state value functions $V(s)$, trajectory reflection memory $R$, and dynamic contingency sub-trees. |
| [`plan_and_solve_prompting_zero_shot_cot.pdf`](../../WAVE_2_FRONT_02/papers/plan_and_solve_prompting_zero_shot_cot.pdf) | Two-phase Plan-and-Solve execution, explicit state variable maintenance, and step-by-step subgoal tracking. |
| [`rap_reasoning_with_lm_planning_with_world_model.pdf`](../../WAVE_2_FRONT_02/papers/rap_reasoning_with_lm_planning_with_world_model.pdf) | Repurposing simulation forward rollouts (`counterfactual.py`) as a world model $P(s' \mid s, a)$, UCT action selection, and Q-value backpropagation. |
| [`self_pr_self_guided_planning_repair_code_gen.pdf`](../../WAVE_2_FRONT_02/papers/self_pr_self_guided_planning_repair_code_gen.pdf) | Failure Feedback Analyzer, localized sub-graph plan patching, and execution exception handling. |
| [`tree_of_thoughts_deliberate_problem_solving.pdf`](../../WAVE_2_FRONT_02/papers/tree_of_thoughts_deliberate_problem_solving.pdf) | Systematic candidate thought generation $G(s, k)$, state evaluation voting/scoring, and BFS/DFS search tree expansion. |
| [`voyager_open_ended_embodied_agent_llms.pdf`](../../WAVE_2_FRONT_02/papers/voyager_open_ended_embodied_agent_llms.pdf) | Persistent skill & strategy library (`PlanLibrary`) indexing executable plan programs $T_\Pi$ by goal type and prerequisite state bounds. |

---

## 2. Integrated Theoretical Architecture & Mathematical Formalism

```
                             [ Agent Primitive State σ_t = (c, w, g, ρ) ]
                                                │
                                                ▼
                             ┌─────────────────────────────────────┐
                             │  1. GOAP / HTN Strategy Retrieval   │
                             │     PlanLibrary Lookup & A* Search  │
                             └──────────────────┬──────────────────┘
                                                │
                                                ▼
                             ┌─────────────────────────────────────┐
                             │  2. Active Plan Object Π            │
                             │     - Task Network TN = (T, S, C)   │
                             │     - Pre/Post conditions φ_pre/post│
                             │     - Invariants φ_inv & Branches B │
                             └──────────────────┬──────────────────┘
                                                │
                                                ▼
                             ┌─────────────────────────────────────┐
                             │  3. Closed-Loop Execution Monitoring│
                             │     (AdaPlanner / GOAP Tick Check)  │
                             └────────┬───────────────────┬────────┘
                                      │                   │
                     Valid Assertion  │                   │ Assertion Failure / State Deviation
                                      ▼                   ▼
                             ┌────────────────┐  ┌──────────────────────────────────┐
                             │ Execute Action │  │ 4. PlanRepairEngine              │
                             │ Step a_t       │  │    - Out-of-Plan Exception        │
                             └────────────────┘  │    - Front 04 Counterfactual Rollout│
                                                 │      (RAP MCTS / LATS Reflection)│
                                                 │    - Refine-Then-Resume at k     │
                                                 └──────────────────────────────────┘
```

### 2.1 Formal Data Model: Plan Object ($\Pi$)
Synthesizing GOAP (Orkin 2003) and HTN (Erol et al. 1994):
A plan object $\Pi$ is formally defined as:

$$\Pi = \left( g, \mathcal{N}, \mathcal{E}, \phi_{\text{global\_inv}}, \text{status}, \text{breakpoint\_k} \right)$$

where:
- $g \in \mathcal{G}$ is the target goal predicate / utility function.
- $\mathcal{N} = \{n_1, n_2, \dots, n_K\}$ is an ordered graph of task nodes. Each node $n_i = (a_i, \phi_{\text{pre}, i}, \phi_{\text{post}, i}, \hat{\sigma}_{i+1}, \Delta u_{\text{expected}, i})$.
- $\mathcal{E} \subseteq \mathcal{N} \times \mathcal{N}$ represents execution edges and dynamic contingency sub-trees $B_{\text{contingency}}(w, c)$ (LATS).
- $\phi_{\text{global\_inv}}$ is the state invariant predicate over $\sigma$.
- $\text{breakpoint\_k} \in \{1 \dots K\}$ is the AdaPlanner execution checkpoint index ("start_from").

### 2.2 Execution & Out-of-Plan Monitoring Dynamics
At each tick $t$, given state $\sigma_t = (c_t, w_t, g_t, \rho_{\text{ext}, t})$:
1. **In-Plan Verification**:
   - Verify global state invariant: $\sigma_t \models \phi_{\text{global\_inv}}$.
   - Verify step precondition: $\sigma_t \models \phi_{\text{pre}, \text{current}}$.
   - Verify state deviation: $\|\text{proj}_w(\sigma_t) - \text{proj}_w(\hat{\sigma}_t)\| \le \epsilon_{\text{deviation}}$.
2. **In-Plan Refinement (AdaPlanner `ask_LLM` / State Extraction)**:
   - If in-plan observations contain dynamic environment updates (e.g. receptacle lists or peer message tokens), update intermediate state variables without altering the macro plan sequence.
3. **Out-of-Plan Interruption**:
   - If any verification check fails, raise `OutOfPlanInterruption(failed_node=k, execution_trace, error_signature)`.

### 2.3 Counterfactual Plan Repair (RAP MCTS + AdaPlanner Refine-Then-Resume)
Upon `OutOfPlanInterruption` at step $k$:
1. Preserve execution context $n_{1 \dots k-1}$ and freeze variable bindings up to breakpoint $k$.
2. Invoke `PlanRepairEngine`, which executes MCTS search over Front 04 counterfactual rollouts (`counterfactual.py` virtual sandbox):
   - **Selection**: UCT formula balancing exploration vs expected utility gain:
     $$UCT(s, a) = Q(s, a) + c_{\text{puct}} P(a \mid s) \frac{\sqrt{N(s)}}{1 + N(s, a)}$$
   - **Local Sub-graph Patching (Self-PR / HTN CFG Repair)**: Generate local candidate patch sequence $\Pi_{\text{patch}}$ substituting failing nodes $n_{k \dots k+m}$.
   - **Evaluation**: Score patch candidate using game-theoretic expected utility gain $E[\Delta u] - C_{\text{patch}}$.
3. Resume execution from breakpoint $k$ using $\Pi_{\text{repaired}} = n_{1 \dots k-1} \mathbin{\Vert} \Pi_{\text{patch}}$.

### 2.4 Reusable Strategy Library (`PlanLibrary` - Voyager Synthesis)
1. **Skill Discovery**: Successful plan executions with net utility gain $\Delta g / \text{cost} \ge \theta_{\text{skill}}$ are abstracted into strategy templates $T_\Pi$.
2. **Skill Filtering**: Candidate templates undergo ablation testing (performance evaluated with vs. without exemplar) before archiving to prevent episode-specific overfitting (AdaPlanner §3.3).
3. **YAML Serialization (Rule 006 & Rule 007)**: Archived plan templates are saved in machine-readable YAML (`schema/plans/`). Under Rule 007, if YAML serialization creates a profiling bottleneck during active simulation runs, prompt the user for binary/compressed IPC format.

---

## 3. Core Module Specifications (`src/hypostases/planning/`)

1. [`plan_types.py`](../../../src/hypostases/planning/plan_types.py): Data classes for `PlanNode`, `ContingencyBranch`, `PlanStatus`, and `Plan`.
2. [`plan_executor.py`](../../../src/hypostases/planning/plan_executor.py): Closed-loop step executor, precondition checker, state deviation monitor, and in-plan info parser.
3. [`plan_repair.py`](../../../src/hypostases/planning/plan_repair.py): Out-of-plan refiner, RAP MCTS search oracle integrated with `counterfactual.py`, and refine-then-resume breakpoint manager.
4. [`plan_library.py`](../../../src/hypostases/planning/plan_library.py): Voyager-style strategy library indexing, A* goal-oriented template matching, skill acquisition, and YAML serialization loader.

---

## 4. Invariant & Safety Guarantees (Rule 005 & Rule 006/007)

- **Rule 005**: All planning choices, branch evaluations, and plan repair selections maximize formal game-theoretic utility $E[\Delta u] - C_{\text{repair}}$. Zero artificial human biases or emotional irrationality.
- **Rule 006**: All default configurations reside in `schema/planning_config.yaml`.
- **Rule 007**: YAML plan persistence format is monitored for performance tax during simulation; if profiling detects bottlenecks, prompt user for compressed format.
