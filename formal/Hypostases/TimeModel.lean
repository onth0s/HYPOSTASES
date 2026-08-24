import Mathlib.Order.Basic
import Mathlib.Data.Real.Basic

/-!
# HYPOSTASES Formal Specification — Part I (§1): 3-Tier Time Model
Formalization of asynchronous local event clocks, causal ordering, and epoch snapshots.
-/

namespace Hypostases

/-- Agent identifier in index set I -/
structure AgentId where
  id : ℕ
deriving DecidableEq, Repr

/-- A discrete asynchronous Tier-1 event on an agent's local clock -/
structure Tier1Event where
  agent : AgentId
  event_index : ℕ
  timestamp : ℝ

/-- Strict local event ordering on an agent's clock: t^(i)_0 < t^(i)_1 < t^(i)_2 < ... -/
def strictly_increasing_clock (clock : ℕ → Tier1Event) (a : AgentId) : Prop :=
  (∀ k, (clock k).agent = a) ∧
  (∀ k, (clock k).event_index = k) ∧
  (∀ k, (clock k).timestamp < (clock (k + 1)).timestamp)

/-- Global total ordering with strict tie-breaking for asynchronous event attribution -/
def global_event_order (e1 e2 : Tier1Event) : Prop :=
  e1.timestamp < e2.timestamp ∨ (e1.timestamp = e2.timestamp ∧ e1.agent.id < e2.agent.id)

/-- Theorem: Transitivity of causal event total order under strictly increasing timestamps. -/
theorem global_order_trans_time {e1 e2 e3 : Tier1Event}
    (h12 : e1.timestamp < e2.timestamp) (h23 : e2.timestamp < e3.timestamp) :
    e1.timestamp < e3.timestamp := by
  exact lt_trans h12 h23

/-- Tier-2 Synchronous Snapshot Barrier -/
structure EpochSnapshot (n : ℕ) where
  epoch_time : ℝ
  event_barrier : Tier1Event → Prop
  causal_barrier : ∀ e, event_barrier e → e.timestamp ≤ epoch_time

end Hypostases
