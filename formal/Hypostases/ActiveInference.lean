import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset

/-!
# HYPOSTASES Formal Specification — Active Perception & Friston EFE
Formalization of Variational Free Energy bounds and Pragmatic/Epistemic value decomposition.
-/

namespace Hypostases

variable {n : ℕ} [NeZero n]

/-- Discrete probability distribution over Fin n -/
structure DiscreteDist (n : ℕ) where
  p : Fin n → ℝ
  p_pos : ∀ i, 0 < p i
  sum_one : Finset.sum Finset.univ p = 1

/-- Kullback-Leibler Divergence: D_KL(q || p) = ∑ q(x) * ln(q(x) / p(x)) -/
noncomputable def kl_divergence (q p : DiscreteDist n) : ℝ :=
  Finset.sum Finset.univ (fun i => q.p i * Real.log (q.p i / p.p i))

/-- Variational Free Energy: F[q, y] = D_KL(q(s) || p(s|y)) - ln p(y) -/
noncomputable def variational_free_energy (q : DiscreteDist n) (prior_joint : DiscreteDist n) (log_evidence : ℝ) : ℝ :=
  kl_divergence q prior_joint - log_evidence

/-- Expected Free Energy (EFE) Action Objective: G(π) = Pragmatic Value + Epistemic Value -/
structure EFEComponents where
  pragmatic_value : ℝ  -- Expected log preference: E_{q(o|π)} [ln P(o)]
  epistemic_value : ℝ  -- Expected information gain: E_{q(o|π)} [D_KL(q(s|o,π) || q(s|π))]
  beta_efe : ℝ         -- Learned epistemic weight parameter (θ_meta[9])

/-- Combined utility under EFE mode vs. linear fallback mixing -/
def total_utility (comp : EFEComponents) : ℝ :=
  (1 - comp.beta_efe) * comp.pragmatic_value + comp.beta_efe * comp.epistemic_value

/-- Theorem: When beta_efe = 0, total utility recovers pure pragmatic exploitation. -/
theorem total_utility_pure_pragmatic (comp : EFEComponents) (h : comp.beta_efe = 0) :
    total_utility comp = comp.pragmatic_value := by
  dsimp [total_utility]
  rw [h]
  ring

/-- Theorem: When beta_efe = 1, total utility recovers pure epistemic exploration (Curiosity). -/
theorem total_utility_pure_epistemic (comp : EFEComponents) (h : comp.beta_efe = 1) :
    total_utility comp = comp.epistemic_value := by
  dsimp [total_utility]
  rw [h]
  ring

end Hypostases
