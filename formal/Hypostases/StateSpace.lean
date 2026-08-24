import Mathlib.Data.Real.Basic
import Mathlib.Topology.Instances.Real
import Mathlib.Probability.ProbabilityMassFunction.Basic

/-!
# HYPOSTASES Formal Specification — Part I (§2): Typed State Spaces
Formalization of Primitive and Derived Component Spaces.
-/

namespace Hypostases

/-- Dimensions for state space components. -/
structure StateDim where
  nc : ℕ       -- Characteristics dimension
  nk : ℕ       -- Goal category count (Survival, Curiosity, etc.)
  nr_ext : ℕ   -- External resources dimension (capital, time, authority)
  nr_int : ℕ   -- Internal resources dimension (energy, physical stamina)
  ne : ℕ       -- Global environment dimension
  h_int_le : nr_int ≤ nc -- Internal resources are a sub-projection of Characteristics

variable (d : StateDim)

/-- Primitive: Characteristics vector c ∈ C = ℝ^{n_c} -/
def Characteristics := Fin d.nc → ℝ

/-- Primitive: Goal utility weights g = u ∈ G = ℝ^{n_k} -/
def GoalUtilities := Fin d.nk → ℝ

/-- Primitive: External resources ρ_ext ∈ R_ext = ℝ≥0^{n_r} -/
def ExternalResources := Fin d.nr_ext → { r : ℝ // 0 ≤ r }

/-- Derived: Dynamic policy allocation simplex π ∈ Δ(K) over goal categories -/
structure PolicySimplex where
  prob : Fin d.nk → ℝ
  nonneg : ∀ i, 0 ≤ prob i
  sum_one : (Finset.univ.sum prob) = 1

/-- Derived: Internal Power projection ρ_int = proj_int(c) ∈ ℝ≥0^{n_{r,int}} -/
def proj_int (c : Characteristics d) (proj_map : Fin d.nr_int → Fin d.nc) :
    Fin d.nr_int → ℝ :=
  fun i => max 0 (c (proj_map i))

/-- Theorem: Internal power projection guarantees non-negativity. -/
theorem proj_int_nonneg (c : Characteristics d) (proj_map : Fin d.nr_int → Fin d.nc) (i : Fin d.nr_int) :
    0 ≤ proj_int d c proj_map i := by
  dsimp [proj_int]
  exact le_max_left 0 (c (proj_map i))

/-- Theorem: Simplex elements are bounded in [0, 1]. -/
theorem simplex_elem_le_one (π : PolicySimplex d) (i : Fin d.nk) : π.prob i ≤ 1 := by
  have h_sum : π.prob i + Finset.sum (Finset.univ.erase i) π.prob = 1 := by
    rw [← π.sum_one]
    exact (Finset.add_sum_erase Finset.univ π.prob (Finset.mem_univ i)).symm
  have h_others_nonneg : 0 ≤ Finset.sum (Finset.univ.erase i) π.prob := by
    apply Finset.sum_nonneg
    intro j _
    exact π.nonneg j
  linarith

end Hypostases
