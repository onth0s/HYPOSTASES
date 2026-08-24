import Lake
open Lake DSL

package "hypostases_formal" where
  version := v!"0.1.0"
  keywords := #["math", "ai", "active-inference", "game-theory"]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.13.0"

@[default_target]
lean_lib «Hypostases» where
  srcDir := "."
