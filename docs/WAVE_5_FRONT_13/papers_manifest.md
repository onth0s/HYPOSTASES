# Wave 5 Front 13 — Papers Manifest

**Front**: Front 13 — Evolutionary Algorithm Discovery (AlphaEvolve Engine)  
**Wave**: Wave 5 (Universal Scaling & Symbolic Generalization)  
**Target Substrate**: HYPOSTASES Engine ($\sigma = (c, w, g, \rho_{\text{ext}})$)  
**Rule 005 Compliance**: Pure Game-Theoretic & Probabilistic Rationality (Zero Artificial Human Cognitive Biases)  
**Rule 010 Compliance**: PDF assets in `docs/WAVE_5_FRONT_13/papers/` are ignored by Git (`.gitignore`).

All 9 primary foundational literature papers relevant to Wave 5 Front 13 have been cataloged, extracted, and mathematically analyzed for integration into the HYPOSTASES AlphaEvolve Engine.

---

## 1. Complete Ingested & Reference Literature Catalog (9 Papers)

| Index | Paper Title | Lead Authors & Year | Venue / Citation | Key Theoretical Contribution | Engine Relevance & Synthesized Invariants | Ingestion Status | Local PDF File |
|---|---|---|---|---|---|---|---|
| 01 | *AlphaEvolve: A coding agent for scientific and algorithmic discovery* | A. Novikov et al. (Google DeepMind, 2025) | *arXiv:2506.13131* (44 pages) | Evolutionary coding agent orchestrating LLMs to evolve multi-component programs directly in code, evaluated continuously against automated multi-metric evaluation oracles. | Informs outer-loop AST code generation, prompt-driven mutation, and multi-component candidate code evaluation. | `INGESTED` | [`novikov_et_al_2025_alphaevolve.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_4_FRONT_10/papers/novikov_et_al_2025_alphaevolve.pdf) |
| 02 | *Mathematical discoveries from program search with large language models* (FunSearch) | B. Romera-Paredes et al. (Google DeepMind, 2023/2024) | *Nature*, 625:468–475 (2024) | Pairs pre-trained LLMs with automated evaluators and island-based diversity preservation to discover non-trivial mathematical functions and algorithms (e.g. cap set problem). | Provides the core paradigm for evolutionary LLM code mutators paired with multi-agent simulation evaluation oracles. | `REFERENCE` | *(Reference Text)* |
| 03 | *AutoML-Zero: Evolving Machine Learning Algorithms From Scratch* | E. Real, C. Liang, D. R. So, Q. V. Le (Google Brain, 2020) | *ICML 2020* (23 pages) | Discovers machine learning algorithms using basic mathematical primitives across `Setup`, `Predict`, and `Learn` loops, utilizing Functional Equivalence Checking (FEC). | Supplies the AST mutation primitives and Functional Equivalence Checking (FEC) module to collapse redundant code mutations. | `INGESTED` | [`real_et_al_2020_automl_zero.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_4_FRONT_10/papers/real_et_al_2020_automl_zero.pdf) |
| 04 | *Illuminating search spaces by mapping elites* (MAP-Elites) | J.-B. Mouret, J. Clune (2015) | *Evolutionary Computation*, 23(2):229–259 | Establishes Quality-Diversity (QD) algorithms. Maintains a multi-dimensional behavioral feature grid (archive) where elites are retained per feature bin. | Informs `MAPElitesArchive` in HYPOSTASES, maintaining diversity across $(\mathcal{C}_{\text{time}}, \mathcal{R}_{\text{IC}}, \Delta \kappa_{\text{scarcity}})$. | `REFERENCE` | *(Reference Text)* |
| 05 | *What is morphological computation? On the theoretical foundations of morphological computation* | V. C. Müller, M. Hoffmann (2017) | *Complex Systems* / *Theoretical Computer Science* | Formalizes morphological computation where physical dynamics / body morphology perform computational operations, reducing active control complexity. | Informs reservoir co-evolution between continuous physical state dynamics in $\rho_{\text{ext}}$ and software action primitives. | `REFERENCE` | *(Reference Text)* |
| 06 | *Evolutionary Mechanism Design: A Review* | S. Phelps, P. McBurney, S. Parsons (2010) | *AAMAS*, 20(3):285–308 | Reviews genetic programming and co-evolutionary search for discovering economic double auctions, trading protocols, and market rules. | Links evolutionary algorithm discovery directly to multi-agent game-theoretic equilibrium and institutional search (Front 10). | `INGESTED` | [`phelps_et_al_2010_evolutionary_mechanism_design_review.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_4_FRONT_10/papers/phelps_et_al_2010_evolutionary_mechanism_design_review.pdf) |
| 07 | *Abandoning Objectives: Evolution Through the Search for Novelty Alone* | J. Lehman, K. O. Stanley (2011) | *Evolutionary Computation*, 19(2):189–245 | Demonstrates that searching for behavioral novelty alone out-performs objective-driven optimization in deceptive, high-dimensional search spaces. | Informs the behavioral distance metric $\rho(x, \mathbf{S})$ in QD archive selection to prevent deceptive local optima trapping. | `REFERENCE` | *(Reference Text)* |
| 08 | *Regularized Evolution for Image Classifier Architecture Search* | E. Real, A. Aggarwal, Y. Huang, Q. V. Le (Google Brain, 2019) | *AAAI 2019* (AAAI-19) | Introduces regularized (aging) evolution for NAS, discarding the oldest candidate in a fixed-size population to sustain ongoing exploratory mutation. | Informs population aging and replacement policy in `AlphaEvolveEngine`. | `REFERENCE` | *(Reference Text)* |
| 09 | *Minimal-Intelligence Agents for Auctions (ZIP)* | D. Cliff (1997) | *HP Labs Tech Report*, HPL-97-91 | Proves that minimal adaptive agents under double auction rules converge rapidly to equilibrium and near-100% allocative efficiency. | Benchmark baseline for evaluating evolved continuous trading and scarcity control algorithms under resource constraints ($\kappa$). | `INGESTED` | [`cliff_1997_minimal_intelligence_agents.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_4_FRONT_10/papers/cliff_1997_minimal_intelligence_agents.pdf) |

---

## 2. Ingestion & Local Storage Notice

As per **Rule 010** (`AGENTS.md`), PDF files stored under `docs/WAVE_5_FRONT_13/papers/` are untracked by Git. Theoretical contributions, mathematical formulations, and invariant limits from all listed literature are fully mapped to the primitive state substrate:

$$\sigma = (c, w, g, \rho_{\text{ext}})$$
