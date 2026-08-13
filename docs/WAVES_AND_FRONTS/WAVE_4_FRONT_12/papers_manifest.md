# Wave 4 Front 12 — Papers Manifest

**Front**: Front 12 — Scientific Discovery Loop  
**Wave**: Wave 4 (Recursive Adaptation & Discovery Loops)  
**Target Substrate**: HYPOSTASES Engine ($\sigma = (c, w, g, \rho_{\text{ext}}$)  
**Rule 010 Compliance**: PDF files in `docs/WAVE_4_FRONT_12/papers/` are ignored by Git.

---

## 1. Papers Manifest Table

| Index | Paper Title | Lead Authors & Year | Venue / Citation | Key Relevance & Synthesized Findings | Ingestion Status | Local PDF File |
|---|---|---|---|---|---|---|
| 01 | *On a Measure of the Information Provided by an Experiment* | D. V. Lindley (1956) | *Ann. Math. Statist.* 27(4):986-1005 | Establishes information-theoretic Bayesian Optimal Experimental Design (BED). Defines Expected Information Gain (EIG) via Shannon entropy reduction: $I(\mathcal{E}, p(\theta)) = \mathbb{E}_x \left[ \int p(\theta \mid x) \log p(\theta \mid x) d\theta - \int p(\theta) \log p(\theta) d\theta \right]$. | `INGESTED` | [`Lindley_1956.pdf`](../../WAVE_4_FRONT_12/papers/Lindley_1956.pdf) |
| 02 | *Information-Based Objective Functions for Active Data Selection* | D. J. C. MacKay (1992) | *Neural Computation* 4(4):590-604 | Applies Bayesian learning to active data selection / active learning. Maximizes EIG to choose measurements where predictive uncertainty (variance) is highest ($\Delta S = \frac{1}{2} \Delta \log \sigma_u^2$). | `INGESTED` | [`MacKay_1992.pdf`](../../WAVE_4_FRONT_12/papers/MacKay_1992.pdf) |
| 03 | *Causality: Models, Reasoning, and Inference* | J. Pearl (2000 / 2009) | Cambridge University Press | Structural Causal Models (SCMs) and interventional experiment design ($do$-calculus), isolating causal directionality from observational equivalence classes. | `INGESTED` | *(Reference Text)* |
| 04 | *Functional Genomic Hypothesis Generation and Experimentation by a Robot Scientist* | R. D. King et al. (2004) | *Nature* 427(6971):247-252 | First autonomous physical scientific discovery loop ("Adam"). Uses abductive logic to generate hypotheses and Active Selection of Experiments (ASE) to minimize expected cost under budget constraints $\rho_{\text{ext}}$. | `INGESTED` | [`Functional_genomic_hypothesis_generation.pdf`](../../WAVE_4_FRONT_12/papers/Functional_genomic_hypothesis_generation.pdf) |
| 05 | *The Automation of Science* | R. D. King et al. (2009) | *Science* 324(5923):85-89 | Philosophical and algorithmic principles for automated scientific discovery loops, cost-minimizing active learning, and formal logical model refinement. | `INGESTED` | *(Reference Text)* |
| 06 | *Deep Adaptive Design: Amortizing Sequential Bayesian Experimental Design* | A. Foster et al. (2020) | *AISTATS / NeurIPS* 2020 | Introduces stochastic gradient BOED with Adaptive Contrastive Estimation (ACE) lower bound ($I_{\text{ACE}}(\xi, \phi, L) = \mathbb{E} \left[ \log \frac{p(y \mid \theta_0, \xi)}{\frac{1}{L+1}\sum_{\ell=0}^L p(y \mid \theta_\ell, \xi)} \right]$), amortizing sequential experimental design across complex state spaces. | `INGESTED` | [`Foster_2020.pdf`](../../WAVE_4_FRONT_12/papers/Foster_2020.pdf) |
| 07 | *The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery* | C. Lu et al. (2024) | *arXiv:2408.06292* | Open-ended AI scientific discovery pipeline: idea generation, automated experiment execution (Aider iteration), data visualization, paper write-up, and automated peer review scoring. | `INGESTED` | [`Lu_2024.pdf`](../../WAVE_4_FRONT_12/papers/Lu_2024.pdf) |
| 08 | *Co-Scientist: Autonomous Hypothesis Generation and Evolutionary Tournaments* | J. Gottweis et al. (2025) | *Google DeepMind / arXiv* | Multi-agent architecture (Generation, Reflection, Ranking, Evolution) using Elo-based evolutionary tournaments to rank and evolve candidate hypotheses through structured scientific debate. | `INGESTED` | [`Gottweis_2025.pdf`](../../WAVE_4_FRONT_12/papers/Gottweis_2025.pdf) |
| 09 | *Discovery Loop: High-Throughput Autonomous Scientific Experimentation* | J. Dean et al. (2026) | *Discovery Loop Inc.* | Closed-loop automated physical experimentation systems accelerating scientific discovery cycles from weeks to hours across biology, chemistry, and physics. | `INGESTED` | *(Reference Text)* |

---

## 2. Ingestion & Local Storage Notice

As per **Rule 010** (`AGENTS.md`), PDF files in `docs/WAVE_4_FRONT_12/papers/` are untracked by Git. All 6 local PDF papers are ingested and mapped to the HYPOSTASES state substrate $\sigma = (c, w, g, \rho_{\text{ext}})$.
