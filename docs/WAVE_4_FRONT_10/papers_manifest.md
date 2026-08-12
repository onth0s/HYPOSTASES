# Wave 4 Front 10 — Mechanism Search Layer: Paper Manifest

**Front**: Front 10 — Mechanism Search  
**Wave**: Wave 4 (Recursive Adaptation & Discovery Loops)  
**Target Substrate**: HYPOSTASES Engine ($\sigma = (c, w, g, \rho_{\text{ext}})$)  
**Rule 005 Invariant**: Pure Game-Theoretic & Probabilistic Rationality (Zero Artificial Human Cognitive Biases)  
**Ingestion Directory**: [`papers/`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_4_FRONT_10/papers/)

All 12 primary foundational papers in the `docs/WAVE_4_FRONT_10/papers/` directory have been fully ingested, extracted, and mathematically analyzed for Front 10 integration. In compliance with Rule 010, PDF assets are untracked in Git (`.gitignore`).

---

## 1. Complete Ingested Literature Catalog (12 Papers)

### 1. Clarke (1971)
- **Title**: *Multipart Pricing of Public Goods*
- **Author**: Edward H. Clarke
- **Venue/Year**: *Public Choice*, 11:17–33, 1971
- **File**: [`clarke_1971_multipart_pricing_public_goods.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_4_FRONT_10/papers/clarke_1971_multipart_pricing_public_goods.pdf)
- **Key Theoretical Contribution**: Introduces the pivot tax mechanism / Clarke tax mechanism (part of the VCG family). Eliminates free-rider incentives by assigning assigned marginal supply price schedules and charging each participant a variable tax equal to the externality they impose on others.
- **Engine Relevance**: Provides the foundational public goods incentive-compatibility invariant ($\mathcal{R}_{\text{IC}}(\mu_{\text{Clarke}}) = 0$) for multi-agent institutional resource allocation in HYPOSTASES.

---

### 2. Cliff (1997)
- **Title**: *Minimal-Intelligence Agents for Auctions (ZIP)*
- **Author**: Dave Cliff
- **Venue/Year**: *Hewlett-Packard Labs Technical Report*, HPL-97-91, 1997 (134 pages)
- **File**: [`cliff_1997_minimal_intelligence_agents.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_4_FRONT_10/papers/cliff_1997_minimal_intelligence_agents.pdf)
- **Key Theoretical Contribution**: Establishes Zero Intelligence Plus (ZIP) agents operating in continuous double auctions (CDA). Proves that simple adaptive agents under double auction market rules achieve rapid equilibrium price convergence and near-100% allocative efficiency.
- **Engine Relevance**: Supplies baseline market dynamics and continuous double auction rules for multi-agent resource exchanges.

---

### 3. Conitzer & Sandholm (2002, 2004)
- **Title**: *Complexity of Mechanism Design* / *Automated Mechanism Design: Concepts and Results*
- **Authors**: Vincent Conitzer, Tuomas Sandholm
- **Venue/Year**: *UAI 2002* / *Journal of Artificial Intelligence Research (JAIR)* 2004
- **File**: [`conitzer_sandholm_2002_complexity_of_mechanism_design.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_4_FRONT_10/papers/conitzer_sandholm_2002_complexity_of_mechanism_design.pdf)
- **Key Theoretical Contribution**: Formulates Automated Mechanism Design (AMD) as an explicit optimization problem over outcome allocations and payment functions. Proves NP-completeness for deterministic mechanism search without side payments, while showing that randomized/probabilistic mechanisms render the design problem polynomial-time solvable via linear programming.
- **Engine Relevance**: Defines the classical AMD optimization formulation over mechanism spaces $\mathcal{M}$ and randomized policy relaxations.

---

### 4. Dütting, Feng, Narasimhan, Parkes, & Ravindranath (2019, 2024)
- **Title**: *Optimal Auctions through Deep Learning* (RegretNet)
- **Authors**: Paul Dütting, Zhe Feng, Harikrishna Narasimhan, David C. Parkes, Sai Srivatsa Ravindranath
- **Venue/Year**: *ICML 2019* / *Operations Research* 2024 (55 pages)
- **File**: [`duetting_et_al_2019_optimal_auctions_deep_learning.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_4_FRONT_10/papers/duetting_et_al_2019_optimal_auctions_deep_learning.pdf)
- **Key Theoretical Contribution**: Establishes Differentiable Economics. Parameterizes multi-item auctions as end-to-end multi-layer neural networks (RegretNet) and utilizes augmented Lagrangian optimization to minimize empirical IC regret $\mathcal{R}_{\text{IC}}(\mu)$ alongside negated revenue.
- **Engine Relevance**: Directly informs the `DifferentiableMechanismSearcher` module in HYPOSTASES using augmented Lagrangian IC penalization.

---

### 5. Groves (1973)
- **Title**: *Incentives in Teams*
- **Author**: Theodore Groves
- **Venue/Year**: *Econometrica*, 41(4):617–631, 1973
- **File**: [`groves_1973_incentives_in_teams.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_4_FRONT_10/papers/groves_1973_incentives_in_teams.pdf)
- **Key Theoretical Contribution**: Proves that the general Groves mechanism family ($P_i(b) = h_i(b_{-i}) - \sum_{j \neq i} v_j(x(b))$) guarantees dominant strategy incentive compatibility (DSIC) for social welfare maximization in team decision-making.
- **Engine Relevance**: Establishes the exact mathematical baseline for team incentive alignment in multi-agent institutions.

---

### 6. Myerson (1981)
- **Title**: *Optimal Auction Design*
- **Author**: Roger B. Myerson
- **Venue/Year**: *Mathematics of Operations Research*, 6(1):58–73, 1981
- **File**: [`myerson_1981_optimal_auction_design.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_4_FRONT_10/papers/myerson_1981_optimal_auction_design.pdf)
- **Key Theoretical Contribution**: Derives single-item revenue-optimal auction mechanisms via virtual valuation transformations $\psi_i(v_i) = v_i - \frac{1 - F_i(v_i)}{f_i(v_i)}$. Allocates to the bidder with highest non-negative virtual valuation.
- **Engine Relevance**: Serves as the ground-truth benchmark for formal math verification in `tests/formal_math/test_mechanism_search_formal.py`.

---

### 7. Novikov et al. / DeepMind (2025)
- **Title**: *AlphaEvolve: A coding agent for scientific and algorithmic discovery*
- **Authors**: Alexander Novikov, Ngan Vu, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, Matej Balog
- **Venue/Year**: *arXiv:2506.13131* (44 pages), June 2025
- **File**: [`novikov_et_al_2025_alphaevolve.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_4_FRONT_10/papers/novikov_et_al_2025_alphaevolve.pdf)
- **Key Theoretical Contribution**: Evolutionary coding agent orchestrating LLMs to evolve multi-component programs directly in code evaluated continuously against automated evaluation oracles across multiple metrics simultaneously.
- **Engine Relevance**: Informs the black-box simulation harness evaluation oracle loop and code-level AST evolutionary rule mutation in HYPOSTASES.

---

### 8. Phelps, McBurney, & Parsons (2010)
- **Title**: *Evolutionary Mechanism Design: A Review*
- **Authors**: Steve Phelps, Peter McBurney, Simon Parsons
- **Venue/Year**: *Autonomous Agents and Multi-Agent Systems*, 2010 (31 pages)
- **File**: [`phelps_et_al_2010_evolutionary_mechanism_design_review.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_4_FRONT_10/papers/phelps_et_al_2010_evolutionary_mechanism_design_review.pdf)
- **Key Theoretical Contribution**: Reviews evolutionary algorithm approaches (genetic programming, co-evolutionary search, AST rule mutations) to discover trading protocols and double auction mechanisms when analytical game theory is intractable.
- **Engine Relevance**: Directly informs the `EvolutionaryMechanismSearcher` module for discrete institutional AST search.

---

### 9. Real et al. / Google Brain (2020)
- **Title**: *AutoML-Zero: Evolving Machine Learning Algorithms From Scratch*
- **Authors**: Esteban Real, Chen Liang, David R. So, Quoc V. Le
- **Venue/Year**: *ICML 2020* (23 pages)
- **File**: [`real_et_al_2020_automl_zero.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_4_FRONT_10/papers/real_et_al_2020_automl_zero.pdf)
- **Key Theoretical Contribution**: Discovers ML algorithms using component functions (`Setup`, `Predict`, `Learn`) mutated via basic mathematical operations on small memory, evaluated against simulation proxy tasks using functional equivalence checking (FEC).
- **Engine Relevance**: Informs functional equivalence checking and proxy evaluation tasks in evolutionary mechanism search.

---

### 10. Roughgarden (2010, 2014)
- **Title**: *Algorithmic Game Theory Lecture Notes*
- **Author**: Tim Roughgarden
- **Venue/Year**: *Stanford University Technical Report*, 2014
- **File**: [`roughgarden_2014_agt_lecture_notes.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_4_FRONT_10/papers/roughgarden_2014_agt_lecture_notes.pdf)
- **Key Theoretical Contribution**: Covers Price of Anarchy (PoA), Price of Stability (PoS), robust mechanism design, and smooth games bounds on multi-agent welfare loss under strategic uncertainty.
- **Engine Relevance**: Supplies stability and welfare loss bounds for evaluating candidate mechanisms.

---

### 11. Vickrey (1961)
- **Title**: *Counterspeculation, Auctions, and Competitive Sealed Tenders*
- **Author**: William Vickrey
- **Venue/Year**: *Journal of Finance*, 16(1):8–37, 1961
- **File**: [`vickrey_1961_counterspeculation_auctions.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_4_FRONT_10/papers/vickrey_1961_counterspeculation_auctions.pdf)
- **Key Theoretical Contribution**: Seminal paper introducing second-price sealed-bid auctions (Vickrey auctions) where truthful bidding is a dominant strategy ($P_i = \max_{j \neq i} b_j$).
- **Engine Relevance**: Baseline invariant $\mathcal{R}_{\text{IC}}(\mu_{\text{Vickrey}}) \equiv 0$ in formal math verification.

---

### 12. Zheng, Trott, Srinivasa, Naik, Gruesbeck, Parkes, & Socher (2020, 2022)
- **Title**: *The AI Economist: Tax Policy Design via Two-Level Deep Reinforcement Learning*
- **Authors**: Stephan Zheng, Alexander Trott, Sunil Srinivasa, Nikhil Naik, Melvin Gruesbeck, David C. Parkes, Richard Socher
- **Venue/Year**: *Science Advances*, 8(17), 2022 / arXiv:2004.13332 (46 pages)
- **File**: [`zheng_et_al_2020_ai_economist.pdf`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_4_FRONT_10/papers/zheng_et_al_2020_ai_economist.pdf)
- **Key Theoretical Contribution**: Applies bi-level multi-agent reinforcement learning to search over economic tax-subsidy schedules $\tau(y)$, demonstrating emergent economic equilibrium under non-stationary multi-agent learning.
- **Engine Relevance**: Directly informs the bi-level simulation harness oracle evaluation loop and social welfare metrics (Productivity $\times$ Equality).

---

## 2. Ingestion Summary

- **Total Papers Cataloged & Extracted**: 12
- **PDF Assets Directory**: `docs/WAVE_4_FRONT_10/papers/`
- **Git Tracking Status**: Ignored in Git (`.gitignore`) per **Rule 010**.
