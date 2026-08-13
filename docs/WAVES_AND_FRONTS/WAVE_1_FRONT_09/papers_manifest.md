# Wave 1 Front 09 — Active Information Gathering & Active Perception: Paper Manifest

**Front**: Front 09 — Active Information Gathering & Active Perception  
**Wave**: Wave 1 (Single-Agent Foundations: Memory, Lookahead & Active Sensing)  
**Target Substrate**: HYPOSTASES Engine ($\sigma = (c, w, g, \rho_{\text{ext}})$)  

---

## 1. Active Inference & Variational Free Energy

### 1. Dodig-Crnkovic (2022)
- **Title**: *Cognition as Morphological/Morphogenetic Embodied Computation In Vivo*
- **Authors**: Gordana Dodig-Crnkovic
- **Journal/Venue**: *Entropy*, 24(11), 1576 (2022)
- **Key Concepts**: Info-computational paradigm, variational free energy minimization $F = D_{KL}(q(w) \parallel p(w)) - \mathbb{E}_q[\ln p(o \mid w)]$, embodied physical/abstract computation.
- **Engine Relevance**: Direct theoretical foundation for `compute_variational_free_energy()` in [`epistemic_utility.py`](../../../src/hypostases/epistemic_utility.py).

### 2. Friston et al. (2015)
- **Title**: *Active Inference, Curiosity and Insight*
- **Authors**: Karl Friston, Francesco Rigoli, Domenico Ognibene, Christian Mathys, Thomas FitzGerald, Giovanni Pezzulo
- **Journal/Venue**: *Neural Computation*, 29(10), 2633–2683 (2017)
- **Key Concepts**: Expected free energy (EFE), epistemic affordances, ambiguity resolution vs. pragmatic risk minimization.
- **Engine Relevance**: Mathematical basis for trading material reserve cost against information variance reduction $\Delta \sigma^2$.

---

## 2. Classic Active Perception & Sensing Affordances

### 3. Bajcsy (1988)
- **Title**: *Active Perception*
- **Authors**: Ruzena Bajcsy
- **Journal/Venue**: *Proceedings of the IEEE*, 76(8), 996–1005 (1988)
- **Key Concepts**: Perception as an active control process, intentional exploratory probing, task-driven sensor allocation.
- **Engine Relevance**: Conceptual origin of explicit epistemic action primitives (`INSPECT`, `PROBE`, `MONITOR`, `SPY`).

### 4. Aloimonos et al. (1988)
- **Title**: *Active vision*
- **Authors**: John Aloimonos, Isaac Weiss, Amit Bandyopadhyay
- **Journal/Venue**: *International Journal of Computer Vision*, 1(4), 333–356 (1988)
- **Key Concepts**: Reformulating ill-posed passive inverse problems into well-posed active sensing optimizations.
- **Engine Relevance**: Formal justification for state-dependent observation precision updates.

---

## 3. Bayesian Experimental Design & Information Theory

### 5. Lindley (1956)
- **Title**: *On a Measure of the Information Provided by an Experiment*
- **Authors**: Dennis V. Lindley
- **Journal/Venue**: *The Annals of Mathematical Statistics*, 27(4), 986–1005 (1956)
- **Key Concepts**: Expected Kullback-Leibler information gain, decision-theoretic sample selection, Shannon entropy over parameter distributions.
- **Engine Relevance**: Foundation of `compute_expected_information_gain()` and Shannon entropy $\Delta H(w) = \frac{1}{2} \ln\left(1 + \frac{\sigma^2_{\text{prior}}}{\sigma^2_{\text{obs}}}\right)$.

### 6. MacKay (1992)
- **Title**: *Information-Based Objective Functions for Active Data Selection*
- **Authors**: David J. C. MacKay
- **Journal/Venue**: *Neural Computation*, 4(4), 590–604 (1992)
- **Key Concepts**: Entropy minimization over latent parameter spaces, active data selection, Bayesian optimal experimental design.
- **Engine Relevance**: Formalization of variance reduction metrics in Bayesian belief state updates.

---

## 4. Intrinsic Motivation, Curiosity & Active Learning

### 7. Schmidhuber (2010)
- **Title**: *Formal Theory of Creativity, Fun, and Intrinsic Motivation (1990–2010)*
- **Authors**: Jürgen Schmidhuber
- **Journal/Venue**: *IEEE Transactions on Autonomous Mental Development*, 2(3), 230–247 (2010)
- **Key Concepts**: Information compression progress, artificial curiosity, intrinsic reward allocation for active exploration.
- **Engine Relevance**: Grounding for epistemic utility weight $\beta$ in total utility optimization $U_{\text{total}} = (1-\beta) U_{\text{pragmatic}} + \beta U_{\text{epistemic}}$.

### 8. Oudeyer, Kaplan, & Hafner (2007)
- **Title**: *Intrinsic Motivation Systems for Autonomous Mental Development*
- **Authors**: Pierre-Yves Oudeyer, Frédéric Kaplan, Verena V. Hafner
- **Journal/Venue**: *IEEE Transactions on Evolutionary Computation*, 11(2), 265–286 (2007)
- **Key Concepts**: Intelligent Adaptive Curiosity (IAC), autonomous learning progress maximization, exploratory action selection.
- **Engine Relevance**: Dynamic allocation of exploration parameters across epistemic action choices.

### 9. Settles (2009)
- **Title**: *Active Learning Literature Survey*
- **Authors**: Burr Settles
- **Journal/Venue**: *University of Wisconsin-Madison Computer Sciences Technical Report 1648* (2009)
- **Key Concepts**: Comprehensive taxonomy of uncertainty sampling, query-by-committee, expected model change, and information-density querying.
- **Engine Relevance**: Blueprint for structured epistemic action modalities (`QUERY`, `EXPERIMENT`, `VERIFY`).

---

## 5. Multi-Agent & Epistemic Probing under Uncertainty

### 10. Yang, Zhang, & Huang (2023)
- **Title**: *Active Information Gathering for Autonomous Multi-Agent Systems under Uncertainty*
- **Authors**: Yiannis Yang, Ji Zhang, Xiaoming Huang
- **Journal/Venue**: *IEEE Transactions on Robotics*, 39(4), 2890–2908 (2023)
- **Key Concepts**: Trading material trajectory/time cost for information gain in multi-agent belief sharing and environment probing.
- **Engine Relevance**: Multi-agent interaction mechanics for `QUERY` and `SPY` epistemic action types in HYPOSTASES.
- **Status**: UNVERIFIED — no IEEE T-RO Vol. 39 (2023) paper matches this title/authors/venue; **NOT DOWNLOADED**. Consider replacing with a verified multi-agent active information gathering paper (e.g., Lauri et al., *Multi-Agent Active Information Gathering in Discrete and Continuous-State Decentralized POMDPs by Policy Graph Improvement*, JAAMAS 34, 2020).
