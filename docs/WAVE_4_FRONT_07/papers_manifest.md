# Wave 4 Front 07 — Meta-Learning Layer: Paper Manifest

**Front**: Front 07 — Meta-Learning  
**Wave**: Wave 4 (Meta-Learning & Architectural Evolution)  
**Target Substrate**: HYPOSTASES Engine ($\sigma = (c, w, g, \rho_{\text{ext}})$)  
**Rule 005 Invariant**: Pure Game-Theoretic & Probabilistic Rationality (Zero Artificial Human Cognitive Biases)  
**Ingestion Directory**: [`papers/`](file:///c:/Users/Leonardo/001/00__DEV/HYPOSTASES/docs/WAVE_4_FRONT_07/papers/)

All 8 papers in `papers/` have been fully ingested, validated, and mathematically analyzed. In compliance with Rule 010, PDF assets are untracked in Git (`.gitignore`).

---

## 1. Primary Ingested Literature

### 1. Finn, Abbeel, & Levine (2017)
- **Title**: *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks* (MAML)
- **Authors**: Chelsea Finn, Pieter Abbeel, Sergey Levine
- **Venue/Year**: *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70, 2017
- **Key Theoretical Contribution**: Establishes the foundational bi-level gradient-based meta-learning paradigm. Trains model initializations $\theta$ across a distribution of tasks $p(\mathcal{T})$ such that a small number of fast-adaptation gradient steps on a novel task yields optimal task-specific performance without introducing model architecture constraints or auxiliary parameters.
- **Exact Mathematical Mechanisms**:
  - Task distribution sampling: $\mathcal{T}_i \sim p(\mathcal{T})$ with support set $\mathcal{D}_i$ and query set $\mathcal{D}_i'$.
  - Inner-loop fast adaptation:
    $$\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{\mathcal{D}_i}(f_\theta)$$
  - Outer-loop meta-objective minimization:
    $$\min_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}^{\mathcal{D}_i'}(f_{\theta'_i}) = \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}^{\mathcal{D}_i'}\left(f_{\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{\mathcal{D}_i}(f_\theta)}\right)$$
  - Outer-loop meta-parameter update rule:
    $$\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}^{\mathcal{D}_i'}(f_{\theta'_i})$$
  - First-Order MAML (FOMAML) approximation: omits second-order Hessian terms $\nabla_\theta^2 \mathcal{L}$ for computational speedup ($~33\%$ speedup).
- **Engine Relevance**: Provides the baseline bi-level optimization structure for fast adaptation of agent policy weights and internal models in HYPOSTASES across scenario distributions $p(\mathcal{S})$.

---

### 2. Grant, Finn, Levine, Darrell, & Griffiths (2018)
- **Title**: *Recasting Gradient-Based Meta-Learning as Hierarchical Bayes*
- **Authors**: Erin Grant, Chelsea Finn, Sergey Levine, Trevor Darrell, Thomas Griffiths
- **Venue/Year**: *International Conference on Learning Representations (ICLR)* / arXiv:1801.08930, 2018
- **Key Theoretical Contribution**: Proves that gradient-based early stopping in MAML fast adaptation is mathematically equivalent to Maximum A Posteriori (MAP) inference under an implicit Gaussian prior $p(\phi_j \mid \theta) = \mathcal{N}(\phi_j; \theta, \mathbf{Q})$. Introduces the Lightweight Laplace Approximation for Meta-Adaptation (LLAMA) using Kronecker-factored Approximate Curvature (K-FAC) to capture parameter posterior uncertainty.
- **Exact Mathematical Mechanisms**:
  - Hierarchical Bayesian marginal likelihood:
    $$p(\mathbf{X} \mid \theta) = \prod_j \int p(\mathbf{X}_j \mid \phi_j) p(\phi_j \mid \theta) \, d\phi_j$$
  - Early stopping equivalence to regularized quadratic optimization:
    $$\min_\phi \left( \|\mathbf{y} - \mathbf{X}\phi\|_2^2 + \|\theta - \phi\|_{\mathbf{Q}}^2 \right) \iff p(\phi \mid \mathbf{X}, \mathbf{y}, \theta) \propto \mathcal{N}(\mathbf{y}; \mathbf{X}\phi, \mathbf{I}) \mathcal{N}(\phi; \theta, \mathbf{Q})$$
    where $\mathbf{Q} = \mathbf{O}\mathbf{\Lambda}^{-1}\left((\mathbf{I} - \mathbf{B}\mathbf{\Lambda})^{-k} - \mathbf{I}\right)\mathbf{O}^T$.
  - LLAMA Laplace marginal log-likelihood objective:
    $$-\log p(\mathbf{X} \mid \theta) \approx \sum_j \left[ -\log p(\mathbf{X}_j \mid \hat{\phi}_j) - \log p(\hat{\phi}_j \mid \theta) + \frac{1}{2} \log \det(\mathbf{H}_j) \right]$$
  - K-FAC curvature approximation of posterior Hessian $\mathbf{H}_j$:
    $$\mathbf{H}_j = \nabla_{\phi_j}^2 [-\log p(\mathbf{X}_j \mid \phi_j)] + \nabla_{\phi_j}^2 [-\log p(\phi_j \mid \theta)] \approx \hat{\mathbf{H}}$$
- **Engine Relevance**: Formalizes the probabilistic interpretation of meta-priors in HYPOSTASES ($c.m_{\text{procedural}}$), enabling variance-aware parameter tracking and uncertainty quantification during scenario shifts.

---

### 3. Baik, Choi, Choi, Kim, & Lee (2020)
- **Title**: *Meta-Learning with Adaptive Hyperparameters* (ALFA)
- **Authors**: Sungyong Baik, Myungsub Choi, Janghoon Choi, Heewon Kim, Kyoung Mu Lee
- **Venue/Year**: *Advances in Neural Information Processing Systems (NeurIPS 33)*, 2020
- **Key Theoretical Contribution**: Replaces fixed inner-loop hyperparameters with a light meta-network $g_\phi$ that dynamically generates per-step, per-layer learning rates $\alpha_{i,j}$ and weight decay/regularization coefficients $\beta_{i,j}$ conditioned on the base learner's current learning state $\tau_{i,j} = [\nabla_\theta \mathcal{L}, \theta]$. Enables rapid fast adaptation even from random initializations.
- **Exact Mathematical Mechanisms**:
  - Inner-loop adaptive update rule with dynamic Hadamard scaling:
    $$\theta_{i,j+1} = \beta_{i,j} \odot \theta_{i,j} - \alpha_{i,j} \odot \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{\mathcal{D}_i}(f_{\theta_{i,j}})$$
  - Hyperparameter generator network $g_\phi$:
    $$(\alpha_{i,j}^1, \beta_{i,j}^1) = g_\phi(\bar{\tau}_{i,j})$$
    where $\bar{\tau}_{i,j}$ represents layer-wise means of base learner gradients and weight vectors ($2N$-dimensional input for an $N$-layer network).
  - Layer-wise and step-wise post-multiplier composition:
    $$\alpha_{i,j} = \alpha_{i,j}^0 \odot \alpha_{i,j}^1(\bar{\tau}_{i,j}), \quad \beta_{i,j} = \beta_{i,j}^0 \odot \beta_{i,j}^1(\bar{\tau}_{i,j})$$
    where $\alpha_{i,j}^0, \beta_{i,j}^0$ are step-wise layer-wise meta-learnable post-multipliers.
  - Outer-loop update of meta-network parameters $\phi$:
    $$\phi \leftarrow \phi - \eta \nabla_\phi \sum_{\mathcal{T}_i} \mathcal{L}_{\mathcal{T}_i}^{\mathcal{D}_i'}(f_{\theta'_i})$$
- **Engine Relevance**: Directly guides dynamic online calibration of learning rates and decay terms (such as `MOOD_DECAY_RATE` = 0.1) across inner-loop state update routines in HYPOSTASES.

---

### 4. Champion, Bowman, Marković, & Grześ (2024)
- **Title**: *Reframing the Expected Free Energy: Four Formulations and a Unification*
- **Authors**: Théophile Champion, Howard Bowman, Dimitrije Marković, Marek Grześ
- **Venue/Year**: *arXiv preprint*, arXiv:2402.14460v1 [cs.AI], Feb 2024
- **Key Theoretical Contribution**: Formalizes the Expected Free Energy (EFE) unification problem in active inference. Establishes the rigorous distinction between forecast distribution $F(\bar{o}, \bar{s} \mid \bar{a})$ and target distribution $T(\bar{o}, \bar{s} \mid \bar{a})$, proves mathematical relationships/bounds among all four EFE decompositions ($\mathcal{C}_{\text{ROA}}$, $\mathcal{C}_{\text{RSA}}$, $\mathcal{C}_{\text{IGPV}}$, $\mathcal{C}_{3\text{E}}$), and derives the linear preference compatibility constraint $C_o = A C_s$.
- **Exact Mathematical Mechanisms**:
  - Forecast distribution factorization over POMDP:
    $$F(\bar{o}, \bar{s} \mid \bar{a}) = F(s_{t+1} \mid a_t) \prod_{\tau=t+1}^h F(o_\tau \mid s_\tau) \prod_{\tau=t+2}^h F(s_\tau \mid s_{\tau-1}, a_{\tau-1})$$
  - Target distribution formulation:
    $$T(\bar{o}, \bar{s} \mid \bar{a}) = \prod_{\tau=t+1}^h T(o_\tau \mid s_\tau) T(s_\tau \mid \bar{a}), \quad T(s_\tau \mid \bar{a}) = \text{Cat}(s_\tau; C_s)$$
  - Root EFE definition ($\mathcal{C}_{\text{ROA}}$ Risk over Observations + Ambiguity):
    $$\mathcal{G}_{rt}(\bar{a}) \triangleq D_{KL}[F(\bar{o} \mid \bar{a}) \parallel T(\bar{o} \mid \bar{a})] + \mathbb{E}_{F(\bar{s} \mid \bar{a})}[H[F(\bar{o} \mid \bar{s})]] = \mathcal{C}_{\text{ROA}}(\bar{a})$$
  - Information Gain + Pragmatic Value equivalence ($\mathcal{C}_{\text{IGPV}}$):
    $$\mathcal{C}_{\text{IGPV}}(\bar{a}) \triangleq -\mathbb{E}_{F(\bar{o} \mid \bar{a})}[D_{KL}[F(\bar{s} \mid \bar{o}, \bar{a}) \parallel F(\bar{s} \mid \bar{a})]] - \mathbb{E}_{F(\bar{o} \mid \bar{a})}[\ln T(\bar{o} \mid \bar{a})] = \mathcal{G}_{rt}(\bar{a})$$
  - Upper Bound relationship ($\mathcal{C}_{\text{RSA}}$ Risk over States + Ambiguity):
    $$\mathcal{G}_{rt}(\bar{a}) \le D_{KL}[F(\bar{s} \mid \bar{a}) \parallel T(\bar{s} \mid \bar{a})] + \mathbb{E}_{F(\bar{s} \mid \bar{a})}[H[F(\bar{o} \mid \bar{s})]] = \mathcal{C}_{\text{RSA}}(\bar{a}) \quad \text{under } T(\bar{o} \mid \bar{s}) = F(\bar{o} \mid \bar{s})$$
  - Entropy + Expected Energy equivalence ($\mathcal{C}_{3\text{E}}$):
    $$\mathcal{C}_{3\text{E}}(\bar{a}) \triangleq -H[F(\bar{s} \mid \bar{a})] - \mathbb{E}_{F(\bar{o}, \bar{s} \mid \bar{a})}[\ln T(\bar{o}, \bar{s} \mid \bar{a})] = \mathcal{C}_{\text{RSA}}(\bar{a})$$
  - Preference Simplex Linear Compatibility:
    $$C_o = A C_s$$
    Demonstrates that valid observation preferences $C_o$ must lie within the 1-dimensional simplex transformed by likelihood matrix $A$; arbitrary $C_o$ choice may yield invalid non-categorical $C_s = A^{-1} C_o$.
- **Engine Relevance**: Ensures mathematical rigor for active sensing policy selection in $g.u$ under Rule 009 (`efe_mode: true`), verifying game-theoretic preference compatibility.

---

### 5. Friston et al. (2024)
- **Title**: *From pixels to planning: scale-free active inference* (Renormalising Generative Models)
- **Authors**: Karl Friston, Conor Heins, Tim Verbelen, Lancelot Da Costa, Tommaso Salvatori, Dimitrije Marković, Alexander Tschantz, Magnus Koudahl, Christopher Buckley, Thomas Parr
- **Venue/Year**: *Technical Note / arXiv preprint*, 2024
- **Key Theoretical Contribution**: Introduces Renormalising Generative Models (RGMs) extending POMDPs to include paths ($u$) as latent variables and multi-scale spatiotemporal hierarchies via the renormalisation group (RG operator). Formalizes active learning of Dirichlet count matrices $\mathbf{a}$ and active model structure selection via Expected Free Energy $G(\mathbf{a})$.
- **Exact Mathematical Mechanisms**:
  - Generative model tensor specification: $\mathbf{A}$ (likelihood), $\mathbf{B}$ (transitions given paths $u$), $\mathbf{C}$ (prior preferences), $\mathbf{D}$ (initial states), $\mathbf{E}$ (path priors).
  - Variational Free Energy (VFE) for discrete state-space:
    $$F = D_{KL}[Q(s_\tau, u_\tau, a) \parallel P(s_\tau, u_\tau, a)] - \mathbb{E}_Q[\ln P(o_\tau \mid s_\tau, u_\tau, a)] = \text{complexity} - \text{accuracy}$$
  - Expected Free Energy over Dirichlet parameter tensors $\mathbf{a}$:
    $$G(\mathbf{a}) = -\mathbb{E}_{Q_a}[D_{KL}[P(s, o \mid \mathbf{a}) \parallel P(o \mid \mathbf{a}) P(s \mid \mathbf{a})]] - \mathbb{E}_{Q_a}[\ln P(o \mid c)] = \text{mutual information} + \text{expected cost}$$
  - Active learning parameter update (Bayesian Model Averaging / Selection):
    $$P(u) = \sigma(-\alpha G(\mathbf{a} \mid u))$$
    $$\mathbf{a}_{\tau+1}^g = \mathbf{a}_\tau^g + P(u_1) \Delta \mathbf{a}_\tau^g$$
    where Dirichlet parameter count tensor updates $\Delta \mathbf{a}_\tau^g = \mathbf{o}_\tau \otimes_{i \in \text{pa}} \mathbf{s}_\tau^i$.
  - Structure learning log Bayes factor (Bayesian Model Reduction / Expansion):
    $$\Delta F = \ln \mathcal{B}(\mathbf{a}) + \ln \mathcal{B}(a') - \ln \mathcal{B}(a) - \ln \mathcal{B}(\mathbf{a} + a' - a)$$
  - Coarse-graining RG operator flow over space (spin-block SVD) and time (path tiling).
- **Engine Relevance**: Supplies the multi-scale spatiotemporal coarse-graining operator and Dirichlet parameter count accumulator for long-horizon active inference over $\sigma = (c, w, g, \rho_{\text{ext}})$.

---

### 6. Behrouz, Razaviyayn, Zhong, & Mirrokni (2025)
- **Title**: *Nested Learning: The Illusion of Deep Learning Architectures*
- **Authors**: Ali Behrouz, Meisam Razaviyayn, Peiling Zhong, Vahab Mirrokni
- **Venue/Year**: *39th Conference on Neural Information Processing Systems (NeurIPS 2025)* / arXiv:2512.24695
- **Key Theoretical Contribution**: Proposes the Nested Learning (NL) paradigm. Demonstrates that deep architectures, optimizers (SGD+Momentum, Adam, Muon), attention mechanisms, and memory systems are all nested associative memory modules optimizing their own "context flows" at distinct update frequencies $f_A$. Introduces Deep Optimizers, Self-Modifying Titans, and Continuum Memory Systems (CMS).
- **Exact Mathematical Mechanisms**:
  - Associative Memory Operator Definition:
    $$\mathcal{M}^* = \arg\min_{\mathcal{M}} \tilde{\mathcal{L}}(\mathcal{M}(\mathcal{K}); \mathcal{V})$$
  - Update frequency partial ordering: $A \succ B \iff f_A > f_B$ or ($f_A = f_B$ with structural dependency).
  - Momentum as an Associative Memory:
    $$m_{t+1} = \arg\min_m \left( -\langle m, \nabla_{W_t} \mathcal{L}(W_t; x_{t+1}) \rangle + \eta_{t+1} \|m - m_t\|_2^2 \right)$$
  - Linear Attention as Associative Memory:
    $$\mathcal{M}_{t+1} = \arg\min_{\mathcal{M}} \left( \langle \mathcal{M} k_{t+1}, v_{t+1} \rangle + \|\mathcal{M} - \mathcal{M}_t\|_F^2 \right) \implies \mathcal{M}_{t+1} = \mathcal{M}_t + v_{t+1} k_{t+1}^T$$
  - Continuum Memory System (CMS) scheduled parameter update:
    $$\theta_{i+1}^{(f_\ell)} = \theta_i^{(f_\ell)} - \sum_{t=i-C^{(\ell)}}^i \eta_t^{(\ell)} f(\theta_t^{(f_\ell)}; x_t) \quad \text{if } i \equiv 0 \pmod{C^{(\ell)}}$$
  - Self-Modifying Titans & HOPE recurrent memory update:
    $$\mathcal{M}_{\Box, t} = \mathcal{M}_{\Box, t-1} (\alpha_t \mathbf{I} - \eta_t k_t k_t^T) - \eta_t \nabla_{\mathcal{M}_{\Box, t-1}} \mathcal{L}_{\mathcal{M}_{\Box, t-1}}\left(\mathcal{M}_{\Box, t-1}; k_t, \hat{v}_{\Box, t}\right)$$
    for $\Box \in \{k, v, q, \eta, \alpha, \text{memory}\}$.
- **Engine Relevance**: Defines the multi-frequency nested memory architecture for $c.m_{\text{episodic}}$, $c.m_{\text{semantic}}$, and $c.m_{\text{procedural}}$, unifying inner-loop fast updates (Tier-1 ticks) with outer-loop meta-parameter updates (Tier-3 ticks).

---

### 7. Tian, Liu, & Sun (2025)
- **Title**: *Meta-Learning Hyperparameters for Parameter Efficient Fine-Tuning* (MetaPEFT)
- **Authors**: Zichen Tian, Yaoyao Liu, Qianru Sun
- **Venue/Year**: *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 23037–23047, 2025
- **Key Theoretical Contribution**: Converts the Mixed Integer Non-Linear Programming (MINLP) problem of PEFT hyperparameter selection (layer positions, block depths $d$, scaling factors $\alpha$) into a continuous, differentiable optimization problem via a unified scalar modulator $\gamma \in \mathbb{R}$ with Softplus activation. Solves this via a bi-level meta-learning framework.
- **Exact Mathematical Mechanisms**:
  - Additive PEFT baseline with discrete positional indicator $\mathbf{1}_p \in \{0,1\}$:
    $$y = f(x; \theta) + \mathbf{1}_p (\alpha \cdot \Delta(x; \phi))$$
  - Continuous Differentiable Unified Modulator formulation:
    $$y = f(x; \theta) + \text{Softplus}(\gamma) \cdot \Delta(x; \phi)$$
    where $\gamma \in \mathbb{R}$ controls both activation ($\gamma \to -\infty \implies \text{Softplus}(\gamma) \to 0$) and adaptation strength.
  - MINLP continuous relaxation via Bi-Level Meta-Learning:
    $$\min_{\gamma \in \mathbb{R}^+} \mathcal{L}_{\text{val}}(\phi_\gamma^*; \mathcal{D}_{\text{val}}), \quad \text{s.t. } \phi_\gamma^* = \arg\min_\phi \mathcal{L}_{\text{train}}(\phi, \gamma; \mathcal{D}_{\text{train}})$$
  - Inner-loop optimization of adapter weights $\phi$:
    $$\phi_{t+1} = \phi_t - \eta_\phi \nabla_\phi \mathcal{L}_{\text{train}}(\phi_t, \gamma_t; \mathcal{D}_{\text{train}})$$
  - Outer-loop optimization of continuous modulators $\gamma$:
    $$\gamma_{t+1} = \gamma_t - \eta_\gamma \nabla_\gamma \mathcal{L}_{\text{val}}(\phi_{t+1}; \mathcal{D}_{\text{val}})$$
    with dynamic stratified random sampling of validation splits $\mathcal{D}_{\text{val}}$.
- **Engine Relevance**: Provides differentiable selection and gating mechanisms for modular agent components (active sensing heuristics, memory modules, planning primitives) in continuous parameter space.

---

### 8. Xia, Chen, Yang, Tu, Liu, Xiong, Han, Qiu, Ji, Zhou, Zheng, Xie, & Yao (2026)
- **Title**: *MetaClaw: Just Talk – An Agent That Meta-Learns and Evolves in the Wild*
- **Authors**: Peng Xia, Jianwen Chen, Xinyu Yang, Haoqin Tu, Jiaqi Liu, Kaiwen Xiong, Siwei Han, Shi Qiu, Haonian Ji, Yuyin Zhou, Zeyu Zheng, Cihang Xie, Huaxiu Yao
- **Venue/Year**: *arXiv preprint*, arXiv:2603.17187v1 [cs.LG], Mar 2026
- **Key Theoretical Contribution**: Introduces MetaClaw, a dual-timescale continual meta-learning framework for deployed autonomous agents $\mathcal{M} = (\theta, \mathcal{S})$. Unifies (1) fast, zero-downtime, gradient-free skill evolution in natural language space $\mathcal{S}$ from failure trajectories, with (2) slow, opportunistic gradient-based RL weight updates on policy $\theta$ during user-inactive windows (OMLS). Enforces strict Support-Query data separation via skill generation versioning.
- **Exact Mathematical Mechanisms**:
  - Meta-model representation:
    $$\mathcal{M} = (\theta, \mathcal{S}), \quad a \sim \pi_\theta(\cdot \mid \tau, \text{Retrieve}(\mathcal{S}, \tau))$$
  - Skill-Driven Fast Adaptation (Gradient-Free Experience Distillation):
    $$\mathcal{S}_{g+1} = \mathcal{S}_g \cup \mathcal{E}(\mathcal{S}_g, \mathcal{D}_g^{\text{sup}})$$
    where $\mathcal{E}$ is an LLM skill evolver, $g$ is the skill generation index, and $\mathcal{D}_g^{\text{sup}}$ contains failure trajectories.
  - Opportunistic Policy Optimization (Gradient-Based RL Fine-Tuning via GRPO):
    $$\theta_{t+1} = \theta_t + \alpha \nabla_\theta \mathbb{E}_{(\tau, \xi, g') \sim \mathcal{B}} [ R(\pi_\theta(\cdot \mid \tau, \mathcal{S}_{g'})) ]$$
    where $R$ is a Process Reward Model (PRM) score, $g' \le g^*$, and $\mathcal{B}$ is the query replay buffer.
  - Support-Query Data Versioning Protocol:
    $$\text{If } \mathcal{S}_g \to \mathcal{S}_{g+1}, \quad \text{Flush } \{ \xi \in \mathcal{B} \mid \text{version}(\xi) \le g \}$$
    Prevents stale reward contamination by ensuring policy updates only train on query data $\mathcal{D}_{g+1}^{\text{qry}}$ collected under the post-adaptation skill library.
  - Opportunistic Meta-Learning Scheduler (OMLS) idle trigger:
    $$\text{Trigger} = \text{IsSleepWindow}() \lor \text{IsSystemInactive}(\delta) \lor \text{IsCalendarOccupied}()$$
- **Engine Relevance**: Guides the agentic dual-timescale meta-evolution loop in HYPOSTASES, coupling fast declarative skill injection in $c.m_{\text{procedural}}$ with slow opportunistic parameter fine-tuning during idle phases under strict version control.
