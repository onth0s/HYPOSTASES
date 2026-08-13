# Ingested Literature & Reference Inventory — Wave 3 Front 06: Communication as Bayesian Evidence

**Location of Ingested PDFs**: [`docs/WAVE_3_FRONT_06/papers/`](../../WAVE_3_FRONT_06/papers)  
**Target Substrate**: HYPOSTASES Multi-Agent Engine ($\sigma = (c, w, g, \rho_{\text{ext}})$)  
**Rule 005 Invariant Compliance**: Pure Game-Theoretic & Probabilistic Rationality (Zero Artificial Human Cognitive Biases)

---

## Ingested Papers Summary & Technical Synthesis

### 1. Bayesian Persuasion & Information Design
- **Ingested PDF**: [`kamenica_gentzkow_2011_bayesian_persuasion.pdf`](../../WAVE_3_FRONT_06/papers/kamenica_gentzkow_2011_bayesian_persuasion.pdf)
- **Full Reference**: Kamenica, E., & Gentzkow, M. (2011). *Bayesian Persuasion*. **American Economic Review**, 101(6), 2590–2615.
- **Key Formulas & Mechanisms**:
  - Sender chooses signal distribution $\pi(s \mid \omega)$ over realization space $S$.
  - Receiver Bayes-plausible posterior belief:
    $$\mu_s(\omega) = \frac{\pi(s \mid \omega) \mu_0(\omega)}{\sum_{\omega'} \pi(s \mid \omega') \mu_0(\omega')}$$
  - Bayes plausibility constraint: $\sum_{s \in S} \mu_s \tau(\mu_s) = \mu_0$.
  - Concave closure: $V(\mu_0) \equiv \sup \{ z \mid (\mu_0, z) \in \text{co}(\hat{v}) \}$. Sender gains from persuasion iff $V(\mu_0) > \hat{v}(\mu_0)$.
  - Equivalence to straightforward recommendation signals $S \subseteq A$ where Receiver always follows recommendations.

### 2. Strategic Communication & Cheap Talk
- **Ingested PDF**: [`crawford_sobel_1982_strategic_information_transmission.pdf`](../../WAVE_3_FRONT_06/papers/crawford_sobel_1982_strategic_information_transmission.pdf)
- **Full Reference**: Crawford, V. P., & Sobel, J. (1982). *Strategic Information Transmission*. **Econometrica**, 50(6), 1431–1451.
- **Key Formulas & Mechanisms**:
  - Preference misalignment bias $b_{\text{bias}} = \|g_j - g_i\|$.
  - Sender utility $U^S(y, m, b) = -(y - (m + b))^2$, Receiver utility $U^R(y, m) = -(y - m)^2$.
  - Arbitrage condition over partition boundaries $a_i$:
    $$U^S(\bar{y}(a_i, a_{i+1}), a_i, b) - U^S(\bar{y}(a_{i-1}, a_i), a_i, b) = 0 \implies a_{i+1} = 2a_i - a_{i-1} + 4b$$
  - Max partition count: $N(b) = \left\langle -\frac{1}{2} + \frac{1}{2}\sqrt{1 + \frac{2}{b}} \right\rangle$.
  - Residual communication variance: $\sigma_m^2 = \frac{1}{12 N^2} + \frac{b^2(N^2 - 1)}{3}$.

### 3. Rational Speech Act (RSA) Informativeness
- **Ingested PDF**: [`frank_goodman_2012_predicting_pragmatic_reasoning.pdf`](../../WAVE_3_FRONT_06/papers/frank_goodman_2012_predicting_pragmatic_reasoning.pdf)
- **Full Reference**: Frank, M. C., & Goodman, N. D. (2012). *Predicting Pragmatic Reasoning in Language Games*. **Science**, 336(6084), 998.
- **Key Formulas & Mechanisms**:
  - Surprisal-based informativeness: $I(w; r_S, C) = -\log p(x) = -\log(|w|^{-1})$.
  - Speaker utility: $U(w; r_S, C) = I(w; r_S, C) - D(w)$.
  - Speaker Luce choice rule: $P(w \mid r_S, C) \propto \exp(\alpha \cdot U(w; r_S, C))$.
  - Pragmatic listener Bayesian update: $P(r_S \mid w, C) \propto P(w \mid r_S, C) P(r_S)$.

### 4. Recursive Probabilistic Pragmatics & Uncertain RSA (uRSA)
- **Ingested PDF**: [`goodman_frank_2016_pragmatic_language_interpretation.pdf`](../../WAVE_3_FRONT_06/papers/goodman_frank_2016_pragmatic_language_interpretation.pdf)
- **Full Reference**: Goodman, N. D., & Frank, M. C. (2016). *Pragmatic Language Interpretation as Probabilistic Inference*. **Trends in Cognitive Sciences**, 20(11), 818–829.
- **Key Formulas & Mechanisms**:
  - Literal Listener: $P_{\text{Lit}}(w \mid u) \propto \delta_{\llbracket u \rrbracket(w)} P(w)$.
  - Pragmatic Speaker: $P_S(u \mid w) \propto \exp(\alpha (\log P_{\text{Lit}}(w \mid u) - \text{cost}(u)))$.
  - Pragmatic Listener: $P_L(w \mid u) \propto P_S(u \mid w) P(w)$.
  - Uncertain RSA (uRSA): Joint inference over world state $w$ and speaker traits/topic $s$: $P_L(w, s \mid u) \propto P_S(u \mid w, s) P(s) P(w)$.

### 5. Multi-Faceted Computational Trust & Reputation (ReGreT)
- **Ingested PDF**: [`sabater_sierra_2005_computational_trust_reputation.pdf`](../../WAVE_3_FRONT_06/papers/sabater_sierra_2005_computational_trust_reputation.pdf)
- **Full Reference**: Sabater, J., & Sierra, C. (2005). *Review on Computational Trust and Reputation Models*. **Artificial Intelligence Review**, 24(1), 33–60.
- **Key Formulas & Mechanisms**:
  - Classification across Direct Interaction (DI), Direct Observation (DO), Witness Reputation ($R_W$), Neighborhood Reputation ($R_N$), System/Institutional Reputation ($R_I$).
  - Credibility module evaluating witness reliability and filtering dishonest/biased witness reports.
  - Subjective Logic Beta-binomial $(\alpha, \beta)$ updates and multi-context trust reliability metrics.

### 6. Bayesian Learning in Social Networks & Information Aggregation
- **Ingested PDF**: [`acemoglu_2011_bayesian_learning_social_networks.pdf`](../../WAVE_3_FRONT_06/papers/acemoglu_2011_bayesian_learning_social_networks.pdf)
- **Full Reference**: Acemoglu, D., Dahleh, M. A., Lobel, I., & Ozdaglar, A. (2011). *Bayesian Learning in Social Networks*. **Econometrica**, 79(6), 1795–1827.
- **Key Formulas & Mechanisms**:
  - Sequential decision over network topology $\{Q_n\}_{n \in \mathbb{N}}$.
  - Private belief $p_n = P(\theta = 1 \mid s_n) = \left( 1 + \frac{d\mathbb{F}_0}{d\mathbb{F}_1}(s_n) \right)^{-1}$.
  - Additive decision decomposition: $x_n = 1 \iff P(p_n \mid s_n) + P(\text{social belief}) > 1$.
  - Expanding observations condition: $\lim_{n \to \infty} \mathbb{Q}_n(\max_{b \in B(n)} b < K) = 0$.
  - Asymptotic learning under unbounded private beliefs (Theorem 2) and non-persuasive neighbourhoods (Theorem 4).
  - Provenance tree deduplication preventing echo-chamber overcounting in cyclic network topologies.
