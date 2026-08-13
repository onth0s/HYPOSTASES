# Wave 3 Front 06 — Communication as Bayesian Evidence: Paper Manifest

**Front**: Front 06 — Communication as Bayesian Evidence  
**Wave**: Wave 3 (Social Epistemology & Swarm Mechanics)  
**Target Substrate**: HYPOSTASES Engine ($\sigma = (c, w, g, \rho_{\text{ext}})$)  
**Rule 005 Invariant**: Pure Game-Theoretic & Probabilistic Rationality (Zero Artificial Human Cognitive Biases)  
**Ingestion Directory**: [`papers/`](../../WAVE_3_FRONT_06/papers)

All files validated for `%PDF` magic bytes and content (pypdf first-page text extraction). Files are tracked locally under `.gitignore` (`!docs/*/papers/*.pdf`).

---

## 1. Information Design & Bayesian Persuasion

### 1. Kamenica & Gentzkow (2011)
- **Title**: *Bayesian Persuasion*
- **Authors**: Emir Kamenica, Matthew Gentzkow
- **Journal/Venue**: *American Economic Review*, 101(6), 2590–2615 (2011)
- **Key Concepts**: Sender signal selection policy $\pi(m \mid \theta)$; concavification of receiver value function $V(\mu)$ over belief space $\Delta(\Theta)$; Bayes plausibility constraint on posterior distributions.
- **Engine Relevance**: Formalizes strategic sender-side signal design for the Strategic Likelihood Evaluator — how a peer agent $j$ chooses a signaling policy to optimize its expected utility under receiver Bayesian belief updates $P(\theta \mid m)$.
- **Source**: Author-hosted published version, `web.stanford.edu/~gentzkow/research/BayesianPersuasion.pdf`.

### 2. Crawford & Sobel (1982)
- **Title**: *Strategic Information Transmission*
- **Authors**: Vincent P. Crawford, Joel Sobel
- **Journal/Venue**: *Econometrica*, 50(6), 1431–1451 (1982)
- **Key Concepts**: Partition equilibria in cheap-talk signaling games; sender-receiver preference misalignment bias $b_{\text{bias}}$; coarse-graining of transmitted message vectors and equilibrium noise floors.
- **Engine Relevance**: Sets the expected message noise floor and coarse-graining regime for cheap-talk filtering under goal misalignment $\Delta u = \|g_j - g_i\|$.
- **Source**: Author-hosted scan, `econweb.ucsd.edu/~v2crawford/CrawfordSobel82EMT.pdf`.
- **Status**: Scanned document with no embedded text layer (validated via provenance + page count); OCR text extraction returns empty.

---

## 2. Rational Speech Act (RSA) & Pragmatic Probabilistic Inference

### 3. Frank & Goodman (2012)
- **Title**: *Predicting Pragmatic Reasoning in Language Games*
- **Authors**: Michael C. Frank, Noah D. Goodman
- **Journal/Venue**: *Science*, 336(6084), 998 (2012)
- **Key Concepts**: Speaker $S_1(m \mid \theta) \propto \exp(\alpha \cdot U(m; \theta))$; pragmatic listener $L_1(\theta \mid m) \propto P(\theta) S_1(m \mid \theta)$; parameter-free Bayesian referential communication.
- **Engine Relevance**: Baseline RSA formulation for message likelihood generation and pragmatic listener interpretation in the Strategic Likelihood Evaluator.
- **Source**: Stanford LangCog lab, `langcog.stanford.edu/papers/FG-science2012.pdf`.

### 4. Goodman & Frank (2016)
- **Title**: *Pragmatic Language Interpretation as Probabilistic Inference*
- **Authors**: Noah D. Goodman, Michael C. Frank
- **Journal/Venue**: *Trends in Cognitive Sciences*, 20(11), 818–829 (2016)
- **Key Concepts**: Recursive Theory of Mind speaker-listener inference; communicative intent as utility-theoretic cooperative principle; noisy-channel integration.
- **Engine Relevance**: Higher-order ToM state updates ($w.\text{peer\_beliefs}$) under communicative intent; formal grounding for pragmatic receiver likelihoods.
- **Source**: Stanford LangCog lab author manuscript (`goodman-2016-underrev.pdf`); content matches the published TiCS paper.

---

## 3. Subjective Logic, Epistemic Trust & Reputation Systems

### 5. Sabater & Sierra (2005)
- **Title**: *Review on Computational Trust and Reputation Models*
- **Authors**: Jordi Sabater, Carles Sierra
- **Journal/Venue**: *Artificial Intelligence Review*, 24(1), 33–60 (2005)
- **Key Concepts**: Multi-faceted trust decomposition (direct experience, witness information, sociological/structural role); trust model classification dimensions; game-theoretical trust as subjective probability.
- **Engine Relevance**: Blueprint for the dual-aspect trust profile $T_j = (T_{\text{honesty}}, T_{\text{competence}})$ and multi-agent network reputation routing.
- **Source**: Author-hosted, `www.iiia.csic.es/~jsabater/Publications/2005-AIR.pdf`.

---

## 4. Bayesian Learning in Social Networks & Echo-Chamber Filtering

### 6. Acemoglu, Dahleh, Lobel, & Ozdaglar (2011)
- **Title**: *Bayesian Learning in Social Networks*
- **Authors**: Daron Acemoglu, Munther A. Dahleh, Ilan Lobel, Asuman Ozdaglar
- **Journal/Venue**: *Review of Economic Studies*, 78(4), 1201–1236 (2011)
- **Key Concepts**: Perfect Bayesian equilibrium of sequential learning over general social network topologies; asymptotic learning conditions; bounded vs. unbounded private beliefs; misinformation resistance.
- **Engine Relevance**: Asymptotic learning conditions across swarm topologies and network structure influence on belief convergence; basis for the information deduplication / correlated noise filter.
- **Source**: Author-hosted published version, `pages.stern.nyu.edu/~ilobel/bayesian-learning-social-networks.pdf`.
- **Note**: Venue corrected from "Econometrica 79(6)" (erroneously cited in earlier drafts) to the verified *Review of Economic Studies* citation; confirmed by the file's own first page.

---

## Referenced but Not Ingested

The following bibliography entries in [`pertinent_literature.md`](../../WAVE_3_FRONT_06/pertinent_literature.md) have **no PDF in `papers/`** and are covered via the spec's summary mapping only:

- Bergemann & Morris (2019), *Information Design: A Unified Perspective*, JEL 57(1)
- Sobel (1985), *A Theory of Credibility*, ReStud 52(4)
- Farrell & Rabin (1996), *Cheap Talk*, JEP 10(3)
- Kao, Wu, Bergen, & Goodman (2014), *Nonliteral Language Understanding as Intentional Inference*, PNAS 111(35)
- Jøsang (2007), *Subjective Logic: A Formalism for Reasoning Under Uncertainty*, Springer (commercial monograph — not freely downloadable)
- Teacy, Patel, Jennings, & Luck (2006), *TRAVOS*, AAMAS 2006
- Bovens & Hartmann (2003), *Bayesian Epistemology*, Oxford University Press (commercial monograph — not freely downloadable)
- Liefgreen, Tešić, & Lagnado (2020), *Communicating Uncertainty and Unreliability in Bayesian Networks*, Cognitive Science 44(8)
- Golub & Jackson (2010), *Naïve Learning in Social Networks and the Wisdom of Crowds*, AEJ: Micro 2(1)
- Mossel & Tamuz (2017), *Opinion Exchange Dynamics: A Survey*, IEEE TNSE 4(3)
