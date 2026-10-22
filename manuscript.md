# Do Sentiment Models Generalize Across Product Categories? A Reproducible Cross-Domain Evaluation on Amazon Reviews

**Author**: Darash Mishra  
**Affiliation**: Independent Researcher, Navi Mumbai, India  
**Date**: October 2026  
**Artifact Repository**: `https://github.com/DM1ShRa/cross-domain-sentiment`

---

## 1. Abstract

Automated sentiment classification in electronic commerce is frequently undermined by cross-category domain divergence. We present a reproducible cross-domain evaluation on 27,000 balanced customer reviews from the Amazon Reviews 2023 benchmark across Books, Electronics, and All_Beauty. We evaluate three paradigms under a strict ternary formulation (Negative, Neutral, Positive): a rule-based lexicon (VADER), an in-domain and cross-domain supervised linear model (TF-IDF with Logistic Regression), and a zero-shot contextual transformer (RoBERTa-base). Our experimental results reveal an overall cross-domain generalization gap of $0.0940$ Macro-F1 for Logistic Regression, falling from an in-domain mean of $0.7460$ to a cross-domain mean of $0.6520$. We demonstrate pronounced transfer asymmetry: transferring between physical consumer goods (Electronics $\leftrightarrow$ All_Beauty) incurs minimal decay ($\Delta \approx 0.045$), whereas transferring between literary media (Books) and physical merchandise induces severe degradation ($\Delta = 0.1211$). Furthermore, VADER suffers a catastrophic neutral class collapse (neutral F1 $\le 0.1583$; 63.7%–72.3% of neutral reviews misclassified as positive). In throughput benchmarking, TF-IDF with Logistic Regression on a commodity CPU processes 6,417.38 reviews/sec—executing 154.67× faster than GPU RoBERTa (41.49 reviews/sec) while outperforming it by 11.71 Macro-F1 points in-domain. These findings provide actionable guidance for sustainable, production-scale e-commerce sentiment pipelines.

**Keywords**: Cross-Domain Sentiment Analysis, Domain Adaptation, Amazon Reviews 2023, Lexical Decay, Green AI, Multi-Class Sentiment.

---

## 2. Introduction

Automated sentiment analysis of customer reviews is essential for e-commerce catalog discovery, merchant feedback analysis, and recommendation ranking (Hou et al., 2024). While supervised text classifiers deliver high predictive accuracy when evaluated on the product categories on which they were trained, their accuracy drops sharply when deployed on unfamiliar merchandise genres—a phenomenon termed _domain divergence_ (Blitzer et al., 2007; Ramponi & Plank, 2020).

Customer reviews exhibit strong linguistic heterogeneity across product verticals. Books reviews discuss narrative flow and prose style; Electronics reviews evaluate hardware specifications, battery life, and compatibility; and All*Beauty reviews assess sensory texture, pigmentation, and skin reactions. When a supervised model learns weights over category-specific n-grams, out-of-domain transfer induces \_lexical decay* and _feature starvation_ (Chang et al., 2023), as target-domain vocabulary is omitted as out-of-vocabulary (OOV) tokens or misinterpreted due to semantic shifts.

Most prior cross-domain sentiment studies artificially binarize sentiment by discarding 3-star neutral reviews (Blitzer et al., 2007; Du et al., 2020; Wu & Shi, 2022), ignoring industrial settings where neutral and mixed reviews constitute a substantial share of volume. Neutral texts exhibit complex rhetorical properties—such as mixed pros and cons and faint praise—that challenge decision boundaries (Koppel & Schler, 2006; Ribeiro et al., 2016). Furthermore, following the "Green AI" paradigm (Schwartz et al., 2020), deployment requires balancing accuracy against inference throughput, latency, and hardware efficiency.

To address these empirical questions, we formulate four Research Questions (RQs) and four Hypotheses (Hs):

- **RQ1**: How does sentiment classification performance differ across Books, Electronics, and All_Beauty for rule-based, linear supervised, and pretrained transformer architectures?
- **RQ2**: How much does supervised Logistic Regression performance decrease when trained on one category and evaluated on another?
- **RQ3**: Which sentiment classes and linguistic characteristics cause the most cross-category classification errors?
- **RQ4**: What quantitative trade-offs govern predictive Macro-F1 performance versus computational inference throughput across CPU and GPU hardware?

Correspondingly, we evaluate four hypotheses:

- **H1**: Supervised Logistic Regression will achieve strong in-domain Macro-F1 performance but will lose Macro-F1 when transferred out-of-domain.
- **H2**: Pretrained RoBERTa evaluated zero-shot will exhibit greater cross-category metric stability across diverse product genres than unadapted linear transfer.
- **H3**: The rule-based lexicon (VADER) will deliver the highest computational throughput among evaluated architectures but will struggle with neutral and mixed-sentiment reviews.
- **H4**: Category-specific non-pivot vocabulary and domain-dependent semantic connotations will serve as primary drivers of cross-domain transfer errors.

This study delivers four contributions: (1) an independently reproducible benchmark on 27,000 balanced reviews from the Amazon Reviews 2023 dataset; (2) methodological disentanglement between true category transfer (a 3×3 supervised matrix) and category-independent comparison baselines (zero-shot VADER and RoBERTa); (3) full 3-class sentiment fidelity exposing a critical neutral class collapse; and (4) an end-to-end Green AI evaluation demonstrating the throughput and accuracy advantages of linear classifiers on commodity CPUs.

The paper is structured as follows: Section 3 synthesizes related work; Section 4 details data curation; Section 5 outlines methodology; Section 6 presents results; Section 7 provides error analysis; Section 8 discusses findings; Section 9 covers limitations; Section 10 concludes; and Section 11 lists references.

---

## 3. Related Work

### 3.1 Domain Discrepancy Theory and Adaptation Bounds

Cross-domain sentiment classification is formally grounded in statistical learning theory. Ben-David et al. (2010) formulated the canonical generalization bound on the expected target error $\epsilon_T(h)$ of hypothesis $h$ on target distribution $\mathcal{D}_T$ when trained on source distribution $\mathcal{D}_S$:

$$\epsilon_T(h) \le \epsilon_S(h) + \frac{1}{2} d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) + \lambda^*$$

where $\epsilon_S(h)$ is the source empirical error, $d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T)$ is the $\mathcal{H}\Delta\mathcal{H}$-divergence measuring feature distribution divergence, and $\lambda^*$ is the combined error of the ideal joint hypothesis $\lambda^* = \min_{h' \in \mathcal{H}} (\epsilon_S(h') + \epsilon_T(h'))$. In sentiment classification, non-zero $\lambda^*$ emerges when identical lexical tokens convey contradictory polarities in different domains (e.g., "unpredictable" signaling narrative intrigue in Books, but hardware malfunction in Electronics) (Blitzer et al., 2007; Chang et al., 2023). The empirical Generalization Gap tracked in our study ($\Delta_{\text{gen}} = \text{Macro-F1}_{\text{in-domain}} - \text{Macro-F1}_{\text{cross-domain}}$) operationalizes this theoretical penalty.

### 3.2 Pivot Features and Neural Domain Adaptation

Blitzer, Dredze, and Pereira (2007) introduced Structural Correspondence Learning (SCL) on legacy Amazon reviews, distinguishing between domain-invariant _pivot features_ ("excellent", "defective") and category-specific _non-pivot features_ ("unputdownable" in Books; "rechargeable" in Electronics; "pigmented" in Beauty). They proved that unadapted linear classifiers suffer severe transfer decay across dissimilar categories, whereas categories sharing physical utility (Electronics $\leftrightarrow$ Kitchen) transfer with much lower error than media-to-hardware pairings (Books $\leftrightarrow$ Electronics).

While deep transformers possess rich representations, Du et al. (2020) demonstrated that contextual language models suffer from a "Domain-Awareness Gap," proposing adversarial training (BERT-DAAT) to align category representations. Similarly, Wu and Shi (2022) developed AdSPT (Adversarial Soft Prompt Tuning) to address category verbalizer drift. These findings illuminate our evaluation of pretrained RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`; Barbieri et al., 2020; Loureiro et al., 2022): zero-shot RoBERTa avoids out-of-vocabulary loss, yielding stable cross-domain F1, but lacks domain-specific calibration without explicit adaptation.

### 3.3 Lexical Decay and Dataset Drift in E-Commerce

The **Amazon Reviews 2023** benchmark (Hou et al., 2024) provides an updated foundation for cross-domain NLP across 571 million reviews. Hou et al. demonstrated extreme linguistic divergence between cultural media and physical goods. Chang et al. (2023) categorized linguistic dataset drift into _vocabulary drift_, _structural drift_, and _semantic drift_, identifying vocabulary drift as the primary driver of performance degradation. In TF-IDF models, vocabulary drift causes feature starvation as target non-pivots receive zero weights ($w_t = 0$). While neural subword tokenizers avoid discrete OOV omissions (Ramponi & Plank, 2020), contextual encoders remain susceptible to semantic drift when category connotations diverge from pretraining distributions.

### 3.4 Neutral Sentiment Challenges and Green AI

Koppel and Schler (2006) proved that neutral examples exhibit distinct rhetorical properties and cannot be treated merely as intermediate thresholds. Ribeiro et al. (2016) demonstrated across 24 sentiment methods on 18 datasets that transitioning from binary to 3-class evaluation causes accuracy and Macro-F1 to plummet by 20–35 percentage points, with rule-based lexicons (VADER; Hutto & Gilbert, 2014) collapsing acutely on neutral text.

Enterprise systems must balance accuracy against computational costs. Schwartz et al. (2020) formalized "Green AI," making inference throughput, latency, and energy efficiency primary evaluation metrics alongside leaderboard accuracy. Our study operationalizes Green AI by benchmarking throughput across CPU and GPU hardware.

### 3.5 Positioning and Literature Gaps

This work bridges four persistent gaps in the literature:

1. **Modern Corpus Benchmark**: We evaluate the contemporary Amazon Reviews 2023 dataset (Hou et al., 2024) rather than legacy corpora (Blitzer et al., 2007).
2. **Realistic 3-Class Evaluation**: Unlike binary studies discarding 3-star reviews (Du et al., 2020; Wu & Shi, 2022), we evaluate ternary sentiment (Koppel & Schler, 2006; Ribeiro et al., 2016).
3. **Methodological Disentanglement**: We separate true category transfer (3×3 supervised matrix) from category-independent baselines (VADER and RoBERTa).
4. **Holistic Multi-Metric Benchmarking**: We evaluate predictive accuracy alongside throughput under Green AI principles (Schwartz et al., 2020).

---

## 4. Dataset and Preprocessing

### 4.1 Data Source and Category Selection

Data is obtained from the **Amazon Reviews 2023** repository (`McAuley-Lab/Amazon-Reviews-2023`). We select three diverse top-level categories:

- **Books**: Cultural media characterized by extended prose, narrative critique, and literary vocabulary.
- **Electronics**: Hardware consumer goods characterized by technical specifications, compatibility, and durability.
- **All_Beauty**: Personal care merchandise characterized by sensory descriptions, cosmetic outcomes, and tactile application feedback.

### 4.2 Sentiment Class Mapping and Balanced Sampling

Reviews are mapped from 1–5 star ratings into three sentiment classes:

- **Negative ($y = 0$)**: 1-star and 2-star reviews.
- **Neutral ($y = 1$)**: 3-star reviews.
- **Positive ($y = 2$)**: 4-star and 5-star reviews.

To isolate domain divergence from class imbalance skew, we perform balanced stratified sampling of exactly 3,000 reviews per sentiment class per category (9,000 reviews per category; 27,000 total).

### 4.3 Data Cleaning, Hygiene, and Leakage Prevention

1. **Title and Body Concatenation**: Review title and text are concatenated: `review_text = title + " " + text`.
2. **Empty Review Filtering**: Null and whitespace-only entries are removed.
3. **Deduplication**: Exact duplicate review texts are purged across all categories.
4. **Stratified Split**: Each category corpus is split 80/20 into 7,200 training reviews (2,400 per class) and 1,800 test reviews (600 per class) using a fixed deterministic seed (`seed = 42`).
5. **Leakage Prevention**: TF-IDF vectorizers are fitted strictly on training sets and transformed on test sets. All models are evaluated on identical, immutable test sets.

---

## 5. Experimental Methodology

### 5.1 Model Architectures

- **Rule-Based Lexicon (VADER)**: Unsupervised lexicon engine (Hutto & Gilbert, 2014) combining 7,500 valence tokens with grammatical heuristics. Compound score $s \in [-1, +1]$ maps to Positive ($s \ge +0.05$), Neutral ($-0.05 < s < +0.05$), and Negative ($s \le -0.05$).
- **Supervised Linear Model (TF-IDF + Logistic Regression)**: Sublinear TF-IDF ($n \in \{1, 2\}$, vocab 20,000) with $\ell_2$-regularized Logistic Regression ($C = 1.0$, L-BFGS).
- **Pretrained Contextual Transformer (RoBERTa-base)**: 124M-parameter `cardiffnlp/twitter-roberta-base-sentiment-latest` (Barbieri et al., 2020; Loureiro et al., 2022), evaluated zero-shot without fine-tuning.

### 5.2 Mandatory Methodological Distinction: Category Transfer vs. Category-Independent Comparison

We enforce a strict distinction between model evaluation roles:

- **Category-Transfer Evaluation (TF-IDF + Logistic Regression)**: Trained on a single category (7,200 reviews) and evaluated across all three test sets (1,800 reviews each), generating a full **3×3 cross-category transfer matrix** (9 cells). Diagonal cells measure in-domain performance; off-diagonal cells measure true cross-domain transfer.
- **Category-Independent Evaluation (VADER and RoBERTa)**: Neither model receives category training on Amazon data. Evaluated directly on the identical test sets, they provide category-independent reference baselines benchmarking general lexical and contextual capacity.

### 5.3 Evaluation Metrics and Hardware Benchmarking

The primary metric is **Macro-averaged F1** ($ ext{Macro-F1} = \frac{1}{3} \sum*c F*{1, c}$). Secondary metrics include Accuracy, per-class Precision/Recall/F1, 3×3 confusion matrices, and throughput (reviews/sec). The Generalization Gap is defined as $\Delta_{\text{gen}}(\mathcal{S}) = \text{Macro-F1}(\mathcal{S} \rightarrow \mathcal{S}) - \text{mean}_{\mathcal{T} \ne \mathcal{S}}(\text{Macro-F1}(\mathcal{S} \rightarrow \mathcal{T}))$. Throughput is benchmarked over 5,400 test reviews on commodity CPU (VADER, LogReg) and GPU (RoBERTa, CUDA, batch 32).

---

## 6. Results

### 6.1 In-Domain and Model Comparison Performance

Table 1 reports classification performance across models and categories on identical 1,800-review test sets.

#### Table 1: Model Comparison across Product Categories

_Macro-F1, Accuracy, and per-class Precision, Recall, and F1 across Books, Electronics, and All_Beauty test sets (1,800 reviews per category, 600 per class)._

| Model Architecture              | Test Category | Macro-F1   | Accuracy   | Neg Prec | Neg Rec | Neg F1 | Neu Prec | Neu Rec | Neu F1 | Pos Prec | Pos Rec | Pos F1 |
| ------------------------------- | ------------- | ---------- | ---------- | -------- | ------- | ------ | -------- | ------- | ------ | -------- | ------- | ------ |
| **VADER**                       | Books         | 0.3834     | 0.4633     | 0.5903   | 0.4683  | 0.5223 | 0.2576   | 0.0283  | 0.0511 | 0.4261   | 0.8933  | 0.5770 |
| **VADER**                       | Electronics   | 0.4545     | 0.5244     | 0.6309   | 0.5583  | 0.5924 | 0.4312   | 0.0783  | 0.1326 | 0.4845   | 0.9367  | 0.6386 |
| **VADER**                       | All_Beauty    | 0.4518     | 0.5156     | 0.6698   | 0.4733  | 0.5547 | 0.3797   | 0.1000  | 0.1583 | 0.4795   | 0.9733  | 0.6425 |
| **TF-IDF + LogReg (In-Domain)** | Books         | **0.7578** | **0.7583** | 0.7406   | 0.7517  | 0.7461 | 0.6878   | 0.6683  | 0.6779 | 0.8438   | 0.8550  | 0.8493 |
| **TF-IDF + LogReg (In-Domain)** | Electronics   | **0.7390** | **0.7378** | 0.7394   | 0.7233  | 0.7313 | 0.6377   | 0.6717  | 0.6542 | 0.8451   | 0.8183  | 0.8315 |
| **TF-IDF + LogReg (In-Domain)** | All_Beauty    | **0.7412** | **0.7422** | 0.7803   | 0.7400  | 0.7596 | 0.6548   | 0.6417  | 0.6481 | 0.7885   | 0.8450  | 0.8158 |
| **Pretrained RoBERTa**          | Books         | 0.6381     | 0.6583     | 0.6816   | 0.7850  | 0.7297 | 0.5726   | 0.3483  | 0.4332 | 0.6788   | 0.8417  | 0.7515 |
| **Pretrained RoBERTa**          | Electronics   | 0.6429     | 0.6806     | 0.6334   | 0.8667  | 0.7319 | 0.6722   | 0.2700  | 0.3853 | 0.7358   | 0.9050  | 0.8117 |
| **Pretrained RoBERTa**          | All_Beauty    | 0.6058     | 0.6533     | 0.6182   | 0.8367  | 0.7110 | 0.5814   | 0.2083  | 0.3067 | 0.7102   | 0.9150  | 0.7997 |

In-domain supervised Logistic Regression achieves the highest performance across all domains (Macro-F1 $0.7578$ Books, $0.7390$ Electronics, $0.7412$ All_Beauty; mean $0.7460$). Pretrained RoBERTa delivers intermediate, category-stable performance ($0.6058$–$0.6429$). VADER fails substantially ($0.3834$–$0.4545$) due to neutral class collapse (neutral F1 $\le 0.1583$; recall $2.83\%$–$10.00\%$). RoBERTa also struggles on neutral reviews (recall $20.83\%$–$34.83\%$), whereas in-domain Logistic Regression maintains robust neutral classification (F1 $> 0.648$, recall $> 64\%$).

### 6.2 3×3 Logistic Regression Transfer Matrix

Table 2 presents the 3×3 transfer matrix for TF-IDF with Logistic Regression.

#### Table 2: 3×3 Logistic Regression Cross-Category Transfer Matrix (Macro-F1)

_Rows represent training category (7,200 reviews); columns represent test category (1,800 reviews). In-domain baselines in bold._

| Training Source $\downarrow$ / Evaluation Target $\rightarrow$ | Books      | Electronics | All_Beauty |
| -------------------------------------------------------------- | ---------- | ----------- | ---------- |
| **Books**                                                      | **0.7578** | 0.6415      | 0.6319     |
| **Electronics**                                                | 0.6364     | **0.7390**  | 0.6926     |
| **All_Beauty**                                                 | 0.6111     | 0.6986      | **0.7412** |

Transfer between physical goods is highly resilient: Electronics $\rightarrow$ All_Beauty achieves $0.6926$ Macro-F1 (decay of $-0.0464$), and All_Beauty $\rightarrow$ Electronics achieves $0.6986$ (decay of $-0.0426$). In contrast, transferring from Books to physical categories causes severe drops: $-0.1163$ on Electronics ($0.6415$) and $-0.1259$ on All_Beauty ($0.6319$).

![heatmap](figures\transfer_heatmap_page-0001.jpg)

### 6.3 Generalization Gap Analysis

Table 3 details the empirical generalization gap for each category.

#### Table 3: Cross-Category Generalization Gap (Macro-F1)

_In-domain Macro-F1, mean cross-domain Macro-F1, and resulting generalization gap ($\Delta_{\text{gen}}$)._

| Training Category | In-Domain Macro-F1 | Mean Cross-Domain Macro-F1 | Generalization Gap ($\Delta_{\text{gen}}$) |
| ----------------- | ------------------ | -------------------------- | ------------------------------------------ |
| **Books**         | 0.7578             | 0.6367                     | 0.1211                                     |
| **Electronics**   | 0.7390             | 0.6645                     | 0.0745                                     |
| **All_Beauty**    | 0.7412             | 0.6548                     | 0.0864                                     |
| **Overall Mean**  | **0.7460**         | **0.6520**                 | **0.0940**                                 |

The overall mean generalization gap across all domains is **$0.0940$**. Books exhibits the largest gap ($0.1211$), nearly double that of Electronics ($0.0745$).

### 6.4 Computational Throughput Benchmarks

Table 4 reports throughput and latency benchmarks over 5,400 test reviews.

#### Table 4: Computational Throughput and Hardware Benchmarking

_Throughput and elapsed time on 5,400 test reviews. Speedups normalized to GPU RoBERTa._

| Model Architecture               | Hardware Device    | Elapsed Time (s) | Throughput (reviews/sec) | Speedup vs. RoBERTa  |
| -------------------------------- | ------------------ | ---------------- | ------------------------ | -------------------- |
| **TF-IDF + Logistic Regression** | Commodity CPU      | **0.8415**       | **6,417.38**             | **154.67×**          |
| **Rule-Based Lexicon (VADER)**   | Commodity CPU      | **5.1044**       | **1,057.92**             | **25.50×**           |
| **Pretrained RoBERTa-base**      | Dedicated CUDA GPU | **130.1570**     | **41.49**                | **1.00× (Baseline)** |

TF-IDF with Logistic Regression on a commodity CPU processes **6,417.38 reviews/sec**, running **154.67× faster** than GPU RoBERTa ($41.49$ reviews/sec) and **6.07× faster** than CPU VADER ($1,057.92$ reviews/sec).

![paretoCurve](figures\pareto_curve_page-0001.jpg)

### 6.5 Confusion Matrix Error Distributions

Table 5 summarizes confusion matrix distributions across test sets.

#### Table 5: Confusion Matrix Error Flow and Neutral Class Dynamics

_True label versus predicted label distribution (count and within-class percentage) across test sets (600 reviews per class)._

| Model Architecture     | Category    | True Class     | Predicted as Negative | Predicted as Neutral | Predicted as Positive |
| ---------------------- | ----------- | -------------- | --------------------- | -------------------- | --------------------- |
| **VADER**              | Books       | Negative (600) | 281 (46.83%)          | 31 (5.17%)           | 288 (48.00%)          |
| **VADER**              | Books       | Neutral (600)  | 149 (24.83%)          | 17 (2.83%)           | **434 (72.33%)**      |
| **VADER**              | Books       | Positive (600) | 46 (7.67%)            | 18 (3.00%)           | 536 (89.33%)          |
| **VADER**              | Electronics | Neutral (600)  | 171 (28.50%)          | 47 (7.83%)           | **382 (63.67%)**      |
| **VADER**              | All_Beauty  | Neutral (600)  | 132 (22.00%)          | 60 (10.00%)          | **408 (68.00%)**      |
| **LogReg (In-Domain)** | Books       | Negative (600) | 451 (75.17%)          | 123 (20.50%)         | 26 (4.33%)            |
| **LogReg (In-Domain)** | Books       | Neutral (600)  | 130 (21.67%)          | 401 (66.83%)         | 69 (11.50%)           |
| **LogReg (In-Domain)** | Books       | Positive (600) | 28 (4.67%)            | 59 (9.83%)           | 513 (85.50%)          |
| **LogReg (In-Domain)** | Electronics | Neutral (600)  | 133 (22.17%)          | 403 (67.17%)         | 64 (10.67%)           |
| **LogReg (In-Domain)** | All_Beauty  | Neutral (600)  | 111 (18.50%)          | 385 (64.17%)         | 104 (17.33%)          |
| **RoBERTa**            | Books       | Neutral (600)  | 203 (33.83%)          | 209 (34.83%)         | 188 (31.33%)          |
| **RoBERTa**            | Electronics | Negative (600) | 520 (86.67%)          | 44 (7.33%)           | 36 (6.00%)            |
| **RoBERTa**            | Electronics | Neutral (600)  | **279 (46.50%)**      | 162 (27.00%)         | 159 (26.50%)          |
| **RoBERTa**            | Electronics | Positive (600) | 22 (3.67%)            | 35 (5.83%)           | 543 (90.50%)          |
| **RoBERTa**            | All_Beauty  | Neutral (600)  | **292 (48.67%)**      | 125 (20.83%)         | 183 (30.50%)          |

The confusion matrices reveal an inverse error bias: VADER exhibits a strong **Positive bias** on neutral text ($63.7\%–72.3\%$ predicted Positive), whereas RoBERTa exhibits a **Negative bias** ($46.5\%–48.7\%$ predicted Negative in Electronics and Beauty). In-domain Logistic Regression remains well-calibrated ($64.2\%–67.2\%$ correct neutral identification).

---

## 7. Error Analysis

We examine misclassified reviews from `misclassified_sample.csv` across seven structural error categories:

### 7.1 Category 1: Mixed Positive and Negative Opinions

Reviews frequently combine praise with criticism. Transferred models lacking domain-specific vocabulary latch onto whichever clause matches available weights.

- _Sample Review 1_: `"Meh. I do like the length my eyelashes can get with this product. I dont like the clumpy,spidery leg look I get when I apply it"` (All*Beauty; True: Neutral; Predicted: Negative by LogReg Books $\rightarrow$ All_Beauty). \_Mechanism*: The review balances praise ("like the length") and criticism ("clumpy"). Lacking cosmetic feature weights, the Books-trained model defaulted to generic negative tokens.
- _Sample Review 2_: `"It's a READER, not really a tablet... As a reader, it's exceptional... Be prepared... glitches... on the 'up side'? The video was flawless... the Fire rocks."` (Electronics; True: Positive [4 stars]; Predicted: Negative by LogReg Books $\rightarrow$ Electronics). _Mechanism_: Hardware defect terms ("glitches", "failures") overwhelmed the transferred model despite the 4-star summary.

### 7.2 Category 2: Negation Scope

Negation interacting with domain utility constraints defeats rule heuristics.

- _Sample Review_: `"Three Stars. not super sticky but holds the cord"` (Electronics; True: Neutral; Predicted: Negative by VADER). _Mechanism_: VADER penalizes "not super sticky", ignoring that moderate adhesion is functional in cord clips and "holds the cord" denotes task completion.

### 7.3 Category 3: Implicit Sentiment and Faint Praise

Implicit critique conveys dissatisfaction without explicit negative adjectives.

- _Sample Review 1_: `"For beginners..."` (All*Beauty; True: Negative [1 star]; Predicted: Neutral by LogReg Electronics $\rightarrow$ All_Beauty). \_Mechanism*: In cosmetics, "for beginners" signals inferior formulation. Transferred from Electronics—where it denotes usability—the model assigned neutral valence.
- _Sample Review 2_: `"2nd one cracked with 3 hours of it being installed... Works with my case, so gets two stars... Down to 1 star."` (All*Beauty; True: Negative [1 star]; Predicted: Positive by VADER). \_Mechanism*: Polite hedging ("Works with my case") triggered VADER's positive additive dictionary, overriding structural product failure.

### 7.4 Category 4: Category-Specific Terminology and Non-Pivot Drift

Domain-specific terms carry evaluative gravity not recognized across categories.

- _Sample Review 1_: `"Ok if you like it oily..."` (All*Beauty; True: Negative [1 star]; Predicted: Neutral by RoBERTa). \_Mechanism*: In skincare, "oily" indicates product defect; in Twitter pretraining, it is neutral physical description, leading RoBERTa to misclassify it as Neutral.
- _Sample Review 2_: `"USB Cables. Not the best cables, I bought these a long time ago, I am glad they have better cables now."` (Electronics; True: Negative [2 stars]; Predicted: Positive by LogReg All*Beauty $\rightarrow$ Electronics). \_Mechanism*: The All_Beauty model lacks electronic degradation weights; encountering "glad" and "better", it predicted Positive.

### 7.5 Category 5: Very Short Reviews

Terse reviews with informal noise deprive models of syntactic context.

- _Sample Review_: `"Cords. Very chip made but does the job"` (Electronics; True: Positive [5 stars]; Predicted: Neutral by RoBERTa). _Mechanism_: A phonetic typo ("chip made" for _cheaply made_) in an ultra-short 7-token review deprives RoBERTa of syntactic structure, causing it to default to Neutral.

### 7.6 Category 6: Star-Rating/Text Disagreement

Differences in category rating norms induce label-text misalignment.

- _Sample Review_: `"Chaucer - great introductory book with fantastic illustrations!... perfect for the entire family. Beautiful illustrations... Delighted to have it!"` (Books; True: Neutral [3 stars]; Predicted: Positive by VADER and LogReg). _Mechanism_: Literary reviewers frequently assign 3 stars as an objective appraisal while deploying effusive praise ("great", "fantastic", "delighted"). Models predictably predict Positive based on surface valence.

### 7.7 Category 7: Neutral or Ambiguous Language

Subtle scalar balance or faint praise creates predictive uncertainty.

- _Sample Review 1_: `"Hardly ANY product. It works but you get hardly any product! I'm so disappointed..."` (All*Beauty; True: Neutral [3 stars]; Predicted: Negative by VADER). \_Mechanism*: Volume dissatisfaction ("hardly ANY product", "disappointed") caused VADER to classify it Negative, overriding the 3-star rating based on formula efficacy.
- _Sample Review 2_: `"WORKS GREAT! very time involved replacment!"` (Electronics; True: Positive [5 stars]; Predicted: Neutral by LogReg Electronics $\rightarrow$ Electronics). _Mechanism_: Pairing superlative praise with an operational caveat creates probability dilution into Neutral.

---

## 8. Discussion

### 8.1 Research Question Answers

- **RQ1 (Cross-Category Differences)**: In-domain supervised LogReg achieves the highest Macro-F1 across all domains ($0.7578$ Books, $0.7390$ Electronics, $0.7412$ All_Beauty; mean $0.7460$). Pretrained RoBERTa delivers intermediate, category-stable performance ($0.6058$–$0.6429$; mean $0.6289$). VADER fails across all categories ($0.3834$ Books, $0.4545$ Electronics, $0.4518$ All_Beauty) due to neutral class collapse.
- **RQ2 (Magnitude of Transfer Degradation)**: Logistic Regression incurs an overall mean generalization gap of **$0.0940$** ($0.7460$ in-domain vs. $0.6520$ cross-domain). Transfer is strongly asymmetric: Books suffers steep decay ($\Delta = 0.1211$; dropping to $0.6415$ on Electronics and $0.6319$ on Beauty), whereas physical goods transfer is resilient (Electronics $\leftrightarrow$ Beauty $\Delta \approx 0.045$).
- **RQ3 (Sentiment Classes and Linguistic Drivers)**: The Neutral class is the primary locus of failure across all models. VADER suffers neutral collapse (neutral F1 $\le 0.1583$; recall $2.83\%$–$10.00\%$; $>63\%$ predicted Positive). RoBERTa exhibits an inverse negative bias ($46.5\%$–$48.7\%$ of neutral reviews predicted Negative). Linguistic drivers include non-pivot vocabulary drift, negation in functional contexts, and rating-text disagreement.
- **RQ4 (Predictive vs. Computational Efficiency Trade-Offs)**: On CPU hardware, TF-IDF + Logistic Regression processes **6,417.38 reviews/sec**—running **154.67× faster** than GPU RoBERTa ($41.49$ reviews/sec) and **6.07× faster** than CPU VADER ($1,057.92$ reviews/sec). Supervised LogReg dominates RoBERTa in both throughput and Macro-F1 ($0.7460$ vs. $0.6289$ in-domain; $0.6520$ vs. $0.6289$ cross-domain), validating Green AI principles (Schwartz et al., 2020).

### 8.2 Hypothesis Verdicts

- **H1 (LogReg Transfer Degradation)**: **ACCEPTED**. In-domain LogReg achieves $0.7460$ mean Macro-F1 ($0.7578$ Books, $0.7390$ Electronics, $0.7412$ All_Beauty). In all six transfer evaluations, Macro-F1 drops (mean $0.6520$), yielding a $0.0940$ overall generalization gap.
- **H2 (RoBERTa Metric Stability)**: **ACCEPTED (with qualification on absolute performance)**. RoBERTa's zero-shot Macro-F1 varies by only $0.0371$ across domains (sample standard deviation $s = 0.0202$), versus a $0.1467$ spread for LogReg across its transfer matrix. However, LogReg achieves superior absolute Macro-F1 over RoBERTa's mean baseline ($0.6289$) in 8 of 9 matrix cells, and surpasses target-domain zero-shot RoBERTa in 6 of 9 cells.
- **H3 (VADER Speed and Neutral Struggle)**: **PARTIALLY REJECTED / PARTIALLY ACCEPTED**. The neutral struggle clause is **ACCEPTED**: VADER collapsed on neutral reviews (neutral recall $2.83\%$–$10.00\%$; $>63\%$ predicted Positive). The speed clause is **REJECTED**: VADER was not fastest; CPU LogReg ran 6.07× faster ($6,417.38$ vs. $1,057.92$ rev/s).
- **H4 (Category Vocabulary Contribution to Errors)**: **ACCEPTED**. Non-pivot terms (cosmetic "oily", "clumpy" vs. electronic "cords", "adapter") absent in source vocabularies receive zero weights when transferred, inducing feature starvation and driving the $0.0940$ generalization gap.

### 8.3 Methodological Synthesis

Our findings emphasize the distinction between category transfer and category-independent baselines. Zero-shot transformers establish an _invariance floor_: leveraging subword embeddings, RoBERTa avoids OOV loss, maintaining stable ~0.63 Macro-F1. However, lacking category tuning, it fails to capture domain sentiment semantics. Supervised linear models capture category semantics effectively in-domain, but require structural adaptation (e.g., pivot alignment) when bridging distant genres.

### 8.4 Practical Recommendations for Enterprise Deployment

1. **Retire Rule-Based Lexicons for Multi-Class Ingestion**: VADER's severe neutral collapse ($< 10\%$ recall) renders it unsuitable for production where neutral feedback carries high operational value.
2. **Deploy Supervised Linear Models for Production Workloads**: For catalog domains with training data, TF-IDF with Logistic Regression delivers superior accuracy ($0.7460$ F1) and extreme CPU throughput ($> 6,400$ reviews/sec).
3. **Cluster Merged Domains to Mitigate Transfer Decay**: Organizations can exploit transfer affinities: an Electronics model transfers robustly to Beauty ($\Delta < 0.05$), enabling categories to be clustered into broader domains.
4. **Reserve Transformers for Nuanced Offline Tasks**: Deep transformers should be reserved for offline, aspect-level analysis rather than synchronous high-volume ingestion.

---

## 9. Limitations

1. **Category Scope**: We evaluated three representative categories (Books, Electronics, All_Beauty). Evaluating additional categories (e.g., Clothing, Grocery, Automotive) is necessary to map domain divergence across the full e-commerce taxonomy.
2. **Ternary Rating Discretization**: Discretizing 5-star ratings into 3 classes treats 1-star and 2-star reviews as equivalent and conflates true neutrality with mixed sentiment in 3-star reviews.
3. **Zero-Shot Transformer Evaluation**: Pretrained RoBERTa was evaluated zero-shot to benchmark out-of-the-box transfer. Domain-adaptive fine-tuning would likely improve F1, though at significant computational expense.
4. **Unadapted Linear Transfer**: Logistic Regression transfer was evaluated unadapted to establish an unadulterated baseline for the generalization gap. Unsupervised domain adaptation techniques (e.g., SCL) represent a natural next step.

---

## 10. Conclusion

This study presented a reproducible, multi-metric cross-domain sentiment evaluation on 27,000 balanced reviews from the Amazon Reviews 2023 dataset across Books, Electronics, and All_Beauty. By enforcing a realistic 3-class formulation and distinguishing category transfer from category-independent baselines, we quantified the operational realities of e-commerce domain divergence.

Supervised Logistic Regression exhibits an overall generalization gap of **$0.0940$** Macro-F1, marked by sharp transfer asymmetry: physical merchandise transfers robustly ($\Delta \approx 0.045$), while narrative media transferred to physical goods incurs severe decay ($\Delta = 0.1211$). We identified a critical neutral class collapse in rule-based lexicons (VADER neutral F1 $\le 0.1583$) and an inverse error skew between rule-based and transformer models. In hardware benchmarking, lightweight linear models on commodity CPUs establish an exceptional Green AI Pareto frontier, processing 6,417 reviews/sec (154× faster than GPU RoBERTa) while surpassing zero-shot transformers in Macro-F1. Future work includes evaluating parameter-efficient adaptation (LoRA) and discourse parsing for mixed-sentiment feedback.

---

## 11. References

Barbieri, F., Camacho-Collados, J., Espinosa-Anke, L., & Neves, L. (2020). TweetEval: Unified benchmark and comparative evaluation for tweet classification. In _Findings of the Association for Computational Linguistics: EMNLP 2020_ (pp. 1644–1650). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.findings-emnlp.148

Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan, J. W. (2010). A theory of learning from different domains. _Machine Learning_, 79(1–2), 151–175. https://doi.org/10.1007/s10994-009-5152-4

Blitzer, J., Dredze, M., & Pereira, F. (2007). Biographies, Bollywood, boom-boxes and blenders: Domain adaptation for sentiment classification. In _Proceedings of the 45th Annual Meeting of the Association for Computational Linguistics (ACL 2007)_ (pp. 440–447). Association for Computational Linguistics. https://aclanthology.org/P07-1056

Chang, T. A., Halder, K., John, N. A., Vyas, Y., Benajiba, Y., Ballesteros, M., & Roth, D. (2023). Characterizing and measuring linguistic dataset drift. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ (pp. 8953–8967). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.acl-long.498

Du, C., Sun, H., Wang, J., Qi, Q., & Liao, J. (2020). Adversarial and domain-aware BERT for cross-domain sentiment analysis. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)_ (pp. 4019–4028). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.acl-main.370

Hou, Y., Li, J., Fu, X., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). _Bridging language and items for retrieval and recommendation: Benchmarking LLMs as semantic encoders_. arXiv preprint arXiv:2403.03952. https://arxiv.org/abs/2403.03952

Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. In _Proceedings of the Eighth International AAAI Conference on Weblogs and Social Media (ICWSM-14)_ (pp. 216–225). AAAI Press. https://ojs.aaai.org/index.php/ICWSM/article/view/14550

Koppel, M., & Schler, J. (2006). The importance of neutral examples for learning sentiment. _Computational Intelligence_, 22(2), 100–109. https://doi.org/10.1111/j.1467-8640.2006.00276.x

Loureiro, D., Barbieri, F., Neves, L., Espinosa-Anke, L., & Camacho-Collados, J. (2022). TimeLMs: Diachronic language models from Twitter. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: System Demonstrations (ACL 2022)_ (pp. 251–260). Association for Computational Linguistics. https://doi.org/10.18653/v1/2022.acl-demo.25

Ramponi, A., & Plank, B. (2020). Neural unsupervised domain adaptation in NLP—A survey. In _Proceedings of the 28th International Conference on Computational Linguistics (COLING 2020)_ (pp. 6838–6855). International Committee on Computational Linguistics. https://doi.org/10.18653/v1/2020.coling-main.603

Ribeiro, F. N., Araújo, M., Gonçalves, P., André Gonçalves, M., & Benevenuto, F. (2016). SentiBench — a benchmark comparison of state-of-the-practice sentiment analysis methods. _EPJ Data Science_, 5(1), Article 23. https://doi.org/10.1140/epjds/s13688-016-0085-1

Schwartz, R., Dodge, J., Smith, N. A., & Etzioni, O. (2020). Green AI. _Communications of the ACM_, 63(12), 54–63. https://doi.org/10.1145/3381831

Wu, H., & Shi, X. (2022). Adversarial soft prompt tuning for cross-domain sentiment analysis. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ (pp. 2438–2447). Association for Computational Linguistics. https://doi.org/10.18653/v1/2022.acl-long.174
