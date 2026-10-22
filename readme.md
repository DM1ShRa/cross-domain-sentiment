# Do Sentiment Models Generalize Across Product Categories?

### A Reproducible Cross-Domain Evaluation on Amazon Reviews

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified-brightgreen.svg)]()

<!-- DOI Badge will be added here upon Zenodo release -->

This repository contains the complete experimental artifacts, dataset curation pipelines, and evaluation code for the research paper:  
**"Do Sentiment Models Generalize Across Product Categories? A Reproducible Cross-Domain Evaluation on Amazon Reviews"**.

The study investigates domain divergence, neutral-class dynamics, and Green AI computational trade-offs across 27,000 balanced customer reviews from the **Amazon Reviews 2023** benchmark spanning three distinct verticals: **Books**, **Electronics**, and **All Beauty**[cite: 3, 6].

---

## 📄 Key Research Deliverables

- **Full Manuscript**: [`manuscript.md`](manuscript.md) (Compiled publication-ready manuscript)
- **Raw LaTeX Source**: [`manuscript.md`](manuscript.md), [`references.bib`](references.bib)
- **Preprocessed Dataset**: [`data/amazon_reviews_full_3000.csv`](data\amazon_reviews_demo_200.csv) (27,000 stratified, balanced reviews)[cite: 3, 6]
- **End-to-End Pipeline**: [`pipeline.py`](pipeline.py)
- **Automated Unit & Regression Tests**: [`test_pipeline.py`](test_pipeline.py)

---

## 🔬 Core Empirical Findings

1. **Pronounced Transfer Asymmetry ($\Delta_{\text{gen}} = 0.0940$ Macro-F1)**[cite: 3]:
   - Transfer between physical utility items (**Electronics $\leftrightarrow$ All Beauty**) is resilient, with minimal decay ($\Delta \approx 0.045$)[cite: 3].
   - Transfer between literary prose (**Books**) and physical goods incurs catastrophic degradation ($\Delta = 0.1211$) due to severe feature starvation and out-of-vocabulary non-pivot drift[cite: 3, 5].

2. **The "Neutral Collapse" of Lexicon Classifiers**:
   - **VADER** collapses when handling real-world 3-class sentiment[cite: 3, 5], suffering an extreme positive bias where **63.7%–72.3% of neutral reviews are misclassified as positive** ($F_1 \le 0.1583$)[cite: 3, 7].
   - Zero-shot **RoBERTa-base** demonstrates an inverse bias, incorrectly assigning negative polarity to $46.5\%–48.7\%$ of neutral reviews.

3. **Green AI Efficiency Frontier**:
   - Supervised **TF-IDF + Logistic Regression on a commodity CPU** processes **6,417.38 reviews/sec**—running **154.67× faster** than GPU RoBERTa ($41.49$ rev/s) and **6.07× faster** than CPU VADER ($1,057.92$ rev/s)[cite: 3, 6].
   - CPU-based linear models dominate zero-shot transformers in both throughput and Macro-F1 ($0.7460$ vs. $0.6289$ in-domain)[cite: 3, 7].

---

## 📊 Visualizations

### 1. Cross-Domain Transfer Matrix (Macro-F1)

The $3\times3$ transfer matrix illustrates the in-domain baseline (diagonal) versus cross-category degradation (off-diagonal).

![Cross-Domain Transfer Heatmap](figures\transfer_heatmap_page-0001.jpg)

### 2. Green AI Pareto Frontier

Throughput (reviews processed per second, log scale) versus predictive Macro-F1 performance across hardware architectures[cite: 3, 7].

![Green AI Pareto Frontier](figures\pareto_curve_page-0001.jpg)

---

## 🗂 Repository Structure

```text
├── data/
│   └── amazon_reviews_full_3000.csv  # 27,000 balanced reviews across 3 categories
├── results/
│   ├── model_metrics.csv              # Full precision, recall, and Macro-F1 scores
│   ├── logreg_transfer_matrix.csv     # 3x3 category transfer matrix data
│   ├── generalization_gap.csv         # Computed empirical transfer penalties
│   ├── throughput.csv                 # Latency and throughput benchmarks
│   ├── confusion_matrices.csv         # Disaggregated class-level error flow
│   └── misclassified_sample.csv       # Qualitative error analysis cases
├── figures/
│   ├── transfer_heatmap_page-0001.jpg # Transfer matrix visualization
│   └── pareto_curve_page-0001.jpg     # Green AI efficiency frontier
├── manuscript.md                     # Formatted research paper
├── manuscript.md                      # Raw Markdown source
├── references.bib                     # Academic bibliography
├── pipeline.py                        # Execution script for full benchmark
└── test_pipeline.py                   # Automated validation suite
```

### 1. Environment Setup

Clone this repository and set up a virtual environment:

```bash
git clone https://github.com/DM1ShRa/cross-domain-sentiment.git

cd cross-domain-sentiment

python -m venv venv
# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate
```

Install the dependencies:

```bash
pip install pandas numpy scikit-learn nltk torch transformers matplotlib seaborn
```

Download required NLTK resources:

```python
import nltk
nltk.download('vader_lexicon')
```

### 2. Run the Benchmark Pipeline

To execute the data preprocessing, model training, cross-domain matrix evaluation, and hardware benchmarking:

```bash
python pipeline.py
```

The script will reproduce all metrics and regenerate the output CSV files in the `results/` directory.

### 3. Run Pipeline Tests

Verify data integrity, stratified splits, and model convergence:

```bash
python -m unittest test_pipeline.py
```

---

## 📈 Benchmark Summary Tables

### In-Domain vs. Zero-Shot Baseline Performance

| Model Architecture              | Category    |  Macro-F1  |  Accuracy  |   Neu F1   |
| :------------------------------ | :---------- | :--------: | :--------: | :--------: |
| **TF-IDF + LogReg (In-Domain)** | Books       | **0.7578** | **0.7583** | **0.6779** |
| **TF-IDF + LogReg (In-Domain)** | Electronics | **0.7390** | **0.7378** | **0.6542** |
| **TF-IDF + LogReg (In-Domain)** | All Beauty  | **0.7412** | **0.7422** | **0.6481** |
| Pretrained RoBERTa              | Books       |   0.6381   |   0.6583   |   0.4332   |
| Pretrained RoBERTa              | Electronics |   0.6429   |   0.6806   |   0.3853   |
| Pretrained RoBERTa              | All Beauty  |   0.6058   |   0.6533   |   0.3067   |
| VADER                           | Books       |   0.3834   |   0.4633   |   0.0511   |
| VADER                           | Electronics |   0.4545   |   0.5244   |   0.1326   |
| VADER                           | All Beauty  |   0.4518   |   0.5156   |   0.1583   |

### Logistic Regression $3\times3$ Transfer Matrix (Macro-F1)

| Training Source $\downarrow$ \ Evaluation Target $\rightarrow$ |   Books    | Electronics | All Beauty |
| :------------------------------------------------------------- | :--------: | :---------: | :--------: |
| **Books**                                                      | **0.7578** |   0.6415    |   0.6319   |
| **Electronics**                                                |   0.6364   | **0.7390**  |   0.6926   |
| **All Beauty**                                                 |   0.6111   |   0.6986    | **0.7412** |

---

## 📜 Citation

If you use this benchmark, code, or findings in your research, please cite:

```bibtex
@article{cross_domain_sentiment_2026,
  title={Do Sentiment Models Generalize Across Product Categories? A Reproducible Cross-Domain Evaluation on Amazon Reviews},
  author={Mishra, Darash},
  year={2026},
  url={[https://github.com/DM1ShRa/cross-domain-sentiment](https://github.com/DM1ShRa/cross-domain-sentiment)}
}
```
