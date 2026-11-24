# Global Wellbeing & Socio-Economic Archetypes: Advanced Clustering Analysis
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9%2B-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end unsupervised machine learning and data analytics project that segments 156 nations from the **World Happiness Report** based on fundamental socio-economic, public health, civil liberty, and institutional integrity indicators.

---

## 📌 Executive Summary

While traditional analyses evaluate national happiness through rank tables, this project applies **unsupervised clustering algorithms** to discover natural global archetypes purely from structural drivers, strictly avoiding target leakage. By evaluating multiple clustering paradigms (**K-Means** and **Hierarchical Agglomerative Clustering**) alongside **Principal Component Analysis (PCA)** and rigorous multi-metric validation, the analysis reveals a distinct 3-tier global taxonomy:

1. **Tier 1 — High Wellbeing & Institutional Trust:** Advanced economies distinguished by robust social safety nets and low corruption perception (high institutional trust).
2. **Tier 2 — Transitional & Middle-Income Economies:** Developing nations with solid economic productivity and community cohesion, but hindered by pervasive institutional distrust.
3. **Tier 3 — Vulnerable & Resource-Constrained Nations:** Economies experiencing severe structural challenges across income, healthcare, and civil freedoms, yet demonstrating resilient community generosity.

---

## 📊 Dataset & Feature Dictionary

The dataset is sourced from the **World Happiness Report** (`happiness_report.csv`), encompassing **156 nations** evaluated across 9 attributes:

| Column Name | Type | Role | Description |
| :--- | :---: | :---: | :--- |
| `Overall rank` | Integer | Metadata | Global happiness ranking (1 to 156). |
| `Country or region` | String | Identifier | Country or territory name. |
| `Score` | Float | External Benchmark | Cantril Ladder survey score (range: 2.85 – 7.77), withheld during clustering. |
| `GDP per capita` | Float | Feature | Purchasing Power Parity (PPP) economic production per capita. |
| `Social support` | Float | Feature | National average response to having family/friends to count on in times of need. |
| `Healthy life expectancy` | Float | Feature | Physical health and life expectancy at birth based on WHO data. |
| `Freedom to make life choices` | Float | Feature | Freedom of personal autonomy and life decisions. |
| `Generosity` | Float | Feature | Benchmark of charitable donations and prosocial behavior. |
| `Perceptions of corruption` | Float | Feature | Measure of institutional transparency and absence of public/corporate corruption. |

> **Note on Data Leakage Prevention:** `Score` and `Overall rank` are deliberately excluded from feature matrices during scaling, PCA, and model training. They are only reintroduced during post-clustering validation to independently verify the quality of unsupervised cluster separation.

---

## 🔬 Methodology & Analytical Pipeline

```
┌─────────────────────────────────┐
│     happiness_report.csv        │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Data Integrity & Diagnostics    │ ──> Missing values check (0 nulls), skewness & kurtosis
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  StandardScaler Normalization   │ ──> Standardized 6 structural drivers (μ = 0, σ = 1)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Dimensionality Reduction      │ ──> PCA: PC1 (49.8%) + PC2 (23.8%) = 73.6% variance
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Hyperparameter Optimization    │ ──> Evaluated k ∈ [2, 8] across 4 quantitative metrics
└────────────────┬────────────────┘
                 │
                 ├─────────────────────────────────────────┐
                 ▼                                         ▼
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│        K-Means (k = 3)          │       │    Agglomerative Clustering     │
│   Optimal partition benchmark   │       │   Ward linkage dendrogram split │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │                                         │
                 └───────────────────┬─────────────────────┘
                                     │
                                     ▼
                       ┌───────────────────────────┐
                       │ Cross-Model Consensus     │ ──> Adjusted Rand Index (ARI) = 0.573
                       └─────────────┬─────────────┘
                                     │
                                     ▼
                       ┌───────────────────────────┐
                       │ Archetype Profiling &     │ ──> Z-score centroids, PCA 2D biplot,
                       │ External Validation       │     and Score separation distributions
                       └───────────────────────────┘
```

---

## 📈 Hyperparameter Optimization & Model Selection

Selecting $k=3$ is quantitatively justified across four complementary clustering validation criteria:

| Number of Clusters ($k$) | Silhouette Score (Higher is Better) | Davies-Bouldin Index (Lower is Better) | Calinski-Harabasz Index (Higher is Better) | WCSS / Inertia |
| :---: | :---: | :---: | :---: | :---: |
| $k = 2$ | 0.319 | 1.221 | 77.6 | 622.3 |
| **$k = 3$ (Optimal)** | **0.344** 🏆 | **1.070** 🏆 | **86.0** 🏆 | **440.6** 📉 *(Elbow)* |
| $k = 4$ | 0.280 | 1.250 | 73.5 | 381.8 |
| $k = 5$ | 0.247 | 1.378 | 67.4 | 336.2 |
| $k = 6$ | 0.256 | 1.304 | 63.3 | 300.9 |
| $k = 7$ | 0.244 | 1.368 | 58.2 | 280.0 |
| $k = 8$ | 0.241 | 1.373 | 54.0 | 262.1 |

- **Silhouette Coefficient:** Peaked at **0.344**, indicating optimal cohesion relative to separation.
- **Davies-Bouldin Index:** Minimum at **1.070**, reflecting well-isolated cluster boundaries.
- **Calinski-Harabasz Index:** Maximized at **86.0**, maximizing between-cluster variance over within-cluster variance.
- **Algorithmic Stability:** Hierarchical Clustering (Ward Linkage) demonstrates high consensus with K-Means (**ARI = 0.573**).

---

## 🌐 Discovered National Archetypes

| Cluster Archetype | Countries ($N$) | Avg. Happiness Score | GDP per capita | Social Support | Healthy Life Expectancy | Freedom of Choice | Generosity | Perceptions of Corruption |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier 1: High Wellbeing & Trust** | 28 | **6.73** (6.17 – 7.77) | 1.320 | 1.459 | 0.970 | 0.521 | 0.231 | **0.249** |
| **Tier 2: Moderate Wellbeing & Transition** | 82 | **5.60** (4.36 – 6.44) | 1.045 | 1.313 | 0.764 | 0.392 | 0.147 | **0.069** |
| **Tier 3: Vulnerable & Low Resource** | 46 | **4.26** (2.85 – 5.52) | 0.403 | 0.871 | 0.430 | 0.313 | **0.224** | 0.100 |

### Archetype Highlights

- **Tier 1: High Wellbeing & Institutional Trust (28 Nations)**
  - *Members:* Finland, Denmark, Norway, Iceland, Netherlands, Switzerland, New Zealand, Canada, Singapore, United Kingdom, etc.
  - *Signature:* High economic productivity, robust social safety nets, high life expectancy, and uniquely elevated **institutional trust** (corruption perception metric = 0.249 vs. global mean 0.111).
- **Tier 2: Transitional & Middle-Income Economies (82 Nations)**
  - *Members:* Mexico, Brazil, Argentina, Poland, Romania, Thailand, Colombia, Costa Rica, China, Russia, etc.
  - *Signature:* Solid economic performance and family/social support, but hindered by low public institutional trust (corruption perception metric = 0.069, lowest across all clusters).
- **Tier 3: Vulnerable & Resource-Constrained Nations (46 Nations)**
  - *Members:* South Sudan, Central African Republic, Afghanistan, Yemen, Rwanda, Tanzania, Haiti, Malawi, etc.
  - *Signature:* Severe constraints across basic material needs, life expectancy, and civil freedom. Notably, **Generosity** (0.224) remains strong, exceeding that of Tier 2 nations.

---

## 💡 Key Analytical Findings

1. **Institutional Trust as the Primary Differentiator:** While economic growth and health propel nations from Tier 3 to Tier 2, transparency and low corruption are the defining boundaries that separate Tier 2 from Tier 1.
2. **Prosocial Resilience:** Community generosity does not decline monotonically with economic hardship. Tier 3 nations demonstrate high relative prosocial engagement despite severe economic limitations.
3. **Monotonic External Alignment:** Although the clustering was completely unsupervised, external validation against the Cantril Ladder `Score` revealed clean, monotonic tier stratification with minimal inter-cluster overlap.

---

## 🚀 Getting Started & Reproducibility

### 1. Prerequisites
Ensure you have Python 3.10+ installed. Install the required dependencies:

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn nbformat nbconvert
```

### 2. Running the Analysis
You can explore the pre-rendered analysis directly by opening the Jupyter Notebook:

```bash
jupyter notebook World_Happiness_Advanced_Clustering.ipynb
```

---

## 📄 License
This project is open-source and available under the terms of the [MIT License](LICENSE).
