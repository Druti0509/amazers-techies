# Business Entity Resolution Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![RapidFuzz](https://img.shields.io/badge/String%20Matching-RapidFuzz-orange.svg)](https://github.com/maxbachmann/RapidFuzz)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, modular Business Entity Resolution (ER) framework designed to deduplicate and match multi-source organizational datasets across global regions.

---

## 📌 Architecture Overview

The pipeline operates in four decoupled stages to balance scale and high accuracy:

1. **Preprocessing & Normalization (`utils.py`)**: standardizes business names and addresses by stripping corporate legal suffixes (e.g., *Inc*, *Corp*, *Ltd*, *GmbH*), normalizing whitespace, isolating numerical tokens, and extracting standardized ISO country codes.
2. **Candidate Pair Blocking (`blocking.py`)**: utilizes character-level $n$-gram TF-IDF vectorization and cosine similarity to reduce $O(N \times M)$ comparisons down to a high-recall candidate pool ($k=25$).
3. **Pairwise Feature Extraction (`features.py`)**: computes heavy-grained string alignment metrics (RapidFuzz ratios, partials, token sort/set), numerical digit set overlap (Jaccard similarity), and country-level binary flags.
4. **Classification & Output Formatting (`train_infer.py`)**: trains a LightGBM gradient-boosting classifier on ground truth data and applies a decision boundary threshold ($\tau = 0.78$) to balance Precision and Recall for output formatting.

---

## 📁 Repository Structure

```text
amazer-techies/
├── src/
│   ├── utils.py          # Preprocessing & text normalization routines
│   ├── blocking.py       # Batched TF-IDF candidate blocking
│   ├── features.py       # Pairwise feature engineering
│   ├── train_infer.py    # LightGBM model training & inference engine
│   └── main.py           # Command-line driver script
├── dataset/              # Place input datasets here (gitignored)
│   ├── train/
│   └── test/
├── output/               # Generated matching outputs (gitignored)
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
├── requirements.txt      # Python dependencies
├── .gitignore            # Git exclusion rules
└── README.md             # Project documentation






# amazers-techies
Amazon ML challenge
amazer-techies/
