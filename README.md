# Phishing Website Detection — Critical Reproduction Study

Final Project for **Data Science in Cyber** (Dr. Uri Itai).

This project critically evaluates a public phishing detection tutorial,
reproduces its results, fixes several methodological issues, and re-tests
the author's claims under a more realistic evaluation.

## The source under review

The tutorial being evaluated consists of **two artefacts by the same author,
Shreya Gopal Sundari**:

1. **The blog post** — a written walkthrough of the pipeline and results:
   *"Phishing Domain Detection with ML"*, published on Medium under the Intel
   Student Ambassadors publication.
   https://medium.com/intel-student-ambassadors/phishing-domain-detection-with-ml-5be9c99293e5

2. **The code implementation** — the notebook, feature-extraction scripts,
   and pre-extracted CSV:
   https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques

Throughout this project, "the article" refers to the blog post and "the
original repository" refers to the GitHub project. Together they constitute
the single source I am critically evaluating.

## Project description

Given a URL, the task is to classify it as **phishing** (1) or **legitimate**
(0). The article extracts 17 hand-crafted features from each URL (address-bar
features, domain/WHOIS features, HTML/JavaScript features) and trains 7
classifiers, reporting XGBoost as the best at ~86% test accuracy.

In this project I:

1. Reproduce the pipeline faithfully.
2. Replace the single 80/20 split with 5-fold stratified cross-validation.
3. Add feature scaling for scale-sensitive models (missing in the article).
4. Report a full metric suite (accuracy, precision, recall, F1, F2, ROC-AUC,
   MCC) and confusion matrices, instead of only accuracy.
5. Add feature creation (`URL_Complexity`) and 2D PCA visualisation.
6. Add error analysis and permutation feature importance.
7. Stress-test the best model under a realistic 10/90 class prior.
8. Run a de-duplication stress test after distinguishing between full-row
   duplicates and feature-vector duplicates.

## Dataset sources

- **Phishing URLs:** [PhishTank](https://phishtank.org).
- **Legitimate URLs:** University of New Brunswick URL dataset.

The pre-extracted CSV (`urldata.csv`) from the original repository is
included in this project for convenience — no external download required.

## Repository contents

```
.
├── README.md                       <- this file
├── report.pdf                      <- full PDF report (all 8 sections)
├── phishing_notebook.ipynb         <- end-to-end analysis notebook
├── urldata.csv                     <- 10,000 URLs × 17 features + label
├── requirements.txt                <- pinned Python dependencies
│
├── src/                            <- reusable modules
│   ├── __init__.py
│   ├── data.py                     <- load / split / duplicate analysis
│   ├── features.py                 <- feature engineering, Spearman helpers
│   ├── models.py                   <- model factory (scaled + tree models)
│   ├── evaluate.py                 <- CV wrapper + F2 / MCC scorers
│   └── plots.py                    <- correlation, PCA, importance plots
│
├── scripts/                        <- entry points that produce results/
│   ├── run_analysis.py             <- main CV + feature importance run
│   └── run_dedup_analysis.py       <- dirty vs clean stress test
│
├── results/                        <- outputs of the scripts
│   ├── cv_results.csv              <- 5-fold CV metrics table
│   ├── feature_importance.csv      <- permutation importance ranking
│   ├── dedup_comparison.csv        <- dirty vs clean side-by-side
│   ├── duplicate_summary.txt       <- both kinds of duplicate counts
│   └── plots/                      <- saved figures (PNGs)
│
└── tests/                          <- smoke tests for the src/ modules
    └── test_pipeline.py            <- run with: python -m pytest tests/
```

## Execution instructions

```bash
# 1. clone this repo
git clone https://github.com/jolian-jarrous/phishing-detection.git
cd phishing-detection

# 2. install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# 3. either:

#    (a) reproduce end-to-end with the notebook:
jupyter notebook phishing_notebook.ipynb

#    (b) or re-run just the analysis scripts:
python scripts/run_analysis.py
python scripts/run_dedup_analysis.py

# 4. (optional) run the smoke tests
python -m pytest tests/
```

All random operations are seeded with `RANDOM_STATE = 42`, so results
are reproducible across runs.

## Key findings (short version)

- The article's ~86% balanced-accuracy result reproduces cleanly.
- Under 5-fold cross-validation, **Random Forest wins, not XGBoost** as the
  article claims. Random Forest beats XGBoost on accuracy, recall, F1, F2,
  ROC-AUC, and MCC (though the gaps are small).
- Under a realistic 10% phishing prior, precision drops from 0.94 to 0.66 —
  roughly 1 in 3 alerts becomes a false alarm.
- The two most important features are `URL_Depth` (0.100) and `URL_Length`
  (0.092), not `Web_Traffic` as the article suggests. `URL_Complexity`,
  the engineered feature I added, lands at #3.
- **Biggest finding: the article's dataset has 10,000 unique URLs but
  only 880 unique feature vectors.** ~91% of the rows are feature-vector
  duplicates (different URLs collapsing to the same 17-feature representation).
  After de-duplicating on features, MCC collapses from 0.72 to 0.08–0.35
  depending on the model. The 86% headline is largely an artefact of feature
  duplication.
- Error analysis shows the model fails on "clean-looking" phishing URLs
  with no landing-page obfuscation — exactly the cases where a defender
  needs help.

See `report.pdf` for the full analysis.
