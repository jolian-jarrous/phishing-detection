# Phishing Website Detection --- Critical Reproduction Study

Final Project for **Data Science in Cyber** (Dr. Uri Itai).

This project critically evaluates the article
*"Phishing Website Detection by Machine Learning Techniques"*
by Shreya Gopal Sundari, reproduces its results, fixes several methodological
issues, and re-tests the author's claims under a more realistic evaluation.

## Project description

Given a URL, the task is to classify it as **phishing** (1) or **legitimate** (0).
The original article extracts 17 hand-crafted features from each URL
(address-bar features, domain/WHOIS features, HTML/JavaScript features) and
trains 7 classifiers, reporting XGBoost as the best at ~86% test accuracy.

In this project I:
1. Reproduce the pipeline faithfully.
2. Replace the single 80/20 split with 5-fold stratified cross-validation.
3. Add feature scaling for scale-sensitive models (missing in the article).
4. Report a full metric suite (accuracy, precision, recall, F1, F2, ROC-AUC, MCC)
   and confusion matrices, instead of only accuracy.
5. Add feature creation (URL_Complexity) and PCA visualisation.
6. Add error analysis and permutation feature importance.
7. Stress-test the best model under a realistic 10/90 class prior.
8. Run a de-duplication stress test after finding 5,626 duplicate rows.

## Links

- **Selected article (Medium):**
  https://medium.com/intel-student-ambassadors/phishing-domain-detection-with-ml-5be9c99293e5
- **Original GitHub repository:**
  https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques
- **Dataset sources:** PhishTank (phishing URLs) + University of New Brunswick
  URL dataset (legitimate URLs). The pre-extracted CSV (`urldata.csv`) is
  included in this repository for convenience.

## Repository contents

​```
.
├── README.md                   <- this file
├── report.pdf                  <- full PDF report (8 sections + executive summary)
├── phishing_notebook.ipynb     <- Jupyter notebook: EDA, training, evaluation
├── requirements.txt            <- Python dependencies
└── urldata.csv                 <- the dataset (10,000 URLs, 17 features + label)
​```
## Execution instructions

```bash
# 1. clone this repo
git clone https://github.com/jolian-jarrous/phishing-detection.git
cd phishing-detection

# 2. install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# 3. open and run the notebook (urldata.csv is already in the repo)
jupyter notebook phishing_notebook.ipynb
```

All random operations are seeded with `RANDOM_STATE = 42` so results are
reproducible across runs.

## Key findings (short version)

- The article's ~86% balanced-accuracy result reproduces cleanly.
- Under 5-fold cross-validation, **Random Forest wins, not XGBoost** as the
  article claims. Random Forest beats XGBoost on accuracy, recall, F1, F2,
  ROC-AUC, and MCC, though the gaps are small.
- Under a realistic 10% phishing prior, precision drops from 0.94 to 0.66 ---
  roughly 1 in 3 alerts is a false alarm.
- The two most important features are `URL_Depth` (0.100) and `URL_Length` (0.092),
  not `Web_Traffic` as the article suggests. `URL_Complexity`, the engineered
  feature I added, lands at #3.
- **Biggest finding: the raw dataset contains 5,626 duplicate rows out of 10,000.**
  Only 880 unique feature vectors exist. After de-duplication, MCC collapses
  from 0.72 to between 0.08 and 0.35 depending on the model. The article's
  86% headline number is largely an artefact of duplicate contamination.
- Error analysis shows the model fails on "clean-looking" phishing URLs with
  no landing-page obfuscation --- exactly the cases where a defender needs help.

See `report.pdf` for the full analysis.
