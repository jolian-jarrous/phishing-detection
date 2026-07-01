# Phishing Website Detection --- Critical Reproduction Study

Final Project for **Data Science in Cyber** (Dr. Uri Itai).

This project critically evaluates the tutorial
*"Phishing Website Detection by Machine Learning Techniques"*
by Shreya Gopal Sundari, reproduces its results, fixes several methodological
issues, and re-tests the author's claims under a more realistic evaluation.

## Project description

Given a URL, the task is to classify it as **phishing** (1) or **legitimate** (0).
The original tutorial extracts 17 hand-crafted features from each URL
(address-bar features, domain/WHOIS features, HTML/JavaScript features) and
trains 7 classifiers, reporting XGBoost as the best at ~86% test accuracy.

In this project I:
1. Reproduce the pipeline faithfully.
2. Replace the single 80/20 split with 5-fold stratified cross-validation.
3. Add feature scaling for scale-sensitive models (was missing in the tutorial).
4. Report a full metric suite (accuracy, precision, recall, F1, ROC-AUC, MCC)
   and confusion matrices, instead of only accuracy.
5. Add error analysis and permutation feature importance.
6. Stress-test the best model under a realistic 10/90 class prior.

## Links

- **Selected article (Medium):**
  https://medium.com/intel-student-ambassadors/phishing-domain-detection-with-ml-5be9c99293e5
- **Original GitHub repository:**
  https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques
- **Dataset sources:** PhishTank (phishing URLs) + University of New Brunswick
  URL dataset (legitimate URLs). The pre-extracted CSV (`urldata.csv`) is
  available in the original repository.

## Repository contents

```
.
├── README.md                   <- this file
├── report.pdf                  <- full PDF report (all 8 required sections)
├── report.tex                  <- LaTeX source for the report
├── phishing_notebook.ipynb     <- Jupyter notebook: EDA, training, evaluation
└── requirements.txt            <- pinned dependencies
```

## Execution instructions

```bash
# 1. clone this repo
git clone <your-repo-url>
cd <your-repo>

# 2. install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# 3. download the dataset from the original tutorial repo:
#    https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques
#    place urldata.csv next to the notebook.

# 4. open and run the notebook
jupyter notebook phishing_notebook.ipynb
```

All random operations are seeded with `RANDOM_STATE = 42` so results are
reproducible across runs.

## Key findings (short version)

- The tutorial's ~86% balanced-accuracy result reproduces cleanly.
- Under 5-fold cross-validation, XGBoost, Random Forest, MLP, and Logistic
  Regression are **statistically tied** at ~86-87% accuracy --- the tutorial's
  ranking is within noise.
- Under a realistic 10% phishing prior, precision drops to ~0.70, meaning
  ~1 in 3 alerts is a false alarm.
- The dominant feature is `Web_Traffic` (Alexa rank), which is (a) a
  discontinued service and (b) arguably a leaky feature.
- Error analysis shows the model fails on "clean-looking" phishing URLs ---
  exactly the ones where a defender most needs help.

See `report.pdf` for the full analysis.
