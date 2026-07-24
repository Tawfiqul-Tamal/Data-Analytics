# Cardiovascular Risk Prediction with Explainable Machine Learning

A complete, portfolio-ready ML project: predicting heart disease risk from self-reported
health indicators, comparing four classifiers, and using **SHAP** to explain every
prediction — not just give a black-box "yes/no."

**Author:** Md. Tawfiqul Islam Tamal — American International University-Bangladesh (AIUB)

---

## 1. What this project does

- Loads a heart-disease-risk dataset (columns matching the CDC BRFSS "Personal Key
  Indicators of Heart Disease" survey)
- Runs full EDA: class balance, risk-factor distributions, correlations
- Preprocesses: encoding, stratified split, **SMOTE** for the severe class imbalance
- Trains and compares 4 models: Logistic Regression, Random Forest, XGBoost, LightGBM
- Evaluates with ROC-AUC, PR-AUC, F1, and confusion matrices (not accuracy — with ~5-10%
  positive class, accuracy is meaningless here)
- Explains the best model with **SHAP** (global summary + individual patient waterfall
  plot)
- Includes a basic **fairness check** across sex subgroups

## 2. ✅ Now fully automatic — including in Google Colab

The notebook is now self-contained. When you run it, the very first cells:
1. **Auto-install** any missing packages (`xgboost`, `lightgbm`, `shap`, `imbalanced-learn`, `kagglehub`)
2. **Auto-download the real CDC dataset** via `kagglehub` — no manual Kaggle download,
   no file uploads needed. (You may be prompted to log in to Kaggle the first time.)
3. **Automatically fall back** to an inline-generated synthetic dataset (same column
   schema) if the real dataset can't be reached for any reason — so the notebook always
   completes without crashing.

### To run in Google Colab (recommended, zero setup)
1. Go to `https://colab.research.google.com`
2. **File → Upload notebook** → upload `notebooks/heart_disease_risk_prediction.ipynb`
3. **Runtime → Run all**
4. If prompted, log in to your Kaggle account to authorize the dataset download
5. Check the printed line near the top: `Data source: REAL Kaggle dataset (...)` — this
   confirms you're running on real data, not the synthetic fallback

### Note on the results already saved in this repo
The notebook and figures included in this download were executed in a sandboxed
environment with no access to kaggle.com, so they show the **synthetic fallback**
results (ROC-AUC ~0.60-0.62). Running it yourself in Colab (per above) will pull the
**real dataset** and should reach the published benchmark range of **ROC-AUC ~0.83-0.85**.

## 3. Project structure

```
heart_disease_project/
├── data/
│   └── heart_disease_real.csv         # auto-created here after the first successful run
│                                       # (downloaded from Kaggle, cached for next time)
├── notebooks/
│   ├── heart_disease_risk_prediction.ipynb   # the full analysis (already executed)
│   └── heart_disease_risk_prediction.html    # same notebook, viewable without Jupyter
├── outputs/
│   └── figures/                       # all 10 generated plots (EDA + evaluation + SHAP)
├── build_notebook.py                  # script that programmatically builds the .ipynb
├── requirements.txt
└── README.md                          # this file
```

## 4. How to run it (with auto-updating HTML)

**Important:** if you click "Run All" inside VS Code's notebook UI, only the `.ipynb`
file updates — the `.html` file stays as an old snapshot. To update **both** together
every time, use the provided script instead of the "Run All" button:

**On Mac/Linux**, open a terminal in the project folder and run:
```bash
bash run_project.sh
```

**On Windows**, double-click `run_project.bat` (or run it from Command Prompt).

This runs the whole notebook top to bottom AND regenerates the HTML export in one step,
so the two files never fall out of sync.

### Manual alternative
```bash
pip install -r requirements.txt
jupyter notebook notebooks/heart_disease_risk_prediction.ipynb
```
Or just open `notebooks/heart_disease_risk_prediction.html` in a browser to view the
already-executed results without installing anything (note: this won't reflect any
changes until you re-run via the script above).

## 5. Key results (synthetic run)

| Model | ROC-AUC | Avg Precision | F1-score |
|---|---|---|---|
| Random Forest | 0.621 | 0.070 | 0.117 |
| Logistic Regression | 0.609 | 0.071 | 0.120 |
| LightGBM | 0.602 | 0.064 | 0.105 |
| XGBoost | 0.596 | 0.063 | 0.101 |

*(Numbers will look substantially better — and more publication-worthy — once you swap
in the real dataset per Section 2.)*

SHAP analysis identified age, general health status, difficulty walking, diabetes status,
and stroke history as the strongest risk drivers — consistent with established
cardiovascular risk literature, which is a good sign the pipeline is behaving sensibly
even on synthetic data.

## 6. What to do next (in order)

1. **Run it in Colab** (Section 2) — this is now a single "Run all" with no manual file
   handling, and gets you real-data results instead of the synthetic fallback.
2. **Hyperparameter tuning** — try `Optuna` or `GridSearchCV` on the best model (likely
   XGBoost or LightGBM once real data shows their true signal).
3. **Probability calibration** — tree ensembles often output poorly calibrated
   probabilities; `CalibratedClassifierCV` fixes this and matters if you ever present
   risk *scores* (not just yes/no) to a clinical audience.
4. **Build a demo** — a small Streamlit or Gradio app where someone enters their health
   metrics and gets a risk score + SHAP explanation. This is what turns a notebook into
   something people can actually click through, and is a strong portfolio addition.
5. **Write it up** — a short GitHub README (like this one) with 3-4 key plots embedded,
   or a short Medium/LinkedIn post walking through the SHAP findings, goes a long way for
   visibility.
6. **Push to GitHub** — make sure `data/heart_disease_real.csv` is in `.gitignore` if you
   don't have redistribution rights to the raw CDC data; keep the synthetic generator and
   code public, link to the Kaggle source instead of committing the real CSV.

## 7. Notes on methodology choices (for your own understanding / if asked in an interview or viva)

- **Why SMOTE only on training data?** Applying SMOTE before the train/test split leaks
  synthetic near-duplicates of test-set points into training, inflating performance
  artificially. It's applied strictly after the split, on the training fold only.
- **Why ROC-AUC and PR-AUC instead of accuracy?** With ~90%+ of the data in one class,
  a model that always predicts "No disease" gets high accuracy while being clinically
  useless. AUC-based metrics are threshold-independent and far more informative here.
- **Why SHAP TreeExplainer, and why sample only 500 rows?** `TreeExplainer` is exact and
  fast for tree ensembles, but computing SHAP values scales with dataset size; a 500-row
  random sample gives stable, representative summary statistics without a multi-minute
  wait — standard practice for SHAP on large datasets.
