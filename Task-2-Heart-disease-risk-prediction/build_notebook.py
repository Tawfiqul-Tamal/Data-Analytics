import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ---------------------------------------------------------------------------
md("""# Cardiovascular Risk Prediction with Explainable Machine Learning

**Author:** Md. Tawfiqul Islam Tamal
**Institution:** American International University-Bangladesh (AIUB)

## Problem Statement
Cardiovascular disease is one of the leading causes of death worldwide. Many of its risk
factors (smoking, BMI, physical inactivity, diabetes, general health status, etc.) are
routinely captured in public health surveys. This project builds and compares several
machine learning classifiers to predict heart disease risk from these self-reported
health indicators, and uses **SHAP (SHapley Additive exPlanations)** to make the
predictions interpretable — critical in any healthcare application where a black-box
"yes/no" is not good enough on its own.

## Dataset
Based on the **CDC BRFSS "Personal Key Indicators of Heart Disease"** dataset
(Kaggle: `kamilpytlak/personal-key-indicators-of-heart-disease`). This notebook loads
the **real dataset automatically** via `kagglehub` — no manual download needed. If for
any reason the real data can't be reached (no internet, no Kaggle login, restricted
environment), it automatically falls back to an inline-generated synthetic dataset with
the same schema, so the notebook always completes end-to-end.

## Pipeline
1. Data loading & cleaning
2. Exploratory Data Analysis (EDA)
3. Preprocessing (encoding, imbalance handling)
4. Model training — Logistic Regression, Random Forest, XGBoost, LightGBM
5. Evaluation (ROC-AUC, F1, precision/recall, confusion matrices)
6. Explainability with SHAP
7. Conclusions & next steps
""")

# ---------------------------------------------------------------------------
md("## 1. Setup & Data Loading")

code("""# Installs needed packages if missing (safe to re-run; skips anything already installed)
import importlib
import subprocess
import sys

required = ["xgboost", "lightgbm", "shap", "imbalanced-learn", "kagglehub"]
for pkg in required:
    import_name = {"imbalanced-learn": "imblearn"}.get(pkg, pkg)
    try:
        importlib.import_module(import_name)
        continue
    except ImportError:
        pass
    # Try a normal pip install first; if the environment is "externally managed"
    # (some Linux/sandbox setups), retry with --break-system-packages.
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg],
                             capture_output=True, text=True)
    if result.returncode != 0:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                         "--break-system-packages", pkg], check=False)
print("Package setup complete.")
""")

code("""import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score, classification_report,
                              confusion_matrix, precision_recall_curve, average_precision_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import shap

sns.set_theme(style="whitegrid", palette="Set2")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
""")

md('''### Data source

This cell tries the **real CDC "Personal Key Indicators of Heart Disease" dataset**
first (via `kagglehub`, no manual download needed — it fetches straight from Kaggle's
servers). If that fails for any reason (no internet, no Kaggle account, running in a
locked-down environment, etc.), it automatically falls back to a **synthetic dataset
generated inline** with the same columns, so the notebook always runs end-to-end
regardless of environment.''')

code('''def generate_synthetic_heart_data(n=50000, random_state=42):
    """Self-contained synthetic data generator — same schema as the real CDC dataset.
    Used only as a fallback if the real Kaggle dataset can't be downloaded."""
    rng = np.random.RandomState(random_state)

    age_categories = ["18-24","25-29","30-34","35-39","40-44","45-49","50-54",
                       "55-59","60-64","65-69","70-74","75-79","80 or older"]
    age_midpoints = {"18-24":21,"25-29":27,"30-34":32,"35-39":37,"40-44":42,"45-49":47,
                      "50-54":52,"55-59":57,"60-64":62,"65-69":67,"70-74":72,"75-79":77,
                      "80 or older":85}
    p_age = [0.09,0.08,0.08,0.08,0.08,0.08,0.09,0.09,0.09,0.08,0.06,0.05,0.05]
    age_cat = rng.choice(age_categories, size=n, p=p_age)
    age_years = np.array([age_midpoints[a] for a in age_cat])

    sex = rng.choice(["Male", "Female"], size=n, p=[0.48, 0.52])
    race = rng.choice(["White","Black","Asian","Hispanic","American Indian/Alaskan Native","Other"],
                       size=n, p=[0.60,0.15,0.06,0.13,0.03,0.03])
    bmi = np.clip(rng.normal(28, 6.5, n), 12, 60)
    smoking = rng.choice(["Yes", "No"], size=n, p=[0.19, 0.81])
    alcohol = rng.choice(["Yes", "No"], size=n, p=[0.06, 0.94])
    stroke = rng.choice(["Yes", "No"], size=n, p=[0.04, 0.96])
    physical_health = np.clip(rng.exponential(3, n), 0, 30).round().astype(int)
    mental_health = np.clip(rng.exponential(3.5, n), 0, 30).round().astype(int)
    diff_walking = rng.choice(["Yes", "No"], size=n, p=[0.14, 0.86])
    diabetic = rng.choice(["Yes","No","No, borderline diabetes","Yes (during pregnancy)"],
                            size=n, p=[0.13,0.82,0.03,0.02])
    physical_activity = rng.choice(["Yes", "No"], size=n, p=[0.77, 0.23])
    gen_health = rng.choice(["Excellent","Very good","Good","Fair","Poor"],
                              size=n, p=[0.18,0.34,0.29,0.13,0.06])
    sleep_time = np.clip(rng.normal(7, 1.5, n), 1, 16).round().astype(int)
    asthma = rng.choice(["Yes", "No"], size=n, p=[0.14, 0.86])
    kidney_disease = rng.choice(["Yes", "No"], size=n, p=[0.037, 0.963])
    skin_cancer = rng.choice(["Yes", "No"], size=n, p=[0.09, 0.91])

    risk = (0.045*(age_years-40) + 0.05*(bmi-25) + 0.9*(smoking=="Yes") + 1.1*(stroke=="Yes")
            + 0.5*(diff_walking=="Yes") + 0.6*(diabetic=="Yes")
            + 0.3*((gen_health=="Fair")|(gen_health=="Poor")) + 0.4*(kidney_disease=="Yes")
            + 0.3*(asthma=="Yes") - 0.35*(physical_activity=="Yes") + 0.02*physical_health
            + 0.35*(sex=="Male") - 4.6)
    prob = 1 / (1 + np.exp(-risk))
    heart_disease = np.where(rng.binomial(1, prob) == 1, "Yes", "No")

    return pd.DataFrame({
        "HeartDisease": heart_disease, "BMI": bmi.round(2), "Smoking": smoking,
        "AlcoholDrinking": alcohol, "Stroke": stroke, "PhysicalHealth": physical_health,
        "MentalHealth": mental_health, "DiffWalking": diff_walking, "Sex": sex,
        "AgeCategory": age_cat, "Race": race, "Diabetic": diabetic,
        "PhysicalActivity": physical_activity, "GenHealth": gen_health,
        "SleepTime": sleep_time, "Asthma": asthma, "KidneyDisease": kidney_disease,
        "SkinCancer": skin_cancer,
    })
''')

code('''# Local folders: use the project's existing data/ and outputs/ folders if present
# (running from the full downloaded project), otherwise create local ones (e.g. when
# this notebook is run standalone in Colab with no other project files).
DATA_DIR = "../data" if os.path.isdir("../data") else "data"
FIGURES_DIR = "../outputs/figures" if os.path.isdir("../outputs") else "outputs/figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

LOCAL_REAL_CSV = os.path.join(DATA_DIR, "heart_disease_real.csv")
print(f"Data folder: {DATA_DIR}")
print(f"Figures folder: {FIGURES_DIR}")
''')

code('''df = None
data_source = None

# 1) If we already downloaded the real dataset before, just reuse the local copy —
#    faster, and works even without internet access.
if os.path.exists(LOCAL_REAL_CSV):
    df = pd.read_csv(LOCAL_REAL_CSV)
    data_source = f"REAL Kaggle dataset (cached copy: {LOCAL_REAL_CSV})"

else:
    # 2) Otherwise, try downloading the real dataset via kagglehub
    try:
        import kagglehub
        kaggle_path = kagglehub.dataset_download("kamilpytlak/personal-key-indicators-of-heart-disease")

        csv_files = []
        for root, dirs, files in os.walk(kaggle_path):
            for f in files:
                if f.endswith(".csv"):
                    csv_files.append(os.path.join(root, f))

        if not csv_files:
            raise FileNotFoundError("No CSV files found in the downloaded Kaggle dataset.")

        # Prefer a 2020-dated file if present, otherwise take the first CSV found
        target_csv = next((f for f in csv_files if "2020" in f), csv_files[0])
        df = pd.read_csv(target_csv)

        # Save a permanent local copy into the project's data/ folder, so next time
        # this cell runs it loads instantly from disk instead of downloading again.
        df.to_csv(LOCAL_REAL_CSV, index=False)
        data_source = f"REAL Kaggle dataset (downloaded and saved to: {LOCAL_REAL_CSV})"

    except Exception as e:
        print(f"Could not load the real Kaggle dataset ({type(e).__name__}: {e})")
        print("Falling back to an inline synthetic dataset with the same schema...")
        df = generate_synthetic_heart_data(n=50000, random_state=RANDOM_STATE)
        data_source = "SYNTHETIC dataset (generated inline, not saved to disk)"

print(f"\\nData source: {data_source}")
print(f"Shape: {df.shape}")
df.head()
''')

code("""df.info()
""")

code("""print("Missing values per column:")
print(df.isnull().sum())
print()
print("Target distribution:")
print(df['HeartDisease'].value_counts())
print(df['HeartDisease'].value_counts(normalize=True).round(4) * 100, "%")
""")

# ---------------------------------------------------------------------------
md("""## 2. Exploratory Data Analysis

The target class is heavily **imbalanced** (roughly 90-95% "No" vs. 5-10% "Yes"), which
is realistic for population health surveys and must be handled carefully — plain accuracy
would be a misleading metric here.""")

code("""fig, ax = plt.subplots(figsize=(5, 4))
df['HeartDisease'].value_counts().plot(kind='bar', ax=ax, color=['#4C956C', '#D7263D'])
ax.set_title('Class Distribution: Heart Disease')
ax.set_xlabel('Heart Disease')
ax.set_ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/01_class_distribution.png', dpi=150)
plt.show()
""")

code("""numeric_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.flatten(), numeric_cols):
    sns.histplot(data=df, x=col, hue='HeartDisease', kde=True, ax=ax, element='step', stat='density', common_norm=False)
    ax.set_title(f'Distribution of {col} by Heart Disease status')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/02_numeric_distributions.png', dpi=150)
plt.show()
""")

code("""fig, ax = plt.subplots(figsize=(9, 5))
rate_by_age = df.groupby('AgeCategory')['HeartDisease'].apply(lambda x: (x == 'Yes').mean()).sort_index()
rate_by_age.plot(kind='bar', ax=ax, color='#3A6EA5')
ax.set_title('Heart Disease Rate by Age Category')
ax.set_ylabel('Proportion with Heart Disease')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/03_rate_by_age.png', dpi=150)
plt.show()
""")

code("""categorical_risk_cols = ['Smoking', 'Stroke', 'DiffWalking', 'Diabetic', 'PhysicalActivity', 'GenHealth']
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.flatten(), categorical_risk_cols):
    rate = df.groupby(col)['HeartDisease'].apply(lambda x: (x == 'Yes').mean())
    rate.plot(kind='bar', ax=ax, color='#D7263D')
    ax.set_title(f'Heart Disease Rate by {col}')
    ax.set_ylabel('Rate')
    ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/04_categorical_risk_rates.png', dpi=150)
plt.show()
""")

code("""# Correlation heatmap for numeric features
plt.figure(figsize=(6, 5))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap (Numeric Features)')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/05_correlation_heatmap.png', dpi=150)
plt.show()
""")

# ---------------------------------------------------------------------------
md("""## 3. Preprocessing

Steps:
- Encode the binary target (`HeartDisease`: Yes/No -> 1/0)
- Encode categorical predictors (label encoding for tree models works well; one-hot would
  also be valid but explodes dimensionality with `AgeCategory`/`Race`)
- Stratified train/test split to preserve class ratio
- Apply **SMOTE** on the training set only (never on test data, to avoid leakage) to
  address class imbalance
""")

code("""target_col = 'HeartDisease'
y = (df[target_col] == 'Yes').astype(int)
X = df.drop(columns=[target_col])

categorical_cols = X.select_dtypes(include='object').columns.tolist()
print("Categorical columns:", categorical_cols)

encoders = {}
X_encoded = X.copy()
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train positive rate: {y_train.mean():.4f} | Test positive rate: {y_test.mean():.4f}")
""")

code("""smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"After SMOTE:  {pd.Series(y_train_res).value_counts().to_dict()}")
""")

# ---------------------------------------------------------------------------
md("""## 4. Model Training — Ensemble Comparison

We train four classifiers and compare them on the **untouched, original-distribution
test set** (SMOTE is only ever applied to training data):

1. Logistic Regression (baseline, interpretable)
2. Random Forest
3. XGBoost
4. LightGBM
""")

code("""models = {}

# 1. Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(X_train_scaled, y_train_res)
models['Logistic Regression'] = (log_reg, X_test_scaled)

# 2. Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_res, y_train_res)
models['Random Forest'] = (rf, X_test)

# 3. XGBoost
xgb_clf = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.08,
    eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1
)
xgb_clf.fit(X_train_res, y_train_res)
models['XGBoost'] = (xgb_clf, X_test)

# 4. LightGBM
lgb_clf = lgb.LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.08,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
)
lgb_clf.fit(X_train_res, y_train_res)
models['LightGBM'] = (lgb_clf, X_test)

print("All models trained.")
""")

# ---------------------------------------------------------------------------
md("""## 5. Evaluation

Because of the class imbalance, **accuracy is not a meaningful metric** here (predicting
"No" for everyone would already give ~90%+ accuracy). We instead focus on:
- **ROC-AUC** — ranking quality across thresholds
- **Average Precision (PR-AUC)** — more informative than ROC-AUC under imbalance
- **F1-score** — balance of precision/recall at the default threshold
- **Confusion matrices** — to see the actual false negative rate (missed heart disease
  cases), which is the costliest error in a clinical screening context
""")

code("""results = []
roc_data = {}
pr_data = {}

for name, (model, X_te) in models.items():
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc)

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    pr_data[name] = (prec, rec, ap)

    results.append({'Model': name, 'ROC-AUC': auc, 'Avg Precision': ap, 'F1-score': f1})

results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
results_df
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for name, (fpr, tpr, auc) in roc_data.items():
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves')
axes[0].legend()

for name, (prec, rec, ap) in pr_data.items():
    axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})")
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curves')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/06_roc_pr_curves.png', dpi=150)
plt.show()
""")

code("""fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (name, (model, X_te)) in zip(axes, models.items()):
    y_pred = model.predict(X_te)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/07_confusion_matrices.png', dpi=150)
plt.show()
""")

code("""best_model_name = results_df.iloc[0]['Model']
print(f"Best model by ROC-AUC: {best_model_name}\\n")
best_model, best_X_test = models[best_model_name]
print(classification_report(y_test, best_model.predict(best_X_test), target_names=['No Disease', 'Heart Disease']))
""")

# ---------------------------------------------------------------------------
md("""## 6. Explainability with SHAP

This is the part that turns a black-box classifier into something a clinician (or a
reviewer) can actually trust. We use the best tree-based model for SHAP analysis since
`TreeExplainer` is fast and exact for tree ensembles.
""")

code("""# Pick the best tree-based model for SHAP (TreeExplainer needs a tree model)
tree_models = {k: v for k, v in models.items() if k in ['Random Forest', 'XGBoost', 'LightGBM']}
shap_model_name = results_df[results_df['Model'].isin(tree_models.keys())].iloc[0]['Model']
shap_model, shap_X_test_full = tree_models[shap_model_name]
print(f"Using {shap_model_name} for SHAP explainability")

# SHAP on the full test set can be very slow (especially TreeExplainer on Random Forest).
# A random sample of a few hundred rows gives statistically stable summary plots and is
# standard practice for SHAP analysis on large datasets.
SHAP_SAMPLE_SIZE = 500
shap_X_test = shap_X_test_full.sample(n=min(SHAP_SAMPLE_SIZE, len(shap_X_test_full)), random_state=RANDOM_STATE)

explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(shap_X_test)

# Handle differing SHAP output formats across versions:
# - older shap: list of two arrays [class0_shap, class1_shap]
# - newer shap: single ndarray of shape (n_samples, n_features, n_classes)
if isinstance(shap_values, list):
    shap_values_plot = shap_values[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_values_plot = shap_values[:, :, 1]
else:
    shap_values_plot = shap_values

print(f"shap_values_plot shape: {shap_values_plot.shape} (should be (n_samples, n_features))")
""")

code("""plt.figure()
shap.summary_plot(shap_values_plot, shap_X_test, show=False)
plt.title(f'SHAP Summary — {shap_model_name}')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/08_shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()
""")

code("""plt.figure()
shap.summary_plot(shap_values_plot, shap_X_test, plot_type='bar', show=False)
plt.title(f'SHAP Feature Importance — {shap_model_name}')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/09_shap_importance_bar.png', dpi=150, bbox_inches='tight')
plt.show()
""")

code("""# Local explanation: why did the model flag ONE specific patient as high risk?
high_risk_idx = int(np.argmax(shap_model.predict_proba(shap_X_test)[:, 1]))
print(f"Patient row {high_risk_idx} (within SHAP sample) — predicted risk: {shap_model.predict_proba(shap_X_test)[:, 1][high_risk_idx]:.3f}")
print(shap_X_test.iloc[high_risk_idx])
""")

code("""expected_value = explainer.expected_value
if isinstance(expected_value, (list, np.ndarray)):
    ev_arr = np.atleast_1d(expected_value)
    expected_value = float(ev_arr[1]) if len(ev_arr) > 1 else float(ev_arr[0])
else:
    expected_value = float(expected_value)

# Waterfall plot: modern, robust way to show a single prediction's explanation
# (replaces the older force_plot API, which has compatibility issues across shap versions)
explanation = shap.Explanation(
    values=shap_values_plot[high_risk_idx],
    base_values=expected_value,
    data=shap_X_test.iloc[high_risk_idx].values,
    feature_names=shap_X_test.columns.tolist()
)

plt.figure()
shap.plots.waterfall(explanation, show=False)
plt.title(f'SHAP Waterfall — Individual Prediction (idx={high_risk_idx})')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/10_shap_waterfall_individual.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ---------------------------------------------------------------------------
md("""## 7. Fairness Check (Optional but recommended)

A quick sanity check: does the best model perform consistently across sex? This matters
in any healthcare ML deployment — a model that works well on average but poorly for a
subgroup is a real risk.
""")

code("""X_test_with_sex = X_test.copy()
X_test_with_sex['Sex_label'] = X['Sex'].iloc[X_test.index].values
X_test_with_sex['y_true'] = y_test.values
X_test_with_sex['y_proba'] = shap_model.predict_proba(shap_X_test_full)[:, 1]

fairness_rows = []
for sex_val in X_test_with_sex['Sex_label'].unique():
    subset = X_test_with_sex[X_test_with_sex['Sex_label'] == sex_val]
    auc_sub = roc_auc_score(subset['y_true'], subset['y_proba'])
    fairness_rows.append({'Sex': sex_val, 'N': len(subset), 'ROC-AUC': auc_sub})

pd.DataFrame(fairness_rows)
""")

# ---------------------------------------------------------------------------
md("""## 8. Conclusions & Next Steps

**Key findings** (will vary slightly by random seed / dataset used):
- Class imbalance is the central challenge; SMOTE + threshold-aware metrics (ROC-AUC,
  PR-AUC, F1) were used instead of accuracy.
- Tree-based ensembles (XGBoost / LightGBM / Random Forest) outperform plain Logistic
  Regression, but at some cost to interpretability — which is why SHAP is included.
- SHAP identifies the top clinical risk drivers (age, general health status, stroke
  history, diabetes status, difficulty walking) as the strongest predictors, consistent
  with established cardiovascular risk literature.

**Next steps for extending this project:**
1. Swap in the **real Kaggle dataset** (see the note in Section 1) and re-run — results
   will be directly comparable to published benchmarks on this dataset.
2. Hyperparameter tuning (Optuna / GridSearchCV) for the best-performing model.
3. Try model calibration (`CalibratedClassifierCV`) since raw probabilities from
   tree ensembles are often poorly calibrated.
4. Build a simple **Streamlit or Gradio demo**: user enters health metrics, gets a risk
   score + SHAP explanation for that individual prediction.
5. Package this as a short technical report / blog post for your portfolio (GitHub README
   with the key plots embedded goes a long way).
""")

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"}
}

with open('notebooks/heart_disease_risk_prediction.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook built successfully.")
