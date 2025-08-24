import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from collections import defaultdict

# 1) Load data
df = pd.read_csv("synthetic_claims_dataset.csv")

# 2) Basic checks
print(df.head(3))

print(df.isna().mean().sort_values(ascending=False).head(10))
print("Denied rate:", df["Denied"].mean().round(3))

# 3) Features/Targets
target_main = "Denied"
target_reason = "Denial_Reason"

# (Optional) ensure text columns are strings
str_cols = ["Patient_Gender","Patient_State","Payer_Type","Plan_Name","ICD10_Code","CPT_Code"]
for c in str_cols:
    df[c] = df[c].astype(str)

X = df.drop(columns=[target_main, "Denial_Reason", "Recommended_Fix"])
y = df[target_main]

# 4) Preprocess
numeric_features = ["Patient_Age","Claim_Amount","Days_To_Submit",
                    "Documentation_Complete","Prior_Authorization","Duplicate_Claim"]
categorical_features = ["Patient_Gender","Patient_State","Payer_Type","Plan_Name","ICD10_Code","CPT_Code"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 5) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 6) Baseline model: Logistic Regression (good calibration + interpretability)
base_clf = LogisticRegression(max_iter=200, class_weight="balanced")
clf = Pipeline(steps=[("prep", preprocess), ("clf", base_clf)])

# 7) Fit + metrics (probabilities via calibration)
calibrated = CalibratedClassifierCV(clf, cv=3, method="isotonic")
calibrated.fit(X_train, y_train)

proba_test = calibrated.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, proba_test)
ap  = average_precision_score(y_test, proba_test)
print(f"AUC-ROC: {auc:.3f} | AUC-PR: {ap:.3f}")

# Helper: evaluate thresholds
def evaluate_threshold(y_true, y_scores, thr):
    # Flag high-risk if score >= thr
    flag = (y_scores >= thr).astype(int)
    held_rate = flag.mean()
    # Among flagged, what fraction are actually denials?
    if flag.sum() > 0:
        precision = ( (flag==1) & (y_true==1) ).sum() / flag.sum()
    else:
        precision = np.nan
    # What fraction of all denials did we catch?
    if (y_true==1).sum() > 0:
        recall_denials_captured = ( (flag==1) & (y_true==1) ).sum() / (y_true==1).sum()
    else:
        recall_denials_captured = np.nan
    return held_rate, precision, recall_denials_captured

for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    held, prec, rec = evaluate_threshold(y_test.values, proba_test, thr)
    print(f"thr={thr:.1f} | held={held:.2f} | precision_on_held={prec:.2f} | denials_captured={rec:.2f}")

# Choose an operating threshold (example: 0.6); adjust based on above table
THRESHOLD = 0.6

# 8) Denial-reason model (train on denied rows only)
denied_rows = df[df[target_main] == 1].copy()
# Drop rows where reason is NaN (shouldn’t happen in our synthetic set)
denied_rows = denied_rows.dropna(subset=[target_reason])

Xr = denied_rows.drop(columns=[target_main, "Denial_Reason", "Recommended_Fix"])
yr = denied_rows[target_reason].astype(str)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.25, random_state=42, stratify=yr
)

reason_clf = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=200, class_weight="balanced", multi_class="auto"))
])
reason_clf.fit(Xr_train, yr_train)
print("\nReason model sample predictions:", reason_clf.predict(Xr_test.head(5)))

# 9) Recommendation mapping
fix_map = {
    "Missing Docs": "Attach all required documentation",
    "Invalid Code": "Correct ICD/CPT code",
    "Late Submission": "Submit within payer's deadline",
    "Duplicate": "Remove duplicate entries",
    "No Coverage": "Verify patient’s insurance coverage"
}
default_fix = "Review payer policy & required documentation"

# 10) Inference function (risk + reason + fix)
def predict_with_reasons_and_fixes(df_new):
    scores = calibrated.predict_proba(df_new)[:,1]
    flag = (scores >= THRESHOLD)
    out = df_new.copy()
    out["Denied_Prob"] = scores
    out["Flag_For_Review"] = flag

    # Predict reason only for flagged claims
    flagged_df = df_new[flag]
    if not flagged_df.empty:
        reasons = reason_clf.predict(flagged_df)
        # Map reasons to fixes
        fixes = [fix_map.get(r, default_fix) for r in reasons]
        out.loc[flag, "Predicted_Reason"] = reasons
        out.loc[flag, "Suggested_Fix"]    = fixes
    else:
        out["Predicted_Reason"] = np.nan
        out["Suggested_Fix"]    = np.nan
    return out

# 11) Demo on test set
demo = predict_with_reasons_and_fixes(X_test.copy())
print("\nPredictions preview:")
print(demo[["Denied_Prob","Flag_For_Review","Predicted_Reason","Suggested_Fix"]].head(10))
joblib.dump(calibrated, "denial_model.pkl")
joblib.dump(reason_clf, "reason_model.pkl")
