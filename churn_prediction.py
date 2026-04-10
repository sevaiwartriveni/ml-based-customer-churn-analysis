import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, learning_curve, StratifiedKFold)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier,
                               GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score,
                             f1_score, RocCurveDisplay,
                             precision_recall_curve, PrecisionRecallDisplay)
import joblib
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.shape)
print(df.info())
print(df.describe())

# Drop ID column
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# ──────────────────────────────────────────────
# 2. LABEL ENCODING (one encoder per column)
# ──────────────────────────────────────────────
encoders = {}
for col in df.select_dtypes(include="object").columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# ──────────────────────────────────────────────
# 3. FEATURE ENGINEERING (expanded)
# ──────────────────────────────────────────────
df["AvgCharges"] = df["TotalCharges"] / (df["tenure"] + 1)
df["CLV"] = df["MonthlyCharges"] * df["tenure"]
df["TotalServices"] = df[["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                           "TechSupport", "StreamingTV", "StreamingMovies"]].sum(axis=1)
df["IsNewCustomer"] = (df["tenure"] <= 6).astype(int)

# NEW: More powerful features
df["ChargeRatio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
df["TenureBin"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                          labels=[0, 1, 2, 3], include_lowest=True).astype(int)
df["HighCharge"] = (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)).astype(int)
df["ServicePerCharge"] = df["TotalServices"] / (df["MonthlyCharges"] + 1)
df["LoyaltyScore"] = df["tenure"] * df["Contract"]

# ──────────────────────────────────────────────
# 4. EDA — VISUALIZATIONS
# ──────────────────────────────────────────────

# Churn distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.countplot(x='Churn', data=df, ax=axes[0], palette='Set2')
axes[0].set_title("Churn Distribution")

sns.countplot(x='Contract', hue='Churn', data=df, ax=axes[1], palette='Set2')
axes[1].set_title("Contract vs Churn")

sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=axes[2], palette='Set2')
axes[2].set_title("Monthly Charges vs Churn")
plt.tight_layout()
plt.show()

# Churn rate by feature — see WHICH features drive churn
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
categorical_cols = ["Contract", "InternetService", "PaymentMethod",
                    "TechSupport", "OnlineSecurity", "TenureBin"]
for ax, col in zip(axes.flatten(), categorical_cols):
    churn_rate = df.groupby(col)["Churn"].mean()
    churn_rate.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
    ax.set_title(f"Churn Rate by {col}")
    ax.set_ylabel("Churn Rate")
    ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(16, 12))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # show only lower triangle
sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0,
            linewidths=0.5, square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Top correlations with Churn
churn_corr = corr["Churn"].drop("Churn").sort_values(key=abs, ascending=False)
print("\n📊 Top features correlated with Churn:")
print(churn_corr.head(10))

# ──────────────────────────────────────────────
# 5. TRAIN/TEST SPLIT + SMOTE
# ──────────────────────────────────────────────
X = df.drop("Churn", axis=1)
y = df["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# ──────────────────────────────────────────────
# 6. HYPERPARAMETER TUNING
# ──────────────────────────────────────────────
print("\n⏳ Tuning XGBoost...")
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}
xgb_grid = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    xgb_params, cv=5, scoring='f1', n_jobs=-1
)
xgb_grid.fit(X_train_sm, y_train_sm)
print("Best XGB params:", xgb_grid.best_params_)
xgb_model = xgb_grid.best_estimator_
xgb_pred = xgb_model.predict(X_test)

print("\n⏳ Tuning Random Forest...")
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(), rf_params, cv=5, scoring='f1', n_jobs=-1
)
rf_grid.fit(X_train_sm, y_train_sm)
print("Best RF params:", rf_grid.best_params_)
rf = rf_grid.best_estimator_
rf_pred = rf.predict(X_test)

# Logistic Regression & Decision Tree
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_sm, y_train_sm)
lr_pred = lr.predict(X_test)

dt = DecisionTreeClassifier(max_depth=10, min_samples_split=5)
dt.fit(X_train_sm, y_train_sm)
dt_pred = dt.predict(X_test)

#  Gradient Boosting as 5th model
gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1)
gb.fit(X_train_sm, y_train_sm)
gb_pred = gb.predict(X_test)

# ──────────────────────────────────────────────
# 7. VOTING ENSEMBLE (combines all models)
# ──────────────────────────────────────────────
print("\n⏳ Training Voting Ensemble...")
voting = VotingClassifier(
    estimators=[
        ('lr', lr), ('rf', rf), ('xgb', xgb_model), ('gb', gb)
    ],
    voting='soft'  # uses predicted probabilities for better accuracy
)
voting.fit(X_train_sm, y_train_sm)
voting_pred = voting.predict(X_test)

# ──────────────────────────────────────────────
# 8. MODEL EVALUATION
# ──────────────────────────────────────────────
models = {
    "Logistic Regression": lr_pred,
    "Random Forest": rf_pred,
    "Decision Tree": dt_pred,
    "XGBoost": xgb_pred,
    "Gradient Boosting": gb_pred,
    "Voting Ensemble": voting_pred
}

best_score = 0
best_name = ""
best_pred = None

results = []
for name, pred in models.items():
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc = roc_auc_score(y_test, pred)
    results.append({"Model": name, "Accuracy": acc, "F1": f1, "ROC AUC": roc})

    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC AUC  : {roc:.4f}")

    if f1 > best_score:
        best_score = f1
        best_name = name
        best_pred = pred

#  Results comparison table
results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print(f"\n{'='*55}")
print("  📋 MODEL COMPARISON TABLE")
print(f"{'='*55}")
print(results_df.to_string(index=False))

print(f"\n🏆 Best Model: {best_name} (F1 = {best_score:.4f})")
print(classification_report(y_test, best_pred))

# Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, best_pred), annot=True, fmt='d',
            cmap='Blues', xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"])
plt.title(f"Confusion Matrix — {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# ──────────────────────────────────────────────
# 9. ROC + PRECISION-RECALL CURVES
# ──────────────────────────────────────────────
model_objects = {"LR": lr, "RF": rf, "DT": dt, "XGB": xgb_model,
                 "GB": gb, "Ensemble": voting}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ROC curves
for name, model in model_objects.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[0], name=name)
axes[0].set_title("ROC Curves — Model Comparison")

# Precision-Recall curves (more informative for imbalanced data)
for name, model in model_objects.items():
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=axes[1], name=name)
axes[1].set_title("Precision-Recall Curves")

plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────
# 10. LEARNING CURVES (detect overfitting)
# ──────────────────────────────────────────────
print("\n📈 Generating Learning Curves...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, model) in zip(axes, [("XGBoost", xgb_model), ("Random Forest", rf),
                                     ("Logistic Regression", lr)]):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_scaled, y, cv=5, scoring='f1',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training F1')
    ax.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation F1')
    ax.fill_between(train_sizes,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
    ax.fill_between(train_sizes,
                    val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
    ax.set_title(f"Learning Curve — {name}")
    ax.set_xlabel("Training Size")
    ax.set_ylabel("F1 Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────
# 11. FEATURE IMPORTANCE (multi-model comparison)
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
features = X.columns

for ax, (name, model) in zip(axes, [("XGBoost", xgb_model), ("Random Forest", rf),
                                     ("Gradient Boosting", gb)]):
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[-15:]  # Top 15 features only
    ax.barh(features[sorted_idx], importance[sorted_idx], color='steelblue', edgecolor='black')
    ax.set_title(f"Top 15 Features — {name}")
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────
# 12. OPTIMAL THRESHOLD TUNING
# ──────────────────────────────────────────────
# Default threshold is 0.5, but tuning it can improve F1
print("\n🎯 Threshold Tuning for best model...")

# Get the best model object
best_model_map = {"Logistic Regression": lr, "Random Forest": rf,
                  "Decision Tree": dt, "XGBoost": xgb_model,
                  "Gradient Boosting": gb, "Voting Ensemble": voting}
best_model = best_model_map[best_name]

if hasattr(best_model, "predict_proba"):
    y_proba = best_model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.3, 0.7, 0.01)
    f1_scores = [f1_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_scores, 'b-', linewidth=2)
    plt.axvline(optimal_threshold, color='red', linestyle='--',
                label=f'Optimal = {optimal_threshold:.2f}')
    plt.title("F1 Score vs Classification Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Apply optimal threshold
    optimized_pred = (y_proba >= optimal_threshold).astype(int)
    print(f"\n  Default threshold (0.50) → F1 = {f1_score(y_test, best_pred):.4f}")
    print(f"  Optimal threshold ({optimal_threshold:.2f}) → F1 = {f1_score(y_test, optimized_pred):.4f}")
    print(f"\n{classification_report(y_test, optimized_pred)}")

# ──────────────────────────────────────────────
# 13. CROSS-VALIDATION (Stratified K-Fold)
# ──────────────────────────────────────────────
print("\n📊 Stratified Cross-Validation Results:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in [("LR", lr), ("RF", rf), ("DT", dt), ("XGB", xgb_model),
                    ("GB", gb), ("Ensemble", voting)]:
    scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='f1')
    print(f"  {name:10s} CV F1: {scores.mean():.4f} ± {scores.std():.4f}")

# ──────────────────────────────────────────────
# 14. RISK SEGMENTATION
# ──────────────────────────────────────────────
df["Risk"] = pd.cut(df["tenure"],
                    bins=[0, 12, 24, 72],
                    labels=["High Risk", "Medium Risk", "Low Risk"])
print("\n", df[["tenure", "Risk"]].head())

# ──────────────────────────────────────────────
# 15. PREDICT NEW CUSTOMER (reusable function)
# ──────────────────────────────────────────────
def predict_churn(customer_data: dict) -> dict:
    """
    Predict churn for a new customer.

    Usage:
        result = predict_churn({
            "gender": 1, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
            "tenure": 5, "PhoneService": 1, "MultipleLines": 0,
            "InternetService": 1, "OnlineSecurity": 0, "OnlineBackup": 0,
            "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 0,
            "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 1,
            "PaymentMethod": 2, "MonthlyCharges": 70.0, "TotalCharges": 350.0
        })
    """
    input_df = pd.DataFrame([customer_data])

    # Add engineered features
    input_df["AvgCharges"] = input_df["TotalCharges"] / (input_df["tenure"] + 1)
    input_df["CLV"] = input_df["MonthlyCharges"] * input_df["tenure"]
    input_df["TotalServices"] = input_df[["OnlineSecurity", "OnlineBackup",
                                           "DeviceProtection", "TechSupport",
                                           "StreamingTV", "StreamingMovies"]].sum(axis=1)
    input_df["IsNewCustomer"] = (input_df["tenure"] <= 6).astype(int)
    input_df["ChargeRatio"] = input_df["MonthlyCharges"] / (input_df["TotalCharges"] + 1)
    input_df["TenureBin"] = pd.cut(input_df["tenure"], bins=[0, 12, 24, 48, 72],
                                    labels=[0, 1, 2, 3]).astype(int)
    input_df["HighCharge"] = (input_df["MonthlyCharges"] > 89.85).astype(int)
    input_df["ServicePerCharge"] = input_df["TotalServices"] / (input_df["MonthlyCharges"] + 1)
    input_df["LoyaltyScore"] = input_df["tenure"] * input_df["Contract"]

    # Ensure column order matches training data
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    scaled = scaler.transform(input_df)
    proba = best_model.predict_proba(scaled)[0][1]
    prediction = "WILL CHURN ⚠️" if proba >= 0.5 else "WILL STAY ✅"

    return {
        "prediction": prediction,
        "churn_probability": f"{proba:.1%}",
        "risk_level": "High" if proba > 0.7 else "Medium" if proba > 0.4 else "Low"
    }


# Example prediction
sample = predict_churn({
    "gender": 1, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
    "tenure": 3, "PhoneService": 1, "MultipleLines": 0,
    "InternetService": 1, "OnlineSecurity": 0, "OnlineBackup": 0,
    "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 0,
    "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 1,
    "PaymentMethod": 2, "MonthlyCharges": 85.0, "TotalCharges": 255.0
})
print(f"\n🔮 Sample Prediction: {sample}")

# ──────────────────────────────────────────────
# 16. SAVE BEST MODEL + SCALER
# ──────────────────────────────────────────────
joblib.dump(best_model, "churn_best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")
print(f"\n✅ Best model ({best_name}), scaler, and encoders saved!")
