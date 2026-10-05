import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
RANDOM_STATE = 42
N_RECORDS = 253_680
LABELS = ["Healthy", "Pre-diabetes", "Diabetes"]
def generate_dataset(n_records: int = N_RECORDS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Build a synthetic patient dataset whose class balance, age spread,
    BMI categories and glucose spread resemble a CDC-style diabetes
    health-indicators dataset.

    To use REAL data instead: replace the body of this function with
    something like `pd.read_csv("diabetes_health_indicators.csv")` and
    rename/derive the columns below (Age, Gender, BMI, BloodGlucose,
    SystolicBP, DiastolicBP, PhysicalActivity, FamilyHistory,
    Cholesterol, Smoking, HbA1c, Diagnosis) to match.
    """
    rng = np.random.default_rng(seed)

    age = rng.choice(np.arange(1, 90), size=n_records, p=_age_weights())
    gender = rng.choice(["Female", "Male"], size=n_records, p=[0.552, 0.448])

    bmi = np.clip(rng.normal(loc=26 + age * 0.03, scale=5.5, size=n_records), 14, 55)
    physical_activity = rng.choice(["Low", "Moderate", "High"], size=n_records, p=[0.35, 0.45, 0.20])
    family_history = rng.choice(["Yes", "No"], size=n_records, p=[0.38, 0.62])
    systolic = np.clip(rng.normal(120 + age * 0.35, 12, n_records), 90, 200)
    diastolic = np.clip(systolic * 0.62 + rng.normal(0, 6, n_records), 55, 130)
    cholesterol = np.clip(rng.normal(190 + age * 0.5, 30, n_records), 100, 350)
    smoking = rng.choice(["Yes", "No"], size=n_records, p=[0.22, 0.78])
    hba1c = np.clip(rng.normal(5.3 + age * 0.006, 0.6, n_records), 3.5, 12)

    activity_score = np.select(
        [physical_activity == "Low", physical_activity == "Moderate", physical_activity == "High"],
        [1.0, 0.4, -0.4],
    )
    family_score = np.where(family_history == "Yes", 1.0, 0.0)
    smoking_score = np.where(smoking == "Yes", 0.4, 0.0)

    # Composite risk score -> drives glucose, which drives the label.
    # Weights loosely mirror the feature-importance ranking used in the
    # dashboard (glucose > BMI > age > activity > family history > BP
    # > cholesterol > smoking > HbA1c).
    risk_score = (
        0.019 * (bmi - 25)
        + 0.010 * (age - 40)
        + 0.55 * activity_score
        + 0.45 * family_score
        + 0.10 * smoking_score
        + 0.01 * (cholesterol - 190) / 10
        + 0.15 * (hba1c - 5.3)
        + rng.normal(0, 0.6, n_records)
    )
    glucose = np.clip(95 + risk_score * 28, 60, 300)

    label = np.select(
        [glucose < 100, (glucose >= 100) & (glucose < 126), glucose >= 126],
        LABELS,
        default=LABELS[0],
    )
    label = _rebalance(
        label,
        target={"Healthy": 0.5145, "Pre-diabetes": 0.2731, "Diabetes": 0.2124},
        rng=rng,
    )

    return pd.DataFrame({
        "Age": age.astype(int),
        "Gender": gender,
        "BMI": bmi.round(1),
        "BloodGlucose": glucose.round(0).astype(int),
        "SystolicBP": systolic.round(0).astype(int),
        "DiastolicBP": diastolic.round(0).astype(int),
        "PhysicalActivity": physical_activity,
        "FamilyHistory": family_history,
        "Cholesterol": cholesterol.round(0).astype(int),
        "Smoking": smoking,
        "HbA1c": hba1c.round(1),
        "Diagnosis": label,
    })


def _age_weights():
    """Rough weights so 0-20 / 21-30 / ... / 60+ mirror the dashboard bars."""
    buckets = {
        (1, 21): 0.03,
        (21, 31): 0.20,
        (31, 41): 0.27,
        (41, 51): 0.22,
        (51, 61): 0.18,
        (61, 90): 0.10,
    }
    w = np.zeros(89)
    for (lo, hi), weight in buckets.items():
        span = hi - lo
        w[lo - 1:hi - 1] = weight / span
    return w / w.sum()


def _rebalance(label, target, rng):
    """Randomly flips a few labels so class proportions match `target`."""
    label = label.copy()
    n = len(label)
    current = pd.Series(label).value_counts(normalize=True)
    for cls, target_p in target.items():
        want = int(target_p * n)
        have = int(current.get(cls, 0) * n)
        diff = want - have
        if diff > 0:
            others = np.where(label != cls)[0]
            pick = rng.choice(others, size=min(diff, len(others)), replace=False)
            label[pick] = cls
        current = pd.Series(label).value_counts(normalize=True)
    return label
def bmi_category(bmi):
    return pd.cut(
        bmi,
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obesity"],
    )
def train_model(df: pd.DataFrame):
    """
    Encode features, train a RandomForest classifier and return the
    model plus everything the dashboard needs: encoders, test-set
    metrics, the confusion matrix and feature importances.
    """
    data = df.copy()
    encoders = {}
    for col in ["Gender", "PhysicalActivity", "FamilyHistory", "Smoking"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le
    label_enc = LabelEncoder()
    label_enc.fit(LABELS)
    y = label_enc.transform(data["Diagnosis"])
    feature_cols = [
        "BloodGlucose", "BMI", "Age", "PhysicalActivity", "FamilyHistory",
        "SystolicBP", "Cholesterol", "Smoking", "HbA1c",
    ]
    X = data[feature_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"),
    }

    cm = confusion_matrix(y_test, y_pred)
    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    return {
        "model": model,
        "encoders": encoders,
        "label_enc": label_enc,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "confusion_matrix": cm,
        "importances": importances,
    }


def predict_risk(bundle, age, bmi, glucose, systolic, activity, family_history,
                  cholesterol=190, smoking="No", hba1c=5.5):
    """Turn Quick-Risk-Prediction form inputs into a class + probability."""
    encoders = bundle["encoders"]
    row = pd.DataFrame([{
        "BloodGlucose": glucose,
        "BMI": bmi,
        "Age": age,
        "PhysicalActivity": encoders["PhysicalActivity"].transform([activity])[0],
        "FamilyHistory": encoders["FamilyHistory"].transform([family_history])[0],
        "SystolicBP": systolic,
        "Cholesterol": cholesterol,
        "Smoking": encoders["Smoking"].transform([smoking])[0],
        "HbA1c": hba1c,
    }])[bundle["feature_cols"]]

    proba = bundle["model"].predict_proba(row)[0]
    pred_idx = proba.argmax()
    pred_label = bundle["label_enc"].inverse_transform([pred_idx])[0]
    return pred_label, float(proba[pred_idx]), dict(zip(bundle["label_enc"].classes_, proba))