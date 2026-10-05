# HealthCare Analytics Dashboard — Python/Streamlit

Recreates the dashboard: KPI cards, risk/age/BMI/glucose/gender charts,
a Quick Risk Prediction form (RandomForest), and a Model Performance
tab (accuracy/precision/recall/F1/ROC-AUC, confusion matrix, feature
importances).

## Files
- `utils.py` — synthetic data generator + RandomForest training/prediction
- `app.py` — the Streamlit UI
- `requirements.txt` — dependencies

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```
It opens at `http://localhost:8501`.

## Using real data instead of synthetic data
`generate_dataset()` in `utils.py` currently fabricates 253,680 patient
rows so the app runs out-of-the-box. To use a real dataset (e.g. the
CDC/UCI "Diabetes Health Indicators" dataset), replace the body of
`generate_dataset()` with:
```python
df = pd.read_csv("your_file.csv")
```
then rename/derive its columns to match: `Age, Gender, BMI,
BloodGlucose, SystolicBP, DiastolicBP, PhysicalActivity,
FamilyHistory, Cholesterol, Smoking, HbA1c, Diagnosis` (Diagnosis ∈
`Healthy / Pre-diabetes / Diabetes`). Everything downstream
(`train_model`, the charts, the prediction form) works unchanged.

## Notes
- The model is a `RandomForestClassifier` (300 trees, cached with
  `@st.cache_resource` so it only trains once per session).
- Reported metrics will differ from the screenshot since the
  synthetic data isn't identical to whatever generated the original
  numbers — swap in real data for a faithful match.