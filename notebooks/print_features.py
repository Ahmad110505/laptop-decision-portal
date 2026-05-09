import joblib
from pathlib import Path

artifact_dir = Path("c:/Users/ahmad/Desktop/uni/3rd year/2nd sem/applied/laptop_decision_portal/notebooks/artifacts")

clf_features = joblib.load(artifact_dir / "classification_features.pkl")
reg_features = joblib.load(artifact_dir / "regression_features.pkl")

print("CLF features:", clf_features)
print("REG features:", reg_features)
