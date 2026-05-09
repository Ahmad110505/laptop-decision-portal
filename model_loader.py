import os
import joblib


MODEL_DIR = "models"

def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    return joblib.load(path)


best_classifier = load_model("best_classifier_model.pkl")
best_regressor = load_model("best_regressor_model.pkl")

fair_classifier = load_model("fair_classifier.pkl")
fair_regressor = load_model("fair_regressor.pkl")

clf_preprocessor = load_model("clf_preprocessor.pkl")
reg_preprocessor = load_model("reg_preprocessor.pkl")

fair_clf_preprocessor = load_model("fair_clf_preprocessor.pkl")
fair_reg_preprocessor = load_model("fair_reg_preprocessor.pkl")

clf_features = load_model("classification_features.pkl")
reg_features = load_model("regression_features.pkl")

anomaly_model = load_model("isolation_forest_model.pkl")

q_table = load_model("q_table.pkl")
states = load_model("states.pkl")
actions = load_model("actions.pkl")