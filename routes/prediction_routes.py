from flask import Blueprint, render_template, request
import pandas as pd

from model_loader import (
    best_classifier,
    best_regressor,
    fair_classifier,
    fair_regressor,
    clf_preprocessor,
    reg_preprocessor,
    fair_clf_preprocessor,
    fair_reg_preprocessor,
    clf_features,
    reg_features
)

from feature_engineering import (
    read_simple_predict_inputs,
    build_model_input
)


prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    model_choice = request.form.get("model_choice")

    simple_data = read_simple_predict_inputs()

    clf_input = build_model_input(simple_data, clf_features)
    reg_input = build_model_input(simple_data, reg_features)

    if model_choice == "fair":
        clf_processed = fair_clf_preprocessor.transform(clf_input)
        reg_processed = fair_reg_preprocessor.transform(reg_input)

        upgrade_prediction = fair_classifier.predict(clf_processed)[0]
        cost_prediction = fair_regressor.predict(reg_processed)[0]

    else:
        clf_processed = clf_preprocessor.transform(clf_input)
        reg_processed = reg_preprocessor.transform(reg_input)

        upgrade_prediction = best_classifier.predict(clf_processed)[0]
        cost_prediction = best_regressor.predict(reg_processed)[0]

    result = {
        "model_choice": model_choice,
        "upgrade_prediction": upgrade_prediction,
        "cost_prediction": round(float(cost_prediction), 2),
        "ram_gap_gb": clf_input.get("ram_gap_gb", pd.Series([0])).iloc[0],
        "storage_gap_gb": clf_input.get("storage_gap_gb", pd.Series([0])).iloc[0],
        "gpu_gap_score": clf_input.get("gpu_gap_score", pd.Series([0])).iloc[0]
    }

    return render_template("predict_result.html", result=result)