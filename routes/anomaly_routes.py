from flask import Blueprint, render_template, request

from model_loader import anomaly_model
from feature_engineering import read_inputs


anomaly_bp = Blueprint("anomaly", __name__)


@anomaly_bp.route("/anomaly", methods=["GET", "POST"])
def anomaly():
    anomaly_features = [
        "price",
        "recommended_upgrade_cost_est",
        "cpu_score",
        "gpu_score",
        "ram_gb",
        "total_storage_gb",
        "weight_kg"
    ]

    from flask import request

    if request.method == "GET":
        return render_template("anomaly.html", features=anomaly_features)

    input_df = read_inputs(anomaly_features)

    anomaly_prediction = anomaly_model.predict(input_df)[0]
    anomaly_score = anomaly_model.decision_function(input_df)[0]

    if anomaly_prediction == -1:
        label = "Unusual Laptop Record"
        message = "This laptop needs attention because it looks different from normal records."
    else:
        label = "Normal Laptop Record"
        message = "This laptop looks normal compared with the training data."

    result = {
        "score": round(float(anomaly_score), 4),
        "label": label,
        "message": message
    }

    return render_template("anomaly_result.html", result=result)