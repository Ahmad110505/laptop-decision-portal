import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request


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


app = Flask(__name__)


def get_all_features():
    features = []

    for col in clf_features:
        if col not in features:
            features.append(col)

    for col in reg_features:
        if col not in features:
            features.append(col)

    return features


def read_inputs(features):
    data = {}

    for col in features:
        value = request.form.get(col)

        try:
            data[col] = float(value)
        except:
            data[col] = value

    return pd.DataFrame([data])
def read_simple_predict_inputs():
    data = {}

    data["user_profile"] = request.form.get("user_profile")
    data["budget_class"] = request.form.get("budget_class")
    data["laptop_type"] = request.form.get("laptop_type")
    data["brand"] = request.form.get("brand")
    data["cpu_brand"] = request.form.get("cpu_brand")
    data["gpu_brand"] = request.form.get("gpu_brand")

    data["price"] = float(request.form.get("price"))
    data["ram_gb"] = float(request.form.get("ram_gb"))
    data["ssd_gb"] = float(request.form.get("ssd_gb"))
    data["hdd_gb"] = float(request.form.get("hdd_gb"))
    data["gpu_score"] = float(request.form.get("gpu_score"))
    data["cpu_score"] = float(request.form.get("cpu_score"))
    data["weight_kg"] = float(request.form.get("weight_kg"))

    data["target_ram_gb"] = float(request.form.get("target_ram_gb"))
    data["target_storage_gb"] = float(request.form.get("target_storage_gb"))
    data["target_gpu_score"] = float(request.form.get("target_gpu_score"))

    return data

def build_model_input(simple_data, features):
    row = {}

    total_storage_gb = (
        simple_data["ssd_gb"]
        + simple_data["hdd_gb"]
    )

    ram_gap_gb = max(simple_data["target_ram_gb"] - simple_data["ram_gb"], 0)
    storage_gap_gb = max(simple_data["target_storage_gb"] - total_storage_gb, 0)
    gpu_gap_score = max(simple_data["target_gpu_score"] - simple_data["gpu_score"], 0)

    performance_score = simple_data["cpu_score"] + simple_data["gpu_score"]

    if performance_score == 0:
        price_per_performance = simple_data["price"]
    else:
        price_per_performance = simple_data["price"] / performance_score

    for feature in features:

        if feature in simple_data:
            row[feature] = simple_data[feature]

        elif feature == "total_storage_gb":
            row[feature] = total_storage_gb

        elif feature == "has_ssd":
            row[feature] = 1 if simple_data["ssd_gb"] > 0 else 0

        elif feature == "has_hdd":
            row[feature] = 1 if simple_data["hdd_gb"] > 0 else 0

        elif feature == "is_integrated_gpu":
            row[feature] = 1 if simple_data["gpu_score"] <= 2 else 0

        elif feature == "ram_gap_gb":
            row[feature] = ram_gap_gb

        elif feature == "storage_gap_gb":
            row[feature] = storage_gap_gb

        elif feature == "gpu_gap_score":
            row[feature] = gpu_gap_score

        elif feature == "performance_score":
            row[feature] = performance_score

        elif feature == "price_per_performance":
            row[feature] = price_per_performance

        elif feature == "storage_capacity_score":
            row[feature] = total_storage_gb

        elif feature == "storage_speed_score":
            if simple_data["ssd_gb"] > 0:
                row[feature] = 1
            else:
                row[feature] = 0

        elif feature == "flash_gb":
            row[feature] = 0

        elif feature == "hybrid_gb":
            row[feature] = 0

        else:
            row[feature] = 0

    return pd.DataFrame([row])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
@app.route("/predict", methods=["GET", "POST"])
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


@app.route("/fuzzy", methods=["GET", "POST"])
def fuzzy():
    if request.method == "GET":
        return render_template("fuzzy.html")

    ram_gap = float(request.form.get("ram_gap_gb"))
    storage_gap = float(request.form.get("storage_gap_gb"))
    gpu_gap = float(request.form.get("gpu_gap_score"))

    score = 0

    if ram_gap >= 8:
        score += 35
    elif ram_gap >= 4:
        score += 20
    else:
        score += 5

    if storage_gap >= 512:
        score += 35
    elif storage_gap >= 256:
        score += 20
    else:
        score += 5

    if gpu_gap >= 7:
        score += 30
    elif gpu_gap >= 4:
        score += 15
    else:
        score += 5

    if score >= 70:
        decision = "High Upgrade Urgency"
        advice = "Upgrade is strongly recommended."
    elif score >= 40:
        decision = "Medium Upgrade Urgency"
        advice = "Upgrade may be useful soon."
    else:
        decision = "Low Upgrade Urgency"
        advice = "Upgrade is not urgent."

    result = {
        "score": score,
        "decision": decision,
        "advice": advice
    }

    return render_template("fuzzy_result.html", result=result)


@app.route("/anomaly", methods=["GET", "POST"])
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


@app.route("/rl", methods=["GET", "POST"])
def rl():
    if request.method == "GET":
        return render_template("rl.html", states=states)

    current_state = request.form.get("state")

    states_list = list(states)
    actions_list = list(actions)

    state_index = states_list.index(current_state)
    best_action_index = np.argmax(q_table[state_index])
    best_action = actions_list[best_action_index]

    result = {
        "current_state": current_state,
        "best_action": best_action
    }

    return render_template("rl_result.html", result=result)


@app.route("/admin")
def admin():
    model_status = {
        "best_classifier_model.pkl": "loaded",
        "best_regressor_model.pkl": "loaded",
        "fair_classifier.pkl": "loaded",
        "fair_regressor.pkl": "loaded",
        "clf_preprocessor.pkl": "loaded",
        "reg_preprocessor.pkl": "loaded",
        "fair_clf_preprocessor.pkl": "loaded",
        "fair_reg_preprocessor.pkl": "loaded",
        "isolation_forest_model.pkl": "loaded",
        "q_table.pkl": "loaded",
        "states.pkl": "loaded",
        "actions.pkl": "loaded"
    }

    feature_count = {
        "classification_features": len(clf_features),
        "regression_features": len(reg_features),
        "states": len(states),
        "actions": len(actions)
    }

    return render_template(
        "admin.html",
        model_status=model_status,
        feature_count=feature_count
    )


if __name__ == "__main__":
    app.run(debug=True)