import pandas as pd

from model_loader import anomaly_model
from feature_engineering import calculate_backend_features


ANOMALY_FEATURES = [
    "price",
    "price_per_performance",
    "performance_score",
    "cpu_score",
    "gpu_score",
    "ram_gb",
    "recommended_upgrade_cost_est",
    "total_urgency"
]


def build_anomaly_input(simple_data):
    calculated = calculate_backend_features(simple_data)

    row = {
        "price": float(simple_data.get("price", 0)),
        "price_per_performance": float(calculated.get("price_per_performance", 0)),
        "performance_score": float(calculated.get("performance_score", 0)),
        "cpu_score": float(simple_data.get("cpu_score", 0)),
        "gpu_score": float(simple_data.get("gpu_score", 0)),
        "ram_gb": float(simple_data.get("ram_gb", 0)),
        "recommended_upgrade_cost_est": float(
            simple_data.get("recommended_upgrade_cost_est", 0)
        ),
        "total_urgency": float(calculated.get("total_urgency", simple_data.get("total_urgency", 0)))
    }

    return pd.DataFrame([row], columns=ANOMALY_FEATURES)


def explain_anomaly(row):
    reasons = []

    price = float(row["price"])
    price_per_performance = float(row["price_per_performance"])
    performance_score = float(row["performance_score"])
    cpu_score = float(row["cpu_score"])
    gpu_score = float(row["gpu_score"])
    ram_gb = float(row["ram_gb"])
    upgrade_cost = float(row["recommended_upgrade_cost_est"])
    total_urgency = float(row["total_urgency"])

    if price_per_performance > 120:
        reasons.append("poor price-to-performance ratio")

    if performance_score < 15:
        reasons.append("low overall performance")

    if price > 1000 and gpu_score < 3:
        reasons.append("high price but weak GPU")

    if upgrade_cost > 1000:
        reasons.append("very high estimated upgrade cost")

    if ram_gb < 8:
        reasons.append("low RAM compared with typical laptops")

    if cpu_score < 4:
        reasons.append("low CPU performance")

    if gpu_score < 2:
        reasons.append("low GPU performance")

    if total_urgency >= 8:
        reasons.append("very high upgrade urgency")

    if not reasons:
        return "The laptop is only slightly outside the learned normal pattern, but no strong technical warning was detected."

    return "This laptop was flagged because it has " + ", ".join(reasons) + "."


def label_anomaly(score, flag, explanation):
    explanation_lower = explanation.lower()

    has_strong_reason = any(
        reason in explanation_lower
        for reason in [
            "poor price-to-performance",
            "low overall performance",
            "high price but weak gpu",
            "very high estimated upgrade cost",
            "very high upgrade urgency"
        ]
    )

    if flag == 1:
        return "Normal"

    if score <= -0.10 and has_strong_reason:
        return "High-Risk Mismatch"

    if score <= -0.04 and has_strong_reason:
        return "Needs Attention"

    return "Slightly Unusual"


def run_anomaly_check(simple_data):
    input_df = build_anomaly_input(simple_data)

    score = float(anomaly_model.decision_function(input_df)[0])
    flag = int(anomaly_model.predict(input_df)[0])

    explanation = explain_anomaly(input_df.iloc[0])
    label = label_anomaly(score, flag, explanation)

    return {
        "score": round(score, 4),
        "flag": flag,
        "label": label,
        "message": explanation
    }