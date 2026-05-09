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
        "total_urgency": float(calculated.get("total_urgency", 0))
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
        return "No clear technical anomaly was detected."

    return "This laptop was flagged because it has " + ", ".join(reasons) + "."


def label_anomaly(score, flag, explanation):
    explanation_lower = explanation.lower()

    serious_reasons = [
        "poor price-to-performance",
        "low overall performance",
        "high price but weak gpu",
        "very high estimated upgrade cost",
        "very high upgrade urgency"
    ]

    basic_reasons = [
        "low ram",
        "low cpu",
        "low gpu"
    ]

    serious_count = 0
    basic_count = 0

    for reason in serious_reasons:
        if reason in explanation_lower:
            serious_count += 1

    for reason in basic_reasons:
        if reason in explanation_lower:
            basic_count += 1

    total_reasons = serious_count + basic_count

    # If the model flagged it but we cannot explain why,
    # do not scare the user.
    if total_reasons == 0:
        return "Normal"

    # Only use High-Risk when there are multiple serious signals.
    if serious_count >= 3:
        return "High-Risk Mismatch"

    if serious_count >= 2 and basic_count >= 2:
        return "High-Risk Mismatch"

    if score <= -0.12 and serious_count >= 2:
        return "High-Risk Mismatch"

    # Otherwise keep it moderate.
    return "Needs Attention"


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