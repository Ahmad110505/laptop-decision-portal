import pandas as pd

from model_loader import anomaly_model


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


def label_anomaly(score):
    if score > 0.1:
        return "Normal"
    elif score > 0:
        return "Slightly Unusual"
    elif score > -0.1:
        return "Needs Attention"
    else:
        return "High-Risk Mismatch"


def explain_anomaly(row):
    reasons = []

    if row["price_per_performance"] > 150:
        reasons.append("Poor price-to-performance ratio")

    if row["performance_score"] < 20:
        reasons.append("Low overall performance")

    if row["price"] > 1000 and row["gpu_score"] < 3:
        reasons.append("High price but weak GPU")

    if row["recommended_upgrade_cost_est"] > 1000:
        reasons.append("Very high upgrade cost")

    if row["ram_gb"] < 8:
        reasons.append("Low RAM compared to typical laptops")

    if row["cpu_score"] < 4:
        reasons.append("Low CPU performance")

    if not reasons:
        return "No strong anomaly reason detected"

    return ", ".join(reasons)


def build_anomaly_input(simple_data):
    price = float(simple_data.get("price", 0))
    cpu_score = float(simple_data.get("cpu_score", 0))
    gpu_score = float(simple_data.get("gpu_score", 0))
    ram_gb = float(simple_data.get("ram_gb", 0))
    upgrade_cost = float(simple_data.get("recommended_upgrade_cost_est", 0))

    ssd_gb = float(simple_data.get("ssd_gb", 0))
    hdd_gb = float(simple_data.get("hdd_gb", 0))
    total_storage_gb = ssd_gb + hdd_gb

    performance_score = (
        cpu_score * (1 + ram_gb)
        + gpu_score * 0.8
    )

    if performance_score == 0:
        price_per_performance = price
    else:
        price_per_performance = price / performance_score

    ram_gap_gb = float(simple_data.get("ram_gap_gb", 0))
    storage_gap_gb = float(simple_data.get("storage_gap_gb", 0))
    gpu_gap_score = float(simple_data.get("gpu_gap_score", 0))

    total_urgency = float(simple_data.get("total_urgency", 0))

    if total_urgency == 0:
        total_urgency = min(
            10,
            (ram_gap_gb / 8)
            + (storage_gap_gb / 256)
            + (gpu_gap_score / 10)
        )

    row = {
        "price": price,
        "price_per_performance": price_per_performance,
        "performance_score": performance_score,
        "cpu_score": cpu_score,
        "gpu_score": gpu_score,
        "ram_gb": ram_gb,
        "recommended_upgrade_cost_est": upgrade_cost,
        "total_urgency": total_urgency
    }

    return pd.DataFrame([row], columns=ANOMALY_FEATURES)


def run_anomaly_check(simple_data):
    input_df = build_anomaly_input(simple_data)

    score = anomaly_model.decision_function(input_df)[0]
    flag = anomaly_model.predict(input_df)[0]

    label = label_anomaly(score)
    explanation = explain_anomaly(input_df.iloc[0])

    return {
        "score": round(float(score), 4),
        "flag": int(flag),
        "label": label,
        "message": explanation
    }