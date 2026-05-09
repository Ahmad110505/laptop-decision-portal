import numpy as np
import pandas as pd
from flask import request


def read_inputs(features):
    data = {}

    for feature in features:
        value = request.form.get(feature)

        try:
            data[feature] = float(value)
        except:
            data[feature] = value

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
    data["cpu_score"] = float(request.form.get("cpu_score"))
    data["gpu_score"] = float(request.form.get("gpu_score"))
    data["weight_kg"] = float(request.form.get("weight_kg"))

    data["target_ram_gb"] = float(request.form.get("target_ram_gb"))
    data["target_storage_gb"] = float(request.form.get("target_storage_gb"))
    data["target_cpu_score"] = float(request.form.get("target_cpu_score"))
    data["target_gpu_score"] = float(request.form.get("target_gpu_score"))

    return data


import numpy as np


def calculate_backend_features(simple_data):
    price = float(simple_data.get("price", 0))

    ram_gb = float(simple_data.get("ram_gb", 0))
    ssd_gb = float(simple_data.get("ssd_gb", 0))
    hdd_gb = float(simple_data.get("hdd_gb", 0))
    flash_gb = float(simple_data.get("flash_gb", 0))
    hybrid_gb = float(simple_data.get("hybrid_gb", 0))

    cpu_score = float(simple_data.get("cpu_score", 0))
    gpu_score = float(simple_data.get("gpu_score", 0))

    target_ram_gb = float(simple_data.get("target_ram_gb", ram_gb))
    target_storage_gb = float(simple_data.get("target_storage_gb", ssd_gb + hdd_gb + flash_gb + hybrid_gb))
    target_gpu_score = float(simple_data.get("target_gpu_score", gpu_score))

    total_storage_gb = ssd_gb + hdd_gb + flash_gb + hybrid_gb

    ram_gap_gb = max(target_ram_gb - ram_gb, 0)
    storage_gap_gb = max(target_storage_gb - total_storage_gb, 0)
    gpu_gap_score = max(target_gpu_score - gpu_score, 0)

    performance_score = (
        cpu_score * (1 + np.log1p(ram_gb)) +
        gpu_score * 0.8
    )

    ram_gap_ratio = ram_gap_gb / (target_ram_gb + 1)
    storage_gap_ratio = storage_gap_gb / (target_storage_gb + 1)
    gpu_gap_ratio = gpu_gap_score / (target_gpu_score + 1)

    upgrade_pressure = (
        0.35 * ram_gap_ratio +
        0.25 * storage_gap_ratio +
        0.40 * gpu_gap_ratio
    )

    # Use simple urgency approximation in Flask.
    # The notebook used ram_urgency_score, storage_urgency_score, gpu_urgency_score.
    # If those are not collected from the user, we derive them from the gap ratios.
    ram_urgency_score = min(10, ram_gap_ratio * 10)
    storage_urgency_score = min(10, storage_gap_ratio * 10)
    gpu_urgency_score = min(10, gpu_gap_ratio * 10)

    total_urgency = (
        ram_urgency_score +
        storage_urgency_score +
        gpu_urgency_score
    ) / 3

    price_per_performance = price / (performance_score + 1)

    storage_capacity_score = (
        ssd_gb +
        hdd_gb +
        flash_gb +
        hybrid_gb
    )

    storage_speed_score = (
        1.00 * ssd_gb +
        0.45 * hdd_gb +
        0.75 * flash_gb +
        0.65 * hybrid_gb
    )

    has_ssd = 1 if ssd_gb > 0 else 0
    has_hdd = 1 if hdd_gb > 0 else 0

    gpu_brand = str(simple_data.get("gpu_brand", "")).lower()
    is_integrated_gpu = 1 if gpu_brand == "integrated" or gpu_score <= 2 else 0

    # This is only for your UI decision layer.
    # The trained regression model does not use CPU gap.
    target_cpu_score = float(simple_data.get("target_cpu_score", cpu_score))
    cpu_gap_score = max(target_cpu_score - cpu_score, 0)

    return {
        "total_storage_gb": total_storage_gb,

        "has_ssd": has_ssd,
        "has_hdd": has_hdd,
        "is_integrated_gpu": is_integrated_gpu,

        "ram_gap_gb": ram_gap_gb,
        "storage_gap_gb": storage_gap_gb,
        "gpu_gap_score": gpu_gap_score,
        "cpu_gap_score": cpu_gap_score,

        "ram_gap_ratio": ram_gap_ratio,
        "storage_gap_ratio": storage_gap_ratio,
        "gpu_gap_ratio": gpu_gap_ratio,

        "performance_score": performance_score,
        "upgrade_pressure": upgrade_pressure,
        "total_urgency": total_urgency,
        "price_per_performance": price_per_performance,
        "storage_capacity_score": storage_capacity_score,
        "storage_speed_score": storage_speed_score,

        "ram_urgency_score": ram_urgency_score,
        "storage_urgency_score": storage_urgency_score,
        "gpu_urgency_score": gpu_urgency_score,

        "flash_gb": flash_gb,
        "hybrid_gb": hybrid_gb
    }


def is_numeric_feature(feature):
    numeric_words = [
        "price",
        "cost",
        "score",
        "gb",
        "ram",
        "storage",
        "weight",
        "gap",
        "urgency",
        "performance",
        "pressure",
        "capacity",
        "speed",
        "target",
        "has_",
        "is_"
    ]

    feature_lower = str(feature).lower()

    for word in numeric_words:
        if word in feature_lower:
            return True

    return False


def build_model_input(simple_data, features):
    row = {}
    calculated_features = calculate_backend_features(simple_data)

    for feature in features:
        if feature in calculated_features:
            row[feature] = calculated_features[feature]

        elif feature in simple_data:
            row[feature] = simple_data[feature]

        elif is_numeric_feature(feature):
            row[feature] = 0

        else:
            row[feature] = "Unknown"

    return pd.DataFrame([row], columns=features)