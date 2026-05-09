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
    data["gpu_score"] = float(request.form.get("gpu_score"))
    data["cpu_score"] = float(request.form.get("cpu_score"))
    data["weight_kg"] = float(request.form.get("weight_kg"))

    data["target_ram_gb"] = float(request.form.get("target_ram_gb"))
    data["target_storage_gb"] = float(request.form.get("target_storage_gb"))
    data["target_gpu_score"] = float(request.form.get("target_gpu_score"))

    return data


def build_model_input(simple_data, features):
    row = {}

    total_storage_gb = simple_data["ssd_gb"] + simple_data["hdd_gb"]

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
            row[feature] = 1 if simple_data["ssd_gb"] > 0 else 0

        elif feature == "flash_gb":
            row[feature] = 0

        elif feature == "hybrid_gb":
            row[feature] = 0

        else:
            row[feature] = 0

    return pd.DataFrame([row])