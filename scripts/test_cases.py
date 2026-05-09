import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd

from model_loader import (
    best_classifier,
    best_regressor,
    clf_preprocessor,
    reg_preprocessor,
    clf_features,
    reg_features
)

from feature_engineering import build_model_input, calculate_backend_features
from services.decision_service import make_final_decision


test_cases = [
    {
        "case_name": "No upgrade needed",
        "user_profile": "Student",
        "budget_class": "Medium",
        "laptop_type": "Notebook",
        "brand": "Dell",
        "cpu_brand": "Intel",
        "gpu_brand": "Nvidia",
        "price": 900,
        "ram_gb": 16,
        "ssd_gb": 1024,
        "hdd_gb": 0,
        "cpu_score": 8,
        "gpu_score": 8,
        "weight_kg": 1.7,
        "target_ram_gb": 16,
        "target_storage_gb": 1024,
        "target_cpu_score": 8,
        "target_gpu_score": 8
    },
    {
        "case_name": "RAM problem",
        "user_profile": "Student",
        "budget_class": "Low",
        "laptop_type": "Notebook",
        "brand": "Lenovo",
        "cpu_brand": "Intel",
        "gpu_brand": "Integrated",
        "price": 400,
        "ram_gb": 4,
        "ssd_gb": 512,
        "hdd_gb": 0,
        "cpu_score": 6,
        "gpu_score": 3,
        "weight_kg": 1.8,
        "target_ram_gb": 16,
        "target_storage_gb": 512,
        "target_cpu_score": 6,
        "target_gpu_score": 3
    },
    {
        "case_name": "Storage problem",
        "user_profile": "Office Worker",
        "budget_class": "Medium",
        "laptop_type": "Notebook",
        "brand": "HP",
        "cpu_brand": "Intel",
        "gpu_brand": "Integrated",
        "price": 550,
        "ram_gb": 16,
        "ssd_gb": 128,
        "hdd_gb": 0,
        "cpu_score": 6,
        "gpu_score": 3,
        "weight_kg": 1.6,
        "target_ram_gb": 16,
        "target_storage_gb": 1024,
        "target_cpu_score": 6,
        "target_gpu_score": 3
    },
    {
        "case_name": "GPU problem",
        "user_profile": "Gamer",
        "budget_class": "High",
        "laptop_type": "Gaming",
        "brand": "ASUS",
        "cpu_brand": "AMD",
        "gpu_brand": "Integrated",
        "price": 850,
        "ram_gb": 16,
        "ssd_gb": 1024,
        "hdd_gb": 0,
        "cpu_score": 8,
        "gpu_score": 2,
        "weight_kg": 2.2,
        "target_ram_gb": 16,
        "target_storage_gb": 1024,
        "target_cpu_score": 8,
        "target_gpu_score": 9
    },
    {
        "case_name": "Bad laptop replacement case",
        "user_profile": "Gamer",
        "budget_class": "High",
        "laptop_type": "Notebook",
        "brand": "Old Laptop",
        "cpu_brand": "Intel",
        "gpu_brand": "Integrated",
        "price": 300,
        "ram_gb": 4,
        "ssd_gb": 128,
        "hdd_gb": 0,
        "cpu_score": 2,
        "gpu_score": 1,
        "weight_kg": 2.5,
        "target_ram_gb": 32,
        "target_storage_gb": 2048,
        "target_cpu_score": 9,
        "target_gpu_score": 9
    }
]


for case in test_cases:
    case_name = case["case_name"]

    clf_input = build_model_input(case, clf_features)
    reg_input = build_model_input(case, reg_features)

    clf_processed = clf_preprocessor.transform(clf_input)
    reg_processed = reg_preprocessor.transform(reg_input)

    upgrade_prediction = best_classifier.predict(clf_processed)[0]
    cost_prediction = best_regressor.predict(reg_processed)[0]

    calculated = calculate_backend_features(case)

    final_decision = make_final_decision(
        upgrade_prediction=upgrade_prediction,
        cost_prediction=cost_prediction,
        ram_gap_gb=calculated["ram_gap_gb"],
        storage_gap_gb=calculated["storage_gap_gb"],
        cpu_gap_score=calculated["cpu_gap_score"],
        gpu_gap_score=calculated["gpu_gap_score"],
        total_urgency=calculated["total_urgency"]
    )

    print("\n" + "=" * 70)
    print("CASE:", case_name)
    print("-" * 70)
    print("Classifier output:", upgrade_prediction)
    print("Regression cost:", round(float(cost_prediction), 2))
    print("RAM gap:", calculated["ram_gap_gb"])
    print("Storage gap:", calculated["storage_gap_gb"])
    print("CPU gap:", calculated["cpu_gap_score"])
    print("GPU gap:", calculated["gpu_gap_score"])
    print("Total urgency:", round(calculated["total_urgency"], 2))
    print("Final decision:", final_decision["final_action"])
    print("Decision explanation:", final_decision["explanation"])

    if hasattr(best_classifier, "predict_proba"):
        probs = best_classifier.predict_proba(clf_processed)[0]
        print("Class probabilities:")
        for cls, prob in zip(best_classifier.classes_, probs):
            print(f"  {cls}: {round(float(prob), 3)}")