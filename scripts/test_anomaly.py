import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from services.anomaly_service import run_anomaly_check, build_anomaly_input


test_cases = [
    {
        "case_name": "Normal balanced laptop",
        "expected": "Should usually be Normal or Slightly Unusual",
        "user_profile": "Student",
        "budget_class": "Medium",
        "laptop_type": "Notebook",
        "brand": "Dell",
        "cpu_brand": "Intel",
        "gpu_brand": "Integrated",
        "price": 650,
        "ram_gb": 8,
        "ssd_gb": 512,
        "hdd_gb": 0,
        "cpu_score": 6,
        "gpu_score": 3,
        "weight_kg": 1.7,
        "target_ram_gb": 16,
        "target_storage_gb": 512,
        "target_cpu_score": 6,
        "target_gpu_score": 3,
        "ram_gap_gb": 8,
        "storage_gap_gb": 0,
        "gpu_gap_score": 0,
        "total_urgency": 2,
        "recommended_upgrade_cost_est": 90
    },
    {
        "case_name": "Expensive laptop with weak GPU",
        "expected": "Should usually be Slightly Unusual or Needs Attention",
        "user_profile": "Gamer",
        "budget_class": "High",
        "laptop_type": "Gaming",
        "brand": "ASUS",
        "cpu_brand": "Intel",
        "gpu_brand": "Integrated",
        "price": 1400,
        "ram_gb": 16,
        "ssd_gb": 512,
        "hdd_gb": 0,
        "cpu_score": 7,
        "gpu_score": 2,
        "weight_kg": 2.3,
        "target_ram_gb": 16,
        "target_storage_gb": 1024,
        "target_cpu_score": 8,
        "target_gpu_score": 9,
        "ram_gap_gb": 0,
        "storage_gap_gb": 512,
        "gpu_gap_score": 7,
        "total_urgency": 7,
        "recommended_upgrade_cost_est": 300
    },
    {
        "case_name": "Very high price with weak hardware",
        "expected": "Should usually be Needs Attention or High-Risk Mismatch",
        "user_profile": "Office Worker",
        "budget_class": "High",
        "laptop_type": "Notebook",
        "brand": "Unknown",
        "cpu_brand": "Intel",
        "gpu_brand": "Integrated",
        "price": 1800,
        "ram_gb": 4,
        "ssd_gb": 128,
        "hdd_gb": 0,
        "cpu_score": 2,
        "gpu_score": 1,
        "weight_kg": 2.6,
        "target_ram_gb": 16,
        "target_storage_gb": 1024,
        "target_cpu_score": 7,
        "target_gpu_score": 5,
        "ram_gap_gb": 12,
        "storage_gap_gb": 896,
        "gpu_gap_score": 4,
        "total_urgency": 9,
        "recommended_upgrade_cost_est": 700
    },
    {
        "case_name": "Bad laptop replacement case",
        "expected": "Should usually be Needs Attention or High-Risk Mismatch",
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
        "target_gpu_score": 9,
        "ram_gap_gb": 28,
        "storage_gap_gb": 1920,
        "gpu_gap_score": 8,
        "total_urgency": 10,
        "recommended_upgrade_cost_est": 350
    },
    {
        "case_name": "Suspicious upgrade cost",
        "expected": "Should usually be Needs Attention or High-Risk Mismatch",
        "user_profile": "Student",
        "budget_class": "Low",
        "laptop_type": "Notebook",
        "brand": "Lenovo",
        "cpu_brand": "Intel",
        "gpu_brand": "Integrated",
        "price": 350,
        "ram_gb": 8,
        "ssd_gb": 256,
        "hdd_gb": 0,
        "cpu_score": 5,
        "gpu_score": 2,
        "weight_kg": 1.8,
        "target_ram_gb": 16,
        "target_storage_gb": 512,
        "target_cpu_score": 6,
        "target_gpu_score": 3,
        "ram_gap_gb": 8,
        "storage_gap_gb": 256,
        "gpu_gap_score": 1,
        "total_urgency": 4,
        "recommended_upgrade_cost_est": 1500
    }
]


for case in test_cases:
    print("\n" + "=" * 80)
    print("CASE:", case["case_name"])
    print("-" * 80)

    anomaly_input = build_anomaly_input(case)
    result = run_anomaly_check(case)

    print("Expected:", case["expected"])
    print("\nInput sent to anomaly model:")
    print(anomaly_input.to_string(index=False))

    print("\nAnomaly result:")
    print("Score:", result.get("score"))
    print("Flag:", result.get("flag"))
    print("Label:", result.get("label"))
    print("Message:", result.get("message"))