def count_major_gaps(ram_gap_gb, storage_gap_gb, cpu_gap_score, gpu_gap_score):
    major_gaps = 0

    if ram_gap_gb >= 16:
        major_gaps += 1

    if storage_gap_gb >= 512:
        major_gaps += 1

    if cpu_gap_score >= 4:
        major_gaps += 1

    if gpu_gap_score >= 4:
        major_gaps += 1

    return major_gaps


def make_final_decision(
    upgrade_prediction,
    cost_prediction,
    ram_gap_gb,
    storage_gap_gb,
    cpu_gap_score,
    gpu_gap_score,
    total_urgency
):
    major_gaps = count_major_gaps(
        ram_gap_gb=ram_gap_gb,
        storage_gap_gb=storage_gap_gb,
        cpu_gap_score=cpu_gap_score,
        gpu_gap_score=gpu_gap_score
    )

    if major_gaps >= 3 or total_urgency >= 8:
        return {
            "final_action": "Replace Laptop",
            "decision_type": "replacement",
            "model_prediction": str(upgrade_prediction),
            "component_upgrade_cost": round(float(cost_prediction), 2),
            "major_gaps": major_gaps,
            "explanation": (
                "The laptop has serious gaps in multiple areas, so upgrading only one component "
                "is unlikely to solve the problem. Replacing the laptop is more practical."
            )
        }

    if major_gaps == 2:
        return {
            "final_action": "Major Upgrade Needed",
            "decision_type": "major_upgrade",
            "model_prediction": str(upgrade_prediction),
            "component_upgrade_cost": round(float(cost_prediction), 2),
            "major_gaps": major_gaps,
            "explanation": (
                "The laptop has more than one major limitation. A single upgrade may help, "
                "but a larger upgrade plan should be considered."
            )
        }

    return {
        "final_action": str(upgrade_prediction),
        "decision_type": "single_upgrade",
        "model_prediction": str(upgrade_prediction),
        "component_upgrade_cost": round(float(cost_prediction), 2),
        "major_gaps": major_gaps,
        "explanation": (
            "The model recommendation is suitable because the laptop does not show enough "
            "major gaps to require replacement."
        )
    }