from flask import Blueprint, render_template, request


fuzzy_bp = Blueprint("fuzzy", __name__)


@fuzzy_bp.route("/fuzzy", methods=["GET", "POST"])
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