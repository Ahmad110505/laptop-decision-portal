from flask import Blueprint, render_template, request, redirect, url_for, session

from model_loader import (
    best_classifier,
    best_regressor,
    fair_classifier,
    fair_regressor,
    clf_preprocessor,
    reg_preprocessor,
    fair_clf_preprocessor,
    fair_reg_preprocessor,
    clf_features,
    reg_features
)

from feature_engineering import build_model_input, calculate_backend_features

from services.fuzzy_service import run_fuzzy_upgrade_evaluator
from services.anomaly_service import run_anomaly_check
from services.fairness_service import summarize_fairness_choice
from services.rl_service import get_guidance_from_data, get_route_for_guidance
from services.decision_service import make_final_decision


prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.route("/predict")
def predict_start():
    session.clear()
    return redirect(url_for("prediction.predict_profile"))


@prediction_bp.route("/predict/profile", methods=["GET", "POST"])
def predict_profile():
    if request.method == "GET":
        rl_guidance = get_guidance_from_data(dict(session))

        return render_template(
            "predict_profile.html",
            rl_guidance=rl_guidance
        )

    session["model_choice"] = request.form.get("model_choice")
    session["user_profile"] = request.form.get("user_profile")
    session["budget_class"] = request.form.get("budget_class")

    rl_result = get_guidance_from_data(dict(session))
    next_route = get_route_for_guidance(rl_result)

    return redirect(url_for(next_route))


@prediction_bp.route("/predict/laptop", methods=["GET", "POST"])
def predict_laptop():
    if request.method == "GET":
        rl_guidance = get_guidance_from_data(dict(session))

        return render_template(
            "predict_laptop.html",
            rl_guidance=rl_guidance
        )

    session["laptop_type"] = request.form.get("laptop_type")
    session["brand"] = request.form.get("brand")
    session["cpu_brand"] = request.form.get("cpu_brand")
    session["gpu_brand"] = request.form.get("gpu_brand")

    session["price"] = float(request.form.get("price"))
    session["ram_gb"] = float(request.form.get("ram_gb"))
    session["ssd_gb"] = float(request.form.get("ssd_gb"))
    session["hdd_gb"] = float(request.form.get("hdd_gb"))
    session["cpu_score"] = float(request.form.get("cpu_score"))
    session["gpu_score"] = float(request.form.get("gpu_score"))
    session["weight_kg"] = float(request.form.get("weight_kg"))

    rl_result = get_guidance_from_data(dict(session))
    next_route = get_route_for_guidance(rl_result)

    return redirect(url_for(next_route))


@prediction_bp.route("/predict/needs", methods=["GET", "POST"])
def predict_needs():
    if request.method == "GET":
        rl_guidance = get_guidance_from_data(dict(session))

        return render_template(
            "predict_needs.html",
            rl_guidance=rl_guidance
        )

    session["target_ram_gb"] = float(request.form.get("target_ram_gb"))
    session["target_storage_gb"] = float(request.form.get("target_storage_gb"))
    session["target_cpu_score"] = float(request.form.get("target_cpu_score"))
    session["target_gpu_score"] = float(request.form.get("target_gpu_score"))

    rl_result = get_guidance_from_data(dict(session))
    next_route = get_route_for_guidance(rl_result)

    return redirect(url_for(next_route))


@prediction_bp.route("/predict/result")
def predict_result():
    simple_data = dict(session)

    model_choice = simple_data.get("model_choice")

    # Build model-ready inputs for the classifier and regressor
    clf_input = build_model_input(simple_data, clf_features)
    reg_input = build_model_input(simple_data, reg_features)

    # Run selected model version
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

    # Calculate the real backend features from the user inputs
    calculated_features = calculate_backend_features(simple_data)

    ram_gap_gb = calculated_features["ram_gap_gb"]
    storage_gap_gb = calculated_features["storage_gap_gb"]
    cpu_gap_score = calculated_features["cpu_gap_score"]
    gpu_gap_score = calculated_features["gpu_gap_score"]
    total_urgency = calculated_features["total_urgency"]

    # Add calculated values back into simple_data
    # so fuzzy, anomaly, RL, and decision services all use the same values
    simple_data["ram_gap_gb"] = float(ram_gap_gb)
    simple_data["storage_gap_gb"] = float(storage_gap_gb)
    simple_data["cpu_gap_score"] = float(cpu_gap_score)
    simple_data["gpu_gap_score"] = float(gpu_gap_score)
    simple_data["total_urgency"] = float(total_urgency)
    simple_data["recommended_upgrade_cost_est"] = float(cost_prediction)
    simple_data["upgrade_prediction"] = str(upgrade_prediction)

    fuzzy_result = run_fuzzy_upgrade_evaluator(
        total_urgency_value=float(total_urgency),
        ram_gap_value=float(ram_gap_gb),
        gpu_gap_value=float(gpu_gap_score),
        upgrade_cost_value=float(cost_prediction)
    )

    final_decision = make_final_decision(
        upgrade_prediction=upgrade_prediction,
        cost_prediction=cost_prediction,
        ram_gap_gb=ram_gap_gb,
        storage_gap_gb=storage_gap_gb,
        cpu_gap_score=cpu_gap_score,
        gpu_gap_score=gpu_gap_score,
        total_urgency=total_urgency
    )

    anomaly_result = run_anomaly_check(simple_data)

    # RL now reads the completed session/result data and recommends the current next action
    rl_result = get_guidance_from_data(simple_data)

    fairness_result = summarize_fairness_choice(model_choice)

    result = {
        "model_choice": model_choice,
        "user_profile": simple_data.get("user_profile"),
        "budget_class": simple_data.get("budget_class"),

        "upgrade_prediction": upgrade_prediction,
        "cost_prediction": round(float(cost_prediction), 2),
        "final_decision": final_decision,

        "ram_gap_gb": round(float(ram_gap_gb), 2),
        "storage_gap_gb": round(float(storage_gap_gb), 2),
        "cpu_gap_score": round(float(cpu_gap_score), 2),
        "gpu_gap_score": round(float(gpu_gap_score), 2),
        "total_urgency": round(float(total_urgency), 2),

        "fuzzy": fuzzy_result,
        "anomaly": anomaly_result,
        "rl": rl_result,
        "fairness": fairness_result
    }

    return render_template("predict_result.html", result=result)