def get_model_choice_message(model_choice):
    if model_choice == "fair":
        return {
            "model_choice": "fair",
            "title": "Fair Model Selected",
            "message": (
                "The fair model was selected. This version uses the bias-mitigated "
                "models trained in the fairness notebook."
            )
        }

    return {
        "model_choice": "original",
        "title": "Original Model Selected",
        "message": (
            "The original model was selected. This version uses the first selected "
            "classification and regression models before fairness mitigation."
        )
    }


def get_fairness_metrics_used():
    return {
        "classification_metrics": [
            "Disparate Impact Ratio",
            "Demographic Parity",
            "Equalized Odds"
        ],
        "regression_metrics": [
            "Mean Prediction Difference",
            "Residual Error Parity",
            "R² Performance Parity"
        ],
        "mitigation_strategy": [
            "Pre-processing: remove proxy features / rebalance groups",
            "In-processing: use model weighting when needed",
            "Post-processing: adjust thresholds when bias remains"
        ]
    }


def get_fairness_features_used():
    return {
        "classification_fairness_features": [
            "performance_score",
            "upgrade_pressure",
            "total_urgency",
            "budget_class",
            "user_profile"
        ],
        "regression_fairness_features": [
            "performance_score",
            "upgrade_pressure",
            "budget_class",
            "total_urgency",
            "type_name"
        ],
        "protected_group_classification": "budget_class",
        "protected_group_regression": "budget_class"
    }


def summarize_fairness_choice(model_choice):
    choice_message = get_model_choice_message(model_choice)
    metrics = get_fairness_metrics_used()
    features = get_fairness_features_used()

    return {
        "choice": choice_message,
        "metrics": metrics,
        "features": features
    }