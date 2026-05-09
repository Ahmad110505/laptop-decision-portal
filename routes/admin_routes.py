from flask import Blueprint, render_template

from model_loader import clf_features, reg_features, states, actions


admin_bp = Blueprint("admin", __name__)


@admin_bp.route("/admin")
def admin():
    model_status = {
        "best_classifier_model.pkl": "loaded",
        "best_regressor_model.pkl": "loaded",
        "fair_classifier.pkl": "loaded",
        "fair_regressor.pkl": "loaded",
        "clf_preprocessor.pkl": "loaded",
        "reg_preprocessor.pkl": "loaded",
        "fair_clf_preprocessor.pkl": "loaded",
        "fair_reg_preprocessor.pkl": "loaded",
        "isolation_forest_model.pkl": "loaded",
        "q_table.pkl": "loaded",
        "states.pkl": "loaded",
        "actions.pkl": "loaded"
    }

    feature_count = {
        "classification_features": len(clf_features),
        "regression_features": len(reg_features),
        "states": len(states),
        "actions": len(actions)
    }

    return render_template(
        "admin.html",
        model_status=model_status,
        feature_count=feature_count
    )