import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np

from model_loader import (
    best_classifier,
    fair_classifier,
    best_regressor,
    fair_regressor
)


def print_classifier_outputs(model, model_name):
    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    if hasattr(model, "classes_"):
        print("Possible classification outputs:")
        for cls in model.classes_:
            print("-", cls)
    else:
        print("This classifier does not expose classes_.")


def print_regressor_info(model, model_name):
    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)
    print("Regressor type:", type(model).__name__)

    if hasattr(model, "n_estimators"):
        print("Number of estimators:", model.n_estimators)

    if hasattr(model, "feature_importances_"):
        print("Feature importances available: Yes")
    else:
        print("Feature importances available: No")


print_classifier_outputs(best_classifier, "Original Classifier")
print_classifier_outputs(fair_classifier, "Fair Classifier")

print_regressor_info(best_regressor, "Original Regressor")
print_regressor_info(fair_regressor, "Fair Regressor")