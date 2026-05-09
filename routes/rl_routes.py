from flask import Blueprint, render_template, request
import numpy as np

from model_loader import q_table, states, actions


rl_bp = Blueprint("rl", __name__)


@rl_bp.route("/rl", methods=["GET", "POST"])
def rl():
    if request.method == "GET":
        return render_template("rl.html", states=states)

    current_state = request.form.get("state")

    states_list = list(states)
    actions_list = list(actions)

    state_index = states_list.index(current_state)
    best_action_index = np.argmax(q_table[state_index])
    best_action = actions_list[best_action_index]

    result = {
        "current_state": current_state,
        "best_action": best_action
    }

    return render_template("rl_result.html", result=result)