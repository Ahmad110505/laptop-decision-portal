import numpy as np

from model_loader import q_table, states, actions


def make_action_readable(action):
    labels = {
        "ask_budget": "Ask about the user's budget",
        "ask_profile": "Ask about the user's profile and usage",
        "ask_usage": "Ask about the user's usage needs",
        "ask_specs": "Ask for current laptop specifications",
        "ask_laptop_specs": "Ask for current laptop specifications",
        "ask_target_needs": "Ask for target upgrade needs",
        "run_models": "Run the AI decision modules",
        "recommend": "Show the final laptop recommendation",
        "show_recommendation": "Show the final laptop recommendation",
        "conversation_finished": "Conversation finished"
    }

    return labels.get(action, action)


def explain_action(state, action):
    explanations = {
        "start": "The system has not collected enough information yet.",
        "budget_known": "The system knows the budget, so it should continue collecting user context.",
        "profile_known": "The system now knows the user profile and budget, so it should collect laptop specifications.",
        "laptop_specs_known": "The system has the current laptop specifications, so it should ask for target needs.",
        "needs_known": "The system has enough information to run the AI decision modules.",
        "prediction_ready": "The AI modules have completed their checks, so the system should show the final recommendation."
    }

    return explanations.get(
        state,
        "The RL policy selected this action based on the current dialogue state."
    )


def choose_state_from_session_data(simple_data):
    has_budget = simple_data.get("budget_class") not in [None, ""]
    has_profile = simple_data.get("user_profile") not in [None, ""]

    has_specs = (
        simple_data.get("ram_gb") not in [None, ""]
        and simple_data.get("ssd_gb") not in [None, ""]
        and simple_data.get("cpu_score") not in [None, ""]
        and simple_data.get("gpu_score") not in [None, ""]
    )

    has_needs = (
        simple_data.get("target_ram_gb") not in [None, ""]
        and simple_data.get("target_storage_gb") not in [None, ""]
        and simple_data.get("target_cpu_score") not in [None, ""]
        and simple_data.get("target_gpu_score") not in [None, ""]
    )

    has_outputs = (
        simple_data.get("upgrade_prediction") not in [None, ""]
        and simple_data.get("recommended_upgrade_cost_est") not in [None, ""]
    )

    if has_budget and has_profile and has_specs and has_needs and has_outputs:
        return "prediction_ready"

    if has_budget and has_profile and has_specs and has_needs:
        return "needs_known"

    if has_budget and has_profile and has_specs:
        return "laptop_specs_known"

    if has_budget and has_profile:
        return "profile_known"

    if has_budget:
        return "budget_known"

    return "start"


def get_fallback_action(current_state):
    """
    This protects the Flask app from getting stuck if the saved Q-table
    does not contain the newer state names.
    """

    fallback_policy = {
        "start": "ask_budget",
        "budget_known": "ask_profile",
        "profile_known": "ask_laptop_specs",
        "laptop_specs_known": "ask_target_needs",
        "needs_known": "run_models",
        "prediction_ready": "show_recommendation"
    }

    return fallback_policy.get(current_state, "ask_budget")


def get_best_action(current_state):
    states_list = list(states)
    actions_list = list(actions)

    # If the saved Q-table knows this state, use it.
    if current_state in states_list:
        state_index = states_list.index(current_state)
        q_values = q_table[state_index]

        best_action_index = np.argmax(q_values)
        best_action = actions_list[best_action_index]

        # If Q-table gives a useless first-page loop, fallback to logical flow.
        if current_state == "profile_known" and best_action in ["ask_budget", "ask_profile"]:
            best_action = "ask_laptop_specs"

        if current_state == "laptop_specs_known" and best_action in ["ask_budget", "ask_profile", "ask_laptop_specs"]:
            best_action = "ask_target_needs"

        if current_state == "needs_known" and best_action not in ["run_models", "recommend", "show_recommendation"]:
            best_action = "run_models"

        q_values_dict = {
            actions_list[i]: round(float(q_values[i]), 3)
            for i in range(len(actions_list))
        }

    # If the saved Q-table does not know this state, use the safe fallback policy.
    else:
        best_action = get_fallback_action(current_state)
        q_values_dict = {}

    return {
        "current_state": current_state,
        "best_action": best_action,
        "display_action": make_action_readable(best_action),
        "explanation": explain_action(current_state, best_action),
        "q_values": q_values_dict
    }


def get_guidance_from_data(simple_data):
    current_state = choose_state_from_session_data(simple_data)
    return get_best_action(current_state)


def get_route_for_guidance(rl_result):
    current_state = rl_result["current_state"]
    action = rl_result["best_action"]

    # State-aware routing is safer than raw action routing.
    if current_state == "profile_known":
        return "prediction.predict_laptop"

    if current_state == "laptop_specs_known":
        return "prediction.predict_needs"

    if current_state == "needs_known":
        return "prediction.predict_result"

    if current_state == "prediction_ready":
        return "prediction.predict_result"

    route_map = {
        "ask_budget": "prediction.predict_profile",
        "ask_profile": "prediction.predict_profile",
        "ask_usage": "prediction.predict_profile",
        "ask_specs": "prediction.predict_laptop",
        "ask_laptop_specs": "prediction.predict_laptop",
        "ask_target_needs": "prediction.predict_needs",
        "run_models": "prediction.predict_result",
        "show_recommendation": "prediction.predict_result",
        "recommend": "prediction.predict_result",
        "conversation_finished": "prediction.predict_result"
    }

    return route_map.get(action, "prediction.predict_profile")