
import os
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API"),
    base_url="https://api.groq.com/openai/v1"
)


total_urgency = ctrl.Antecedent(np.arange(0, 11, 1), "total_urgency")
ram_gap_gb = ctrl.Antecedent(np.arange(0, 33, 1), "ram_gap_gb")
gpu_gap_score = ctrl.Antecedent(np.arange(0, 101, 1), "gpu_gap_score")
upgrade_cost = ctrl.Antecedent(np.arange(0, 2001, 50), "upgrade_cost")

upgrade_decision = ctrl.Consequent(np.arange(0, 101, 1), "upgrade_decision")


total_urgency["low"] = fuzz.trimf(total_urgency.universe, [0, 0, 4])
total_urgency["medium"] = fuzz.trimf(total_urgency.universe, [2, 5, 8])
total_urgency["high"] = fuzz.trimf(total_urgency.universe, [6, 10, 10])

ram_gap_gb["low"] = fuzz.trimf(ram_gap_gb.universe, [0, 0, 8])
ram_gap_gb["medium"] = fuzz.trimf(ram_gap_gb.universe, [4, 12, 20])
ram_gap_gb["high"] = fuzz.trimf(ram_gap_gb.universe, [16, 32, 32])

gpu_gap_score["low"] = fuzz.trimf(gpu_gap_score.universe, [0, 0, 35])
gpu_gap_score["medium"] = fuzz.trimf(gpu_gap_score.universe, [25, 50, 75])
gpu_gap_score["high"] = fuzz.trimf(gpu_gap_score.universe, [65, 100, 100])

upgrade_cost["low"] = fuzz.trimf(upgrade_cost.universe, [0, 0, 600])
upgrade_cost["medium"] = fuzz.trimf(upgrade_cost.universe, [400, 900, 1400])
upgrade_cost["high"] = fuzz.trimf(upgrade_cost.universe, [1200, 2000, 2000])

upgrade_decision["low"] = fuzz.trimf(upgrade_decision.universe, [0, 0, 30])
upgrade_decision["medium"] = fuzz.trimf(upgrade_decision.universe, [20, 45, 65])
upgrade_decision["high"] = fuzz.trimf(upgrade_decision.universe, [55, 75, 90])
upgrade_decision["critical"] = fuzz.trimf(upgrade_decision.universe, [80, 100, 100])


rule1 = ctrl.Rule(
    total_urgency["low"] & ram_gap_gb["low"] & gpu_gap_score["low"],
    upgrade_decision["low"]
)

rule2 = ctrl.Rule(
    total_urgency["medium"] | ram_gap_gb["medium"],
    upgrade_decision["medium"]
)

rule3 = ctrl.Rule(
    total_urgency["high"] & ram_gap_gb["medium"],
    upgrade_decision["high"]
)

rule4 = ctrl.Rule(
    total_urgency["high"] & gpu_gap_score["high"],
    upgrade_decision["critical"]
)

rule5 = ctrl.Rule(
    ram_gap_gb["high"] & total_urgency["high"],
    upgrade_decision["critical"]
)

rule6 = ctrl.Rule(
    gpu_gap_score["high"] & total_urgency["medium"],
    upgrade_decision["high"]
)

rule7 = ctrl.Rule(
    upgrade_cost["high"] & total_urgency["low"],
    upgrade_decision["medium"]
)

rule8 = ctrl.Rule(
    upgrade_cost["high"] & total_urgency["high"],
    upgrade_decision["high"]
)

rule9 = ctrl.Rule(
    upgrade_cost["low"] & total_urgency["high"],
    upgrade_decision["critical"]
)

rule10 = ctrl.Rule(
    ram_gap_gb["medium"] & gpu_gap_score["medium"],
    upgrade_decision["medium"]
)


upgrade_ctrl = ctrl.ControlSystem([
    rule1,
    rule2,
    rule3,
    rule4,
    rule5,
    rule6,
    rule7,
    rule8,
    rule9,
    rule10
])


def get_upgrade_label(score):
    if score < 30:
        return "Low Upgrade Urgency"
    elif score < 65:
        return "Medium Upgrade Urgency"
    elif score < 80:
        return "High Upgrade Urgency"
    else:
        return "Critical Upgrade Urgency"


def explain_fuzzy_result_llm(
    total_urgency_value,
    ram_gap_value,
    gpu_gap_value,
    upgrade_cost_value,
    fuzzy_score,
    label,
    model="llama-3.1-8b-instant"
):
    prompt = f"""
You are explaining a fuzzy logic laptop upgrade decision to a non-technical user.

Fuzzy system inputs:
- Total urgency: {total_urgency_value}
- RAM gap in GB: {ram_gap_value}
- GPU gap score: {gpu_gap_value}
- Estimated upgrade cost: {upgrade_cost_value}
- Final fuzzy score: {fuzzy_score}
- Final fuzzy label: {label}

Explain why the fuzzy system gave this result.

Rules:
- Keep it short, 3 to 5 sentences.
- Use simple language.
- Do not mention code.
- Do not invent extra data.
- Base the explanation only on the given inputs.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You explain laptop upgrade decisions clearly."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=180
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return (
            f"The fuzzy system classified this laptop as '{label}'. "
            f"This was based on total urgency {total_urgency_value}, "
            f"RAM gap {ram_gap_value}GB, GPU gap score {gpu_gap_value}, "
            f"and estimated upgrade cost {upgrade_cost_value}."
        )


def run_fuzzy_upgrade_evaluator(
    total_urgency_value,
    ram_gap_value,
    gpu_gap_value,
    upgrade_cost_value
):
    simulator = ctrl.ControlSystemSimulation(upgrade_ctrl)

    simulator.input["total_urgency"] = total_urgency_value
    simulator.input["ram_gap_gb"] = ram_gap_value
    simulator.input["gpu_gap_score"] = gpu_gap_value
    simulator.input["upgrade_cost"] = upgrade_cost_value

    simulator.compute()

    score = simulator.output["upgrade_decision"]
    label = get_upgrade_label(score)

    explanation = explain_fuzzy_result_llm(
        total_urgency_value=total_urgency_value,
        ram_gap_value=ram_gap_value,
        gpu_gap_value=gpu_gap_value,
        upgrade_cost_value=upgrade_cost_value,
        fuzzy_score=round(score, 2),
        label=label
    )

    return {
        "score": round(float(score), 2),
        "decision": label,
        "advice": explanation
    }