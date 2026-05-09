import pandas as pd
from sklearn.metrics import r2_score

class ClassificationBiasFairness:
    def __init__(self, dir_lower=0.8, dir_upper=1.2, dp_threshold=0.05, eo_threshold=0.05):
        self.dir_lower = dir_lower
        self.dir_upper = dir_upper
        self.dp_threshold = dp_threshold
        self.eo_threshold = eo_threshold

    def selection_rate(self, df, group_column, group_value, outcome_col):
        """
        Selection rate for one group.
        outcome_col can be:
        - original label column before training
        - prediction/decision column after training
        """
        group = df[df[group_column] == group_value]

        if len(group) == 0:
            return 0.0

        return group[outcome_col].mean()

    def disparate_impact_ratio(self, df, group_column, protected_group, privileged_group, outcome_col):
        """
        DIR = selection_rate(protected_group) / selection_rate(privileged_group)
        """
        protected_rate = self.selection_rate(df, group_column, protected_group, outcome_col)
        privileged_rate = self.selection_rate(df, group_column, privileged_group, outcome_col)

        if privileged_rate == 0:
            return 0.0

        return protected_rate / privileged_rate

    def dir_fairness(self, dir_value):
        if self.dir_lower <= dir_value <= self.dir_upper:
            return "fair"
        return "bias"

    def demographic_parity_difference(self, df, group_column, group_a, group_b, outcome_col):
        """
        DP difference = absolute difference in selection rates
        """
        rate_a = self.selection_rate(df, group_column, group_a, outcome_col)
        rate_b = self.selection_rate(df, group_column, group_b, outcome_col)

        return abs(rate_a - rate_b)

    def dp_fairness(self, dp_value):
        if dp_value <= self.dp_threshold:
            return "fair"
        return "bias"

    def equalized_odds_for_group(self, df, group_column, group_value, actual_col, decision_col):
        """
        Returns TPR and FPR for one group.

        actual_col   = true labels
        decision_col = predicted labels / model decision
        """
        group = df[df[group_column] == group_value]

        tp = ((group[actual_col] == 1) & (group[decision_col] == 1)).sum()
        fn = ((group[actual_col] == 1) & (group[decision_col] == 0)).sum()
        fp = ((group[actual_col] == 0) & (group[decision_col] == 1)).sum()
        tn = ((group[actual_col] == 0) & (group[decision_col] == 0)).sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {"TPR": tpr, "FPR": fpr}

    def equalized_odds_difference(self, df, group_column, group_a, group_b, actual_col, decision_col):
        group_a_metrics = self.equalized_odds_for_group(df, group_column, group_a, actual_col, decision_col)
        group_b_metrics = self.equalized_odds_for_group(df, group_column, group_b, actual_col, decision_col)

        tpr_diff = abs(group_a_metrics["TPR"] - group_b_metrics["TPR"])
        fpr_diff = abs(group_a_metrics["FPR"] - group_b_metrics["FPR"])

        return {
            "group_a": group_a,
            "group_b": group_b,
            "TPR_diff": tpr_diff,
            "FPR_diff": fpr_diff,
            "group_a_metrics": group_a_metrics,
            "group_b_metrics": group_b_metrics
        }

    def eo_fairness(self, eo_result):
        if eo_result["TPR_diff"] <= self.eo_threshold and eo_result["FPR_diff"] <= self.eo_threshold:
            return "fair"
        return "bias"

    def check_dir(self, df, group_column, protected_group, privileged_group, outcome_col):
        dir_value = self.disparate_impact_ratio(df, group_column, protected_group, privileged_group, outcome_col)

        return {
            "metric": "Disparate Impact Ratio",
            "value": dir_value,
            "fairness": self.dir_fairness(dir_value)
        }

    def check_dp(self, df, group_column, group_a, group_b, outcome_col):
        dp_value = self.demographic_parity_difference(df, group_column, group_a, group_b, outcome_col)

        return {
            "metric": "Demographic Parity Difference",
            "value": dp_value,
            "fairness": self.dp_fairness(dp_value)
        }

    def check_eo(self, df, group_column, group_a, group_b, actual_col, decision_col):
        eo_result = self.equalized_odds_difference(
            df, group_column, group_a, group_b, actual_col, decision_col
        )

        tpr_diff = round(eo_result["TPR_diff"], 3)
        fpr_diff = round(eo_result["FPR_diff"], 3)

        return {
            "group_a": group_a,
            "group_b": group_b,
            "group_a_metrics": {
                "TPR": round(eo_result["group_a_metrics"]["TPR"], 3),
                "FPR": round(eo_result["group_a_metrics"]["FPR"], 3)
            },
            "group_b_metrics": {
                "TPR": round(eo_result["group_b_metrics"]["TPR"], 3),
                "FPR": round(eo_result["group_b_metrics"]["FPR"], 3)
            },
            "differences": {
                "TPR_difference": tpr_diff,
                "FPR_difference": fpr_diff
            },
            "metric": "Equalized Odds",
            "fairness": "fair" if tpr_diff <= self.eo_threshold and fpr_diff <= self.eo_threshold else "bias"
        }
             
  
  
    def check_feature_bias_across_groups(self, df, group_column, groups, outcome_col):
        """
        Check one feature across multiple groups using DIR and DP.
        Returns all pairwise comparisons.
        The student decides the final feature bias.
        """
        comparisons = []

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group_a = groups[i]
                group_b = groups[j]

                rate_a = self.selection_rate(df, group_column, group_a, outcome_col)
                rate_b = self.selection_rate(df, group_column, group_b, outcome_col)

                # DIR = smaller selection rate / larger selection rate
                larger_rate = max(rate_a, rate_b)
                smaller_rate = min(rate_a, rate_b)

                if larger_rate == 0:
                    dir_value = 0.0
                else:
                    dir_value = float(round(smaller_rate / larger_rate, 3))

                dp_value = float(round(abs(rate_a - rate_b), 3))

                dir_fairness = "fair" if self.dir_lower <= dir_value <= self.dir_upper else "bias"
                dp_fairness = "fair" if dp_value <= self.dp_threshold else "bias"

                comparisons.append({
                    "groups": f"{group_a} vs {group_b}",
                    "DIR": dir_value,
                    "DIR_fairness": dir_fairness,
                    "DP": dp_value,
                    "DP_fairness": dp_fairness
                })

        return comparisons

    def check_eo_across_groups(self, df, group_column, groups, actual_col, decision_col):
        """
        Check Equalized Odds across multiple groups.
        Returns EO results for all pairwise group comparisons.
        """
        comparisons = []

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group_a = groups[i]
                group_b = groups[j]

                eo_result = self.check_eo(
                    df=df,
                    group_column=group_column,
                    group_a=group_a,
                    group_b=group_b,
                    actual_col=actual_col,
                    decision_col=decision_col
                )

                comparisons.append({
                    "groups": f"{group_a} vs {group_b}",
                    "group_a_metrics": {
                        "TPR": round(eo_result["group_a_metrics"]["TPR"], 3),
                        "FPR": round(eo_result["group_a_metrics"]["FPR"], 3)
                    },
                    "group_b_metrics": {
                        "TPR": round(eo_result["group_b_metrics"]["TPR"], 3),
                        "FPR": round(eo_result["group_b_metrics"]["FPR"], 3)
                    },
                    "TPR_diff": round(eo_result["differences"]["TPR_difference"], 3),
                    "FPR_diff": round(eo_result["differences"]["FPR_difference"], 3),
                    "EO_fairness": eo_result["fairness"]
                })

        return comparisons


class RegressionBiasFairness:
    def __init__(self, mpd_threshold=0.05, threshold_type="relative", mae_threshold=0.05):
        """
        mpd_threshold:
            - if threshold_type = 'absolute', threshold is in original units
            - if threshold_type = 'relative', threshold is a ratio like 0.05 = 5%
        """
        self.mpd_threshold = mpd_threshold
        self.threshold_type = threshold_type
        self.mae_threshold = mae_threshold

    def mean_value(self, df, group_column, group_value, value_col):
        group = df[df[group_column] == group_value]

        if len(group) == 0:
            return 0.0

        return float(group[value_col].mean())

    def mean_value_difference(self, df, group_column, group_a, group_b, value_col):
        mean_a = self.mean_value(df, group_column, group_a, value_col)
        mean_b = self.mean_value(df, group_column, group_b, value_col)

        mpd = abs(mean_a - mean_b)

        return {
            "group_a": group_a,
            "group_b": group_b,
            "mean_value_a": round(mean_a, 3),
            "mean_value_b": round(mean_b, 3),
            "MPD": round(mpd, 3)
        }

    def relative_mpd(self, df, mpd_value, value_col):
        overall_mean = float(df[value_col].mean())

        if overall_mean == 0:
            return 0.0

        return mpd_value / overall_mean

    def mpd_fairness(self, mpd_value, df=None, value_col=None):
        if self.threshold_type == "absolute":
            return "fair" if mpd_value <= self.mpd_threshold else "bias"

        elif self.threshold_type == "relative":
            relative_value = self.relative_mpd(df, mpd_value, value_col)
            return "fair" if relative_value <= self.mpd_threshold else "bias"

        else:
            raise ValueError("threshold_type must be 'absolute' or 'relative'")

    def check_mpd(self, df, group_column, group_a, group_b, value_col):
        result = self.mean_value_difference(df, group_column, group_a, group_b, value_col)

        result["metric"] = "Mean Prediction Difference"

        if self.threshold_type == "absolute":
            result["threshold_type"] = "absolute"
            result["threshold"] = self.mpd_threshold
            result["fairness"] = self.mpd_fairness(result["MPD"])

        else:
            relative_value = self.relative_mpd(df, result["MPD"], value_col)
            result["threshold_type"] = "relative"
            result["relative_MPD"] = round(relative_value, 3)
            result["threshold"] = self.mpd_threshold
            result["fairness"] = self.mpd_fairness(
                result["MPD"], df=df, value_col=value_col
            )

        return result
        

    # -----------------------------
    # MAE fairness methods
    # -----------------------------
    def group_mae(self, df, group_column, group_value, actual_col, predicted_col):
        """
        Compute MAE for one group.
        """
        group = df[df[group_column] == group_value]

        if len(group) == 0:
            return 0.0

        mae = (group[actual_col] - group[predicted_col]).abs().mean()
        return float(mae)

    def mae_difference(self, df, group_column, group_a, group_b, actual_col, predicted_col):
        """
        Difference in MAE between two groups.
        """
        mae_a = self.group_mae(df, group_column, group_a, actual_col, predicted_col)
        mae_b = self.group_mae(df, group_column, group_b, actual_col, predicted_col)

        return {
            "group_a": group_a,
            "group_b": group_b,
            "MAE_a": round(mae_a, 3),
            "MAE_b": round(mae_b, 3),
            "MAE_diff": round(abs(mae_a - mae_b), 3)
        }

    def relative_mae_difference(self, df, mae_diff, actual_col):
        """
        Relative MAE difference compared with the mean of the actual values.
        """
        overall_mean = float(df[actual_col].mean())

        if overall_mean == 0:
            return 0.0

        return mae_diff / overall_mean

    def mae_fairness(self, mae_diff, df=None, actual_col=None):
        """
        Decide fairness for MAE difference.
        """
        if self.threshold_type == "absolute":
            return "fair" if mae_diff <= self.mae_threshold else "bias"

        elif self.threshold_type == "relative":
            relative_value = self.relative_mae_difference(df, mae_diff, actual_col)
            return "fair" if relative_value <= self.mae_threshold else "bias"

        else:
            raise ValueError("threshold_type must be 'absolute' or 'relative'")

    def check_mae(self, df, group_column, group_a, group_b, actual_col, predicted_col):
        """
        Check MAE fairness between two groups.
        """
        result = self.mae_difference(
            df=df,
            group_column=group_column,
            group_a=group_a,
            group_b=group_b,
            actual_col=actual_col,
            predicted_col=predicted_col
        )

        result["metric"] = "Mean Absolute Error"

        if self.threshold_type == "absolute":
            result["threshold_type"] = "absolute"
            result["threshold"] = self.mae_threshold
            result["fairness"] = self.mae_fairness(result["MAE_diff"])

        else:
            relative_value = self.relative_mae_difference(df, result["MAE_diff"], actual_col)
            result["threshold_type"] = "relative"
            result["relative_MAE_diff"] = round(relative_value, 4)
            result["threshold"] = self.mae_threshold
            result["fairness"] = self.mae_fairness(
                result["MAE_diff"], df=df, actual_col=actual_col
            )

        return result

    def check_mae_across_groups(self, df, group_column, groups, actual_col, predicted_col):
        
        """
        Check MAE fairness across all group pairs.
        """
        comparisons = []

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group_a = groups[i]
                group_b = groups[j]

                result = self.check_mae(
                    df=df,
                    group_column=group_column,
                    group_a=group_a,
                    group_b=group_b,
                    actual_col=actual_col,
                    predicted_col=predicted_col
                )

                row = {
                    "groups": f"{group_a} vs {group_b}",
                    "MAE_a": result["MAE_a"],
                    "MAE_b": result["MAE_b"],
                    "MAE_diff": result["MAE_diff"],
                    "MAE_fairness": result["fairness"]
                }

                if self.threshold_type == "relative":
                    row["relative_MAE_diff"] = result["relative_MAE_diff"]

                comparisons.append(row)

        return comparisons
        
    def group_r2(self, df, group_column, group_value, actual_col, predicted_col):
        """
        Compute R² for one group.
        """
        group = df[df[group_column] == group_value]

        # R² needs at least 2 rows
        if len(group) < 2:
            return 0.0

        # If the actual values are constant, R² is not meaningful
        if group[actual_col].nunique() < 2:
            return 0.0

        r2 = r2_score(group[actual_col], group[predicted_col])
        return float(r2)   
    
    
    def r2_difference(self, df, group_column, group_a, group_b, actual_col, predicted_col):
        """
        Difference in R² between two groups.
        """
        r2_a = self.group_r2(df, group_column, group_a, actual_col, predicted_col)
        r2_b = self.group_r2(df, group_column, group_b, actual_col, predicted_col)

        return {
            "group_a": group_a,
            "group_b": group_b,
            "R2_a": round(r2_a, 3),
            "R2_b": round(r2_b, 3),
            "R2_diff": round(abs(r2_a - r2_b), 3)
        }
        
    def r2_fairness(self, r2_diff, r2_threshold=0.05):
        """
        Small R² difference -> fair
        Large R² difference -> bias
        """
        if r2_diff <= r2_threshold:
            return "fair"
        return "bias"
    
    
    def check_r2_parity(self, df, group_column, group_a, group_b, actual_col, predicted_col, r2_threshold=0.1):
            """
            Check R² performance parity between two groups.
            """
            result = self.r2_difference(
                df=df,
                group_column=group_column,
                group_a=group_a,
                group_b=group_b,
                actual_col=actual_col,
                predicted_col=predicted_col
            )

            result["metric"] = "R² Performance Parity"
            result["threshold"] = r2_threshold
            result["fairness"] = self.r2_fairness(result["R2_diff"], r2_threshold)

            return result
            
    def check_r2_parity_across_groups(self, df, group_column, groups, actual_col, predicted_col, r2_threshold=0.1):
        """
        Check R² performance parity across all group pairs.
        """
        comparisons = []

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group_a = groups[i]
                group_b = groups[j]

                result = self.check_r2_parity(
                    df=df,
                    group_column=group_column,
                    group_a=group_a,
                    group_b=group_b,
                    actual_col=actual_col,
                    predicted_col=predicted_col,
                    r2_threshold=r2_threshold
                )

                comparisons.append({
                    "groups": f"{group_a} vs {group_b}",
                    "R2_a": result["R2_a"],
                    "R2_b": result["R2_b"],
                    "R2_diff": result["R2_diff"],
                    "R2_fairness": result["fairness"]
                })

        return comparisons        