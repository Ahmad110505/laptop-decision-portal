import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def demographic_parity(df, group_col, pred_col="positive_pred"):
    rates = df.groupby(group_col)[pred_col].mean()
    return rates

def disparate_impact_ratio(df, group_col, pred_col="positive_pred"):
    rates = df.groupby(group_col)[pred_col].mean()
    min_rate = rates.min()
    max_rate = rates.max()
    return np.nan if max_rate == 0 else min_rate / max_rate

def equalized_odds_tpr(df, group_col, true_col="positive_true", pred_col="positive_pred"):
    rows = []
    for group, gdf in df.groupby(group_col):
        positives = gdf[gdf[true_col] == 1]
        if len(positives) == 0:
            tpr = np.nan
        else:
            tpr = (positives[pred_col] == 1).mean()
        rows.append((group, tpr))
    return pd.DataFrame(rows, columns=[group_col, "TPR"])

def mean_prediction_difference(df, group_col, pred_col="y_pred"):
    return df.groupby(group_col)[pred_col].mean()

def residual_error_parity(df, group_col, residual_col="residual"):
    return df.groupby(group_col)[residual_col].apply(lambda x: np.mean(np.abs(x)))

def r2_parity(df, group_col, true_col="y_true", pred_col="y_pred"):
    rows = []
    for group, gdf in df.groupby(group_col):
        if len(gdf) < 2:
            score = np.nan
        else:
            score = r2_score(gdf[true_col], gdf[pred_col])
        rows.append((group, score))
    return pd.DataFrame(rows, columns=[group_col, "R2"])
