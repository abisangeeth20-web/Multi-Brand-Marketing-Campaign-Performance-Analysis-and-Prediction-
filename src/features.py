"""Feature engineering.

* Profit_Flag            : 1 if ROI > 0 (campaign made money) else 0
* Multi-label encoding   : Channel_Used ("WhatsApp, YouTube") -> one binary column per channel + Channel_Count
* Derived funnel / cost  : CTR, Lead_Rate, Conversion_Rate, Total_Cost, Cost_per_Click, date parts

`CampaignFeatureBuilder` is an sklearn transformer, so the *same* code runs inside the
saved model pipelines at training time and in the Streamlit app at prediction time.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler

from . import config as C

CHANNEL_COLS = [f"Ch_{c}" for c in C.CHANNELS]

# Shared inputs available for every campaign
BASE_NUMERIC = ["Duration", "Impressions", "Clicks", "Leads", "Conversions", "Acquisition_Cost",
                "Engagement_Score", "Month", "Day_of_Week", "CTR", "Lead_Rate", "Conversion_Rate",
                "Total_Cost", "Cost_per_Click", "Channel_Count"] + CHANNEL_COLS

# Regression target = Revenue. ROI is allowed here (the spec only bans ROI from classification).
REG_NUMERIC = BASE_NUMERIC + ["ROI"]
# Classification target = Profit_Flag. ROI is EXCLUDED (it defines the label -> leakage).
# Revenue_to_Cost is never created because it equals 1 + ROI.
CLF_NUMERIC = BASE_NUMERIC + ["Revenue", "Revenue_per_Conversion"]

# "Pre-launch" variants: only what is known before results come in (no ROI / Revenue).
PRELAUNCH_NUMERIC = BASE_NUMERIC


def encode_channels(series: pd.Series) -> pd.DataFrame:
    """Multi-label (one-vs-rest) encoding of the comma-separated Channel_Used column."""
    lists = series.fillna("Unknown").map(lambda s: [p.strip() for p in str(s).split(",") if p.strip()])
    mlb = MultiLabelBinarizer(classes=C.CHANNELS)
    with warnings.catch_warnings():               # 'Unknown' is intentionally not a class -> all zeros
        warnings.simplefilter("ignore")
        mat = mlb.fit_transform(lists)
    out = pd.DataFrame(mat, columns=CHANNEL_COLS, index=series.index)
    out["Channel_Count"] = out.sum(axis=1)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered columns. Works on a full dataset or a single app input row."""
    out = df.copy()
    date = pd.to_datetime(out["Date"], errors="coerce")
    out["Month"] = date.dt.month.fillna(6).astype(int)
    out["Day_of_Week"] = date.dt.dayofweek.fillna(3).astype(int)

    safe = lambda s: s.replace(0, np.nan)
    out["CTR"] = out["Clicks"] / safe(out["Impressions"])
    out["Lead_Rate"] = out["Leads"] / safe(out["Clicks"])
    out["Conversion_Rate"] = out["Conversions"] / safe(out["Leads"])
    out["Total_Cost"] = out["Acquisition_Cost"] * out["Conversions"]
    out["Cost_per_Click"] = out["Total_Cost"] / safe(out["Clicks"])
    if "Revenue" in out:
        out["Revenue_per_Conversion"] = out["Revenue"] / safe(out["Conversions"])

    out = out.drop(columns=[c for c in CHANNEL_COLS + ["Channel_Count"] if c in out], errors="ignore")
    out = pd.concat([out, encode_channels(out["Channel_Used"])], axis=1)
    return out.fillna({c: 0 for c in ["CTR", "Lead_Rate", "Conversion_Rate", "Cost_per_Click",
                                      "Revenue_per_Conversion"] if c in out})


def add_profit_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Profit_Flag"] = (out["ROI"] > 0).astype(int)
    out["Profit_Status"] = np.where(out["Profit_Flag"] == 1, "Profit", "Loss")
    return out


class CampaignFeatureBuilder(BaseEstimator, TransformerMixin):
    """sklearn step: raw campaign columns -> engineered feature frame."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_features(pd.DataFrame(X))


def build_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), C.CATEGORICAL_COLS),
         ("num", StandardScaler(), numeric_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
