"""Data collection + cleaning.

Two exact identities hold in the raw data (verified on >99.99% of complete rows):

  1. ROI              = (Revenue - Acquisition_Cost * Conversions) / (Acquisition_Cost * Conversions)
                        -> Acquisition_Cost is cost *per conversion*; total spend = Acquisition_Cost * Conversions
  2. Engagement_Score = (Clicks + Leads + Conversions) / Impressions * 100

They let us *recover* (not guess) a missing value whenever only one member of an
identity is null. Statistical imputation is used only for what cannot be recovered.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

FIN_COLS = ["Revenue", "Acquisition_Cost", "Conversions", "ROI"]
ENG_COLS = ["Impressions", "Clicks", "Leads", "Conversions", "Engagement_Score"]


# --------------------------------------------------------------------------- loading
def load_raw() -> pd.DataFrame:
    """Read the three brand CSVs into one DataFrame with a Brand column and a raw row index."""
    frames = []
    for brand, fname in C.RAW_FILES.items():
        df = pd.read_csv(C.RAW_DIR / fname)
        df.insert(0, "Brand", brand)
        df["_row"] = np.arange(len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- identities
def _financial_identity(df: pd.DataFrame) -> int:
    """Fill a single missing member of the ROI identity. Returns number of cells filled."""
    miss = df[FIN_COLS].isna()
    one = miss.sum(axis=1) == 1
    rev, acq, conv, roi = (df[c] for c in FIN_COLS)
    filled = 0

    m = one & miss["ROI"]
    df.loc[m, "ROI"] = ((rev - acq * conv) / (acq * conv))[m].round(2)
    filled += m.sum()

    m = one & miss["Revenue"]
    df.loc[m, "Revenue"] = (acq * conv * (1 + roi))[m].round(0)
    filled += m.sum()

    m = one & miss["Acquisition_Cost"]
    df.loc[m, "Acquisition_Cost"] = (rev / ((1 + roi) * conv))[m].round(2)
    filled += m.sum()

    m = one & miss["Conversions"]
    df.loc[m, "Conversions"] = (rev / ((1 + roi) * acq))[m].round(0)
    filled += m.sum()
    return int(filled)


def _engagement_identity(df: pd.DataFrame) -> int:
    """Fill a single missing member of the Engagement_Score identity."""
    miss = df[ENG_COLS].isna()
    one = miss.sum(axis=1) == 1
    imp, clk, lead, conv, eng = (df[c] for c in ENG_COLS)
    filled = 0

    m = one & miss["Engagement_Score"]
    df.loc[m, "Engagement_Score"] = ((clk + lead + conv) / imp * 100)[m].round(2)
    filled += m.sum()

    m = one & miss["Impressions"]
    df.loc[m, "Impressions"] = ((clk + lead + conv) / eng * 100)[m].round(0)
    filled += m.sum()

    for col, others in [("Clicks", lead + conv), ("Leads", clk + conv), ("Conversions", clk + lead)]:
        m = one & miss[col]
        df.loc[m, col] = (eng * imp / 100 - others)[m].round(0)
        filled += m.sum()
    return int(filled)


def _validate_identities(df: pd.DataFrame, log: dict) -> None:
    """Recompute ROI / Engagement for complete rows and correct values that break the identity."""
    cost = df["Acquisition_Cost"] * df["Conversions"]
    roi_calc = ((df["Revenue"] - cost) / cost).round(2)
    bad = df["ROI"].notna() & roi_calc.notna() & ((df["ROI"] - roi_calc).abs() > 0.011)
    df.loc[bad, "ROI"] = roi_calc[bad]
    log["roi_values_corrected"] = int(bad.sum())

    eng_calc = ((df["Clicks"] + df["Leads"] + df["Conversions"]) / df["Impressions"] * 100).round(2)
    bad = df["Engagement_Score"].notna() & eng_calc.notna() & ((df["Engagement_Score"] - eng_calc).abs() > 0.011)
    df.loc[bad, "Engagement_Score"] = eng_calc[bad]
    log["engagement_values_corrected"] = int(bad.sum())


# --------------------------------------------------------------------------- cleaning
def _standardize_text(df: pd.DataFrame) -> None:
    for col in ["Campaign_Type", "Target_Audience", "Language", "Customer_Segment"]:
        df[col] = df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)

    def norm_channels(v):
        if pd.isna(v):
            return np.nan
        parts = sorted({p.strip().title() for p in str(v).split(",") if p.strip()})
        parts = ["YouTube" if p == "Youtube" else "WhatsApp" if p == "Whatsapp" else p for p in parts]
        return ", ".join(sorted(parts)) if parts else np.nan

    df["Channel_Used"] = df["Channel_Used"].map(norm_channels)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Full cleaning pipeline. Returns the cleaned frame and a log of every action taken."""
    df = raw.copy()
    log: dict = {"raw_rows": len(df), "raw_nulls": df.drop(columns="_row").isna().sum().to_dict()}

    # 1. Campaign_ID - IDs are strictly sequential per brand, so missing ones are reconstructed exactly
    expected = df["Brand"].map(C.ID_PREFIX) + "-" + (C.ID_START + df["_row"]).astype(str)
    log["campaign_id_reconstructed"] = int(df["Campaign_ID"].isna().sum())
    log["campaign_id_mismatch"] = int((df["Campaign_ID"].notna() & (df["Campaign_ID"] != expected)).sum())
    df["Campaign_ID"] = df["Campaign_ID"].fillna(expected)

    # 2. Duplicates
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "_row"])
    df = df.drop_duplicates(subset="Campaign_ID", keep="first")
    log["duplicates_removed"] = before - len(df)

    # 3. Type conversion + text standardisation
    _standardize_text(df)
    df["Date"] = pd.to_datetime(df["Date"], format=C.DATE_FORMAT, errors="coerce")

    # 4. Validate & correct ROI / Engagement on complete rows
    _validate_identities(df, log)

    # 5. Exact recovery via identities (iterate - one recovery can unlock another)
    rec_fin = rec_eng = 0
    for _ in range(3):
        rec_eng += _engagement_identity(df)
        rec_fin += _financial_identity(df)
    log["recovered_via_roi_identity"] = rec_fin
    log["recovered_via_engagement_identity"] = rec_eng

    # 6. Rows whose financials still cannot be made consistent are dropped (target integrity)
    unrecoverable = df[FIN_COLS].isna().any(axis=1)
    log["rows_dropped_unrecoverable_financials"] = int(unrecoverable.sum())
    df = df[~unrecoverable].copy()

    # 7. Remaining funnel gaps: brand-level median conversion ratios, re-running the
    #    engagement identity after each step so rows stay internally consistent.
    ratio_filled = 0
    steps = [("Leads", "Conversions", "Leads"),      # Leads = Conversions / median(Conv/Leads)
             ("Clicks", "Leads", "Clicks"),          # Clicks = Leads / median(Leads/Clicks)
             ("Impressions", "Clicks", "Impressions")]
    for target, base, _ in steps:
        _engagement_identity(df)
        ratio = (df[base] / df[target]).groupby(df["Brand"]).transform("median")
        m = df[target].isna() & df[base].notna()
        df.loc[m, target] = (df[base] / ratio)[m].round(0)
        ratio_filled += int(m.sum())
    _engagement_identity(df)
    log["funnel_filled_by_median_ratio"] = ratio_filled
    # Funnel must be monotonic: Impressions >= Clicks >= Leads >= Conversions
    df["Leads"] = df[["Leads", "Conversions"]].max(axis=1)
    df["Clicks"] = df[["Clicks", "Leads"]].max(axis=1)
    df["Impressions"] = df[["Impressions", "Clicks"]].max(axis=1)
    df["Engagement_Score"] = ((df["Clicks"] + df["Leads"] + df["Conversions"]) / df["Impressions"] * 100).round(2)

    # 8. Duration -> brand median; categoricals -> brand mode; channels -> 'Unknown'
    log["duration_median_filled"] = int(df["Duration"].isna().sum())
    df["Duration"] = df["Duration"].fillna(df.groupby("Brand")["Duration"].transform("median"))
    cat_filled = {}
    for col in ["Campaign_Type", "Target_Audience", "Language", "Customer_Segment"]:
        cat_filled[col] = int(df[col].isna().sum())
        mode = df.groupby("Brand")[col].transform(lambda s: s.mode().iloc[0])
        df[col] = df[col].fillna(mode)
    log["categorical_mode_filled"] = cat_filled
    log["channel_unknown_filled"] = int(df["Channel_Used"].isna().sum())
    df["Channel_Used"] = df["Channel_Used"].fillna("Unknown")

    # 9. Missing dates -> sampled from that brand's observed dates (keeps the seasonal shape;
    #    a single median date would create an artificial spike). Seeded => reproducible.
    rng = np.random.default_rng(C.RANDOM_STATE)
    log["date_sampled_filled"] = int(df["Date"].isna().sum())
    for brand, idx in df[df["Date"].isna()].groupby("Brand").groups.items():
        pool = df.loc[(df["Brand"] == brand) & df["Date"].notna(), "Date"].to_numpy()
        df.loc[idx, "Date"] = rng.choice(pool, size=len(idx))

    # 10. Final dtypes
    for col in ["Duration", "Impressions", "Clicks", "Leads", "Conversions"]:
        df[col] = df[col].round().astype("int64")
    df["Revenue"] = df["Revenue"].round(0)
    for col in ["Campaign_Type", "Target_Audience", "Language", "Customer_Segment", "Brand"]:
        df[col] = df[col].astype(str)

    df = df.drop(columns="_row").reset_index(drop=True)
    log["clean_rows"] = len(df)
    log["remaining_nulls"] = int(df.isna().sum().sum())
    return df, log
