"""Exploratory data analysis: brand, campaign, channel and driver analysis -> figures + summary tables."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import plotting as P

BRANDS = list(C.BRAND_COLORS)


def channel_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (campaign, channel) - for channel-wise analysis of a multi-label column."""
    long = df.assign(Channel=df["Channel_Used"].str.split(", ")).explode("Channel")
    return long[long["Channel"] != "Unknown"]


def brand_kpis(df):
    g = df.groupby("Brand")
    out = pd.DataFrame({
        "Campaigns": g.size(),
        "Total_Revenue": g["Revenue"].sum(),
        "Total_Spend": g["Total_Cost"].sum(),
        "Net_Profit": g["Revenue"].sum() - g["Total_Cost"].sum(),
        "Median_ROI": g["ROI"].median(),
        "Mean_ROI": g["ROI"].mean(),
        "Profit_Rate_%": g["Profit_Flag"].mean() * 100,
        "Avg_CTR_%": g["CTR"].mean() * 100,
        "Avg_Conv_Rate_%": g["Conversion_Rate"].mean() * 100,
        "Avg_Revenue_per_Conv": g["Revenue_per_Conversion"].mean(),
        "Avg_Cost_per_Conv": g["Acquisition_Cost"].mean(),
    })
    out["Portfolio_ROI"] = out["Net_Profit"] / out["Total_Spend"]
    return out


def group_perf(df, col):
    g = df.groupby(col)
    return pd.DataFrame({"Campaigns": g.size(), "Median_ROI": g["ROI"].median(),
                         "Profit_Rate_%": g["Profit_Flag"].mean() * 100,
                         "Avg_Revenue": g["Revenue"].mean(), "Avg_Spend": g["Total_Cost"].mean(),
                         "Avg_Engagement": g["Engagement_Score"].mean()}).sort_values("Median_ROI", ascending=False)


# --------------------------------------------------------------------------- figures
def fig_brand_overview(k):
    fig, axes = P.plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, col, title, fmt in [(axes[0], "Total_Revenue", "Total revenue", P.inr),
                                (axes[1], "Net_Profit", "Net profit (revenue − spend)", P.inr),
                                (axes[2], "Profit_Rate_%", "Share of campaigns in profit", lambda v, _=None: f"{v:.1f}%")]:
        ax.bar(k.index, k[col], color=[C.BRAND_COLORS[b] for b in k.index], width=0.55)
        for x, v in enumerate(k[col]):
            ax.text(x, v, fmt(v), ha="center", va="bottom", fontsize=9, color=C.INK["primary"])
        ax.yaxis.set_major_formatter(fmt)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        lo = k[col].min()
        if col == "Profit_Rate_%":
            ax.set_ylim(0, 100)
    return P.save(fig, "eda_01_brand_overview.png")


def fig_roi_distribution(df):
    fig, ax = P.plt.subplots(figsize=(9, 3.8))
    bins = np.linspace(-1, 6, 71)
    for b in BRANDS:
        ax.hist(df.loc[df.Brand == b, "ROI"].clip(upper=6), bins=bins, histtype="step", lw=2,
                color=C.BRAND_COLORS[b], label=b)
    ax.axvline(0, color=C.INK["secondary"], lw=1, ls="--")
    ax.text(0.05, ax.get_ylim()[1] * 0.92, "break-even (ROI = 0)", color=C.INK["secondary"], fontsize=8)
    ax.set_xlabel("ROI (values above 6 grouped at 6)")
    ax.set_ylabel("Campaigns")
    ax.set_title("ROI distribution by brand - long right tail, ~24% of campaigns lose money")
    ax.legend()
    return P.save(fig, "eda_02_roi_distribution.png")


def _heat(ax, pivot, fmt, title, cmap_cols=C.SEQ_BLUE):
    cmap = P.matplotlib.colors.LinearSegmentedColormap.from_list("seq", cmap_cols)
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
    vmin, vmax = np.nanmin(pivot.values), np.nanmax(pivot.values)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            dark = (v - vmin) / (vmax - vmin + 1e-9) > 0.55
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=9,
                    color="white" if dark else C.INK["primary"])
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=0)
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    ax.grid(False)
    ax.set_title(title)
    return im


def fig_campaign_type(df):
    fig, axes = P.plt.subplots(1, 2, figsize=(12, 3.8))
    pv = df.pivot_table(index="Campaign_Type", columns="Brand", values="Profit_Flag", aggfunc="mean") * 100
    _heat(axes[0], pv, lambda v: f"{v:.1f}%", "Profit rate by campaign type × brand (scale spans only ~2 pp)")
    pv2 = df.pivot_table(index="Campaign_Type", columns="Brand", values="ROI", aggfunc="median")
    _heat(axes[1], pv2, lambda v: f"{v:.2f}", "Median ROI by campaign type × brand")
    return P.save(fig, "eda_03_campaign_type_brand.png")


def fig_channels(df, chl):
    fig, axes = P.plt.subplots(1, 3, figsize=(13, 3.8))
    g = chl.groupby("Channel")
    s1 = (g["Profit_Flag"].mean() * 100).sort_values()
    s2 = g["ROI"].median().reindex(s1.index)
    s3 = g["Revenue"].mean().reindex(s1.index)
    for ax, s, t, fmt in [(axes[0], s1, "Profit rate by channel", lambda v: f"{v:.1f}%"),
                          (axes[1], s2, "Median ROI by channel", lambda v: f"{v:.2f}"),
                          (axes[2], s3, "Avg revenue per campaign", P.inr)]:
        ax.barh(s.index, s.values, color=C.SEQ_BLUE[4], height=0.6)
        for y, v in enumerate(s.values):
            ax.text(v, y, " " + fmt(v), va="center", fontsize=9, color=C.INK["primary"])
        ax.set_xlim(0, s.max() * 1.25)
        ax.set_title(t)
        ax.grid(axis="y", visible=False)
        if fmt is P.inr:
            ax.xaxis.set_major_formatter(P.inr)
    return P.save(fig, "eda_04_channel_effectiveness.png")


def fig_channel_count(df):
    g = df[df.Channel_Count > 0].groupby("Channel_Count")
    fig, axes = P.plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].bar(g.size().index.astype(str), g["Profit_Flag"].mean() * 100, color=C.SEQ_BLUE[4], width=0.55)
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Profit rate by number of channels used")
    axes[0].set_xlabel("Channels in campaign")
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    axes[1].bar(g.size().index.astype(str), g["Total_Cost"].mean(), color=C.SEQ_BLUE[4], width=0.55)
    axes[1].yaxis.set_major_formatter(P.inr)
    axes[1].set_title("Avg spend by number of channels used")
    axes[1].set_xlabel("Channels in campaign")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    return P.save(fig, "eda_05_channel_count.png")


def fig_spend_vs_revenue(df):
    s = df.sample(6000, random_state=C.RANDOM_STATE)
    fig, axes = P.plt.subplots(1, 2, figsize=(12, 4.6))
    for status in ["Loss", "Profit"]:
        d = s[s.Profit_Status == status]
        axes[0].scatter(d.Total_Cost, d.Revenue, s=7, alpha=0.4, lw=0, color=C.PROFIT_COLORS[status], label=status)
        axes[1].scatter(d.Acquisition_Cost, d.Revenue_per_Conversion, s=7, alpha=0.4, lw=0,
                        color=C.PROFIT_COLORS[status], label=status)
    lim = s.Total_Cost.max()
    axes[0].plot([0, lim], [0, lim], color=C.INK["secondary"], lw=1, ls="--")
    axes[0].text(lim * 0.62, lim * 0.5, "revenue = spend", color=C.INK["secondary"], fontsize=8, rotation=0)
    axes[0].set_xlabel("Total spend (Acquisition_Cost × Conversions)")
    axes[0].set_ylabel("Revenue")
    axes[0].xaxis.set_major_formatter(P.inr)
    axes[0].yaxis.set_major_formatter(P.inr)
    axes[0].set_title("Spend vs revenue")
    axes[1].set_xscale("log")
    axes[1].plot([200, 800], [200, 800], color=C.INK["secondary"], lw=1, ls="--")
    axes[1].set_xlabel("Acquisition cost per conversion (₹, log scale)")
    axes[1].set_ylabel("Revenue per conversion (₹)")
    axes[1].set_title("Profit happens when revenue per conversion > cost per conversion")
    for ax in axes:
        ax.legend(markerscale=3, loc="upper left")
    return P.save(fig, "eda_06_spend_vs_revenue.png")


def fig_correlation(df):
    cols = ["Duration", "Impressions", "Clicks", "Leads", "Conversions", "Engagement_Score",
            "Acquisition_Cost", "Total_Cost", "Revenue", "ROI", "Profit_Flag"]
    corr = df[cols].corr()
    fig, ax = P.plt.subplots(figsize=(9, 7))
    cmap = P.matplotlib.colors.LinearSegmentedColormap.from_list(
        "div", ["#e34948", "#f0a6a5", "#f0efec", "#86b6ef", "#2a78d6"])
    ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else C.INK["primary"])
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    ax.set_title("Correlation matrix (blue = positive, red = negative)")
    return P.save(fig, "eda_07_correlation.png")


def fig_monthly(df):
    m = df.assign(Month=df.Date.dt.to_period("M").dt.to_timestamp()).groupby(["Month", "Brand"])["Revenue"].sum().unstack()
    fig, ax = P.plt.subplots(figsize=(11, 3.8))
    for b in BRANDS:
        ax.plot(m.index, m[b], color=C.BRAND_COLORS[b], lw=2, marker="o", ms=4, label=b)
    ax.yaxis.set_major_formatter(P.inr)
    ax.set_title("Monthly revenue by brand (Jul 2024 – Jun 2025; June is partial - data ends 24 Jun)")
    ax.legend(ncol=3, loc="lower left")
    ax.set_ylim(0, m.values.max() * 1.15)
    ax.grid(axis="x", visible=False)
    return P.save(fig, "eda_08_monthly_revenue.png")


def fig_segments(df):
    fig, axes = P.plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, col in zip(axes, ["Target_Audience", "Customer_Segment", "Language"]):
        s = (df.groupby(col)["Profit_Flag"].mean() * 100).sort_values()
        ax.barh(s.index, s.values, color=C.SEQ_BLUE[4], height=0.6)
        for y, v in enumerate(s.values):
            ax.text(v, y, f" {v:.1f}%", va="center", fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_title(f"Profit rate by {col.replace('_', ' ').lower()}")
        ax.grid(axis="y", visible=False)
    return P.save(fig, "eda_09_audience_language.png")


def fig_cost_bands(df):
    bands = pd.cut(df.Acquisition_Cost, [0, 100, 200, 300, 400, 500, 600, 800, 1e6],
                   labels=["<100", "100-200", "200-300", "300-400", "400-500", "500-600", "600-800", ">800"])
    s = df.groupby(bands, observed=True)["Profit_Flag"].mean() * 100
    n = df.groupby(bands, observed=True).size()
    fig, ax = P.plt.subplots(figsize=(10, 3.6))
    ax.bar(s.index.astype(str), s.values, color=C.SEQ_BLUE[4], width=0.6)
    for x, (v, c) in enumerate(zip(s.values, n.values)):
        ax.text(x, v, f"{v:.0f}%\n(n={c:,})", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_xlabel("Acquisition cost per conversion (₹)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_title("Profit rate by cost per conversion - always profitable below ₹200, never above ₹800")
    ax.grid(axis="x", visible=False)
    return P.save(fig, "eda_10_cost_per_conversion_bands.png"), s


# --------------------------------------------------------------------------- main
def run_eda(df: pd.DataFrame) -> dict:
    P.apply_style()
    chl = channel_long(df)
    k = brand_kpis(df)
    fig_brand_overview(k)
    fig_roi_distribution(df)
    fig_campaign_type(df)
    fig_channels(df, chl)
    fig_channel_count(df)
    fig_spend_vs_revenue(df)
    fig_correlation(df)
    fig_monthly(df)
    fig_segments(df)
    _, cost_band = fig_cost_bands(df)

    cols = ["Campaign_ID", "Brand", "Campaign_Type", "Channel_Used", "Revenue", "Total_Cost", "ROI"]
    top = df.nlargest(10, "ROI")[cols]
    bottom = df.nsmallest(10, "ROI")[cols]
    top_rev = df.nlargest(10, "Revenue")[cols]

    tables = {
        "brand_kpis": k,
        "campaign_type": group_perf(df, "Campaign_Type"),
        "channel": group_perf(chl, "Channel"),
        "channel_count": group_perf(df, "Channel_Count"),
        "target_audience": group_perf(df, "Target_Audience"),
        "customer_segment": group_perf(df, "Customer_Segment"),
        "language": group_perf(df, "Language"),
        "brand_x_type": df.pivot_table(index="Campaign_Type", columns="Brand", values="Profit_Flag") * 100,
        "brand_x_channel": chl.pivot_table(index="Channel", columns="Brand", values="Profit_Flag") * 100,
        "cost_band_profit_rate": cost_band.to_frame("Profit_Rate_%"),
        "top10_roi": top, "bottom10_roi": bottom, "top10_revenue": top_rev,
    }
    out_dir = C.REPORTS_DIR / "eda_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv")
    return tables
