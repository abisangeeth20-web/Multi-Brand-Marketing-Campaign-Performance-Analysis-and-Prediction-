"""Streamlit app - Multi-Brand Marketing Campaign Performance Analysis & Prediction.

Run:  streamlit run app.py
"""
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.eda import channel_long

st.set_page_config(page_title="Campaign Performance Predictor", page_icon="📈", layout="wide")

GRID = "rgba(137,135,129,0.25)"


# --------------------------------------------------------------------------- loaders
@st.cache_resource
def load_models():
    reg = joblib.load(C.MODELS_DIR / "revenue_regressor.joblib")
    clf = joblib.load(C.MODELS_DIR / "profit_classifier.joblib")
    meta = json.loads((C.MODELS_DIR / "model_metadata.json").read_text())
    return reg, clf, meta


@st.cache_data
def load_data():
    # Compact gzip file (works on Streamlit Cloud); fall back to the full CSV locally
    src = C.DASHBOARD_GZ if C.DASHBOARD_GZ.exists() else C.FEATURED_CSV
    df = pd.read_csv(src, usecols=C.DASHBOARD_COLS)
    df["Date"] = pd.to_datetime(df["Date"], format=C.DATE_FORMAT)
    return df


def style(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="",
                      font=dict(family="system-ui, Segoe UI, sans-serif"), title_x=0)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def inr(v):
    if abs(v) >= 1e7:
        return f"₹{v / 1e7:,.2f} Cr"
    if abs(v) >= 1e5:
        return f"₹{v / 1e5:,.2f} L"
    return f"₹{v:,.0f}"


# --------------------------------------------------------------------------- pages
def page_predict(reg, clf, meta):
    st.title("Campaign Revenue & Profit/Loss Predictor")
    st.caption("Enter campaign details. The same cleaning-time feature engineering (channel multi-label "
               "encoding, funnel rates, total spend, date parts) runs inside the saved model pipelines.")
    cats = meta["categories"]

    with st.form("campaign"):
        st.subheader("Campaign profile")
        c1, c2, c3 = st.columns(3)
        brand = c1.selectbox("Brand", cats["Brand"])
        ctype = c2.selectbox("Campaign type", cats["Campaign_Type"])
        lang = c3.selectbox("Language", cats["Language"])
        c1, c2, c3 = st.columns(3)
        audience = c1.selectbox("Target audience", cats["Target_Audience"])
        segment = c2.selectbox("Customer segment", cats["Customer_Segment"])
        channels = c3.multiselect("Channels used", meta["channels"], default=["Instagram", "YouTube"])
        c1, c2 = st.columns(2)
        run_date = c1.date_input("Campaign date", date(2025, 3, 15))
        duration = c2.slider("Duration (days)", 5, 30, 17)

        st.subheader("Funnel")
        c1, c2, c3, c4 = st.columns(4)
        impressions = c1.number_input("Impressions", 1000, 500000, 55000, step=1000)
        clicks = c2.number_input("Clicks", 0, 200000, 3900, step=100)
        leads = c3.number_input("Leads", 0, 200000, 1480, step=50)
        conversions = c4.number_input("Conversions", 1, 200000, 780, step=10)

        st.subheader("Financials")
        c1, c2, c3 = st.columns(3)
        acq = c1.number_input("Acquisition cost per conversion (₹)", 1.0, 20000.0, 210.0, step=10.0,
                              help="Total spend = acquisition cost × conversions")
        roi = c2.number_input("Expected ROI (revenue model input)", -1.0, 80.0, 1.25, step=0.05,
                              help="Used only by the revenue regressor. The profit classifier never sees ROI.")
        revenue_in = c3.number_input("Revenue (₹, profit model input)", 0.0, 1e8, float(conversions * 500),
                                     step=10000.0,
                                     help="Actual or forecast revenue for the profit/loss classifier. "
                                          "Default = conversions × ₹500 (the dataset's average order value).")
        submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    if not submitted:
        st.info("Fill in the form and press **Predict**.")
        return

    problems = []
    if clicks > impressions:
        problems.append("Clicks cannot exceed impressions.")
    if leads > clicks:
        problems.append("Leads cannot exceed clicks.")
    if conversions > leads:
        problems.append("Conversions cannot exceed leads.")
    if problems:
        for p in problems:
            st.error(p)
        return

    engagement = round((clicks + leads + conversions) / impressions * 100, 2)
    row = {"Brand": brand, "Campaign_Type": ctype, "Target_Audience": audience, "Language": lang,
           "Customer_Segment": segment, "Channel_Used": ", ".join(sorted(channels)) if channels else "Unknown",
           "Date": pd.Timestamp(run_date), "Duration": duration, "Impressions": impressions, "Clicks": clicks,
           "Leads": leads, "Conversions": conversions, "Acquisition_Cost": acq, "Engagement_Score": engagement}
    X_reg = pd.DataFrame([{**row, "ROI": roi}])
    X_clf = pd.DataFrame([{**row, "Revenue": revenue_in}])

    pred_rev = float(reg.predict(X_reg)[0])
    p_profit = float(clf.predict_proba(X_clf)[0, 1])
    is_profit = p_profit >= 0.5
    spend = acq * conversions

    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Predicted revenue", inr(pred_rev))
    k2.metric("Total spend", inr(spend))
    k3.metric("Profit / Loss prediction", "✅ PROFIT" if is_profit else "⚠️ LOSS")
    k4.metric("Probability of profit", f"{p_profit:.1%}")
    lo, hi = C.TRAIN_SPEND_RANGE
    if not lo <= spend <= hi:
        st.warning(f"Total spend {inr(spend)} is outside the range seen in training ({inr(lo)} – {inr(hi)}). "
                   f"Tree models cannot extrapolate, so treat the revenue figure as approximate. "
                   f"By the ROI formula it would be {inr(spend * (1 + roi))}.")
    st.caption(f"Engagement score (derived from funnel) = {engagement}  ·  "
               f"Revenue model: {meta['regression']['best_model']}  ·  "
               f"Profit model: {meta['classification']['best_model']}")

    c1, c2, c3 = st.columns(3)
    funnel = go.Figure(go.Funnel(y=["Impressions", "Clicks", "Leads", "Conversions"],
                                 x=[impressions, clicks, leads, conversions], textinfo="value+percent previous",
                                 marker_color=C.SEQ_BLUE[3:7]))
    c1.plotly_chart(style(funnel.update_layout(title="Campaign funnel")), width="stretch")

    bars = go.Figure(go.Bar(x=["Spend", "Predicted revenue", "Revenue (profit-model input)"],
                            y=[spend, pred_rev, revenue_in],
                            marker_color=[C.INK["muted"], C.BRAND_COLORS["Nykaa"], C.SEQ_BLUE[2]],
                            text=[inr(v) for v in [spend, pred_rev, revenue_in]], textposition="outside"))
    c2.plotly_chart(style(bars.update_layout(title="Spend vs revenue", showlegend=False)),
                    width="stretch")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=p_profit * 100, number={"suffix": "%"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": C.PROFIT_COLORS["Profit" if is_profit else "Loss"]},
               "steps": [{"range": [0, 50], "color": "rgba(227,73,72,0.12)"},
                         {"range": [50, 100], "color": "rgba(42,120,214,0.12)"}],
               "threshold": {"line": {"color": C.INK["secondary"], "width": 2}, "value": 50}}))
    c3.plotly_chart(style(gauge.update_layout(title="Probability of profit")), width="stretch")

    rpc = revenue_in / conversions
    st.markdown(
        f"**Break-even check:** revenue per conversion is **₹{rpc:,.0f}** vs acquisition cost per conversion "
        f"**₹{acq:,.0f}**. In the historical data a campaign is profitable exactly when revenue per conversion "
        f"exceeds cost per conversion; cost per conversion below ₹200 was profitable in 100% of campaigns, "
        f"above ₹800 in 0%.")


def page_dashboard(df):
    st.title("Campaign Performance Dashboard")
    f1, f2, f3, f4 = st.columns([1.2, 1.6, 1.6, 1.4])
    brands = f1.multiselect("Brand", list(C.BRAND_COLORS), default=list(C.BRAND_COLORS))
    types = f2.multiselect("Campaign type", sorted(df.Campaign_Type.unique()), default=sorted(df.Campaign_Type.unique()))
    auds = f3.multiselect("Target audience", sorted(df.Target_Audience.unique()),
                          default=sorted(df.Target_Audience.unique()))
    dr = f4.date_input("Date range", (df.Date.min().date(), df.Date.max().date()))
    d = df[df.Brand.isin(brands) & df.Campaign_Type.isin(types) & df.Target_Audience.isin(auds)]
    if isinstance(dr, tuple) and len(dr) == 2:
        d = d[(d.Date >= pd.Timestamp(dr[0])) & (d.Date <= pd.Timestamp(dr[1]))]
    if d.empty:
        st.warning("No campaigns match these filters.")
        return

    k = st.columns(5)
    k[0].metric("Campaigns", f"{len(d):,}")
    k[1].metric("Revenue", inr(d.Revenue.sum()))
    k[2].metric("Spend", inr(d.Total_Cost.sum()))
    k[3].metric("Net profit", inr(d.Revenue.sum() - d.Total_Cost.sum()))
    k[4].metric("Campaigns in profit", f"{d.Profit_Flag.mean():.1%}")

    cmap = C.BRAND_COLORS
    c1, c2 = st.columns(2)
    m = d.assign(Month=d.Date.dt.to_period("M").dt.to_timestamp()).groupby(["Month", "Brand"], as_index=False)["Revenue"].sum()
    fig = px.line(m, x="Month", y="Revenue", color="Brand", markers=True, color_discrete_map=cmap,
                  title="Monthly revenue by brand")
    c1.plotly_chart(style(fig), width="stretch")

    t = d.groupby(["Campaign_Type", "Brand"], as_index=False)["Profit_Flag"].mean()
    t["Profit rate %"] = t.Profit_Flag * 100
    fig = px.bar(t, x="Campaign_Type", y="Profit rate %", color="Brand", barmode="group", color_discrete_map=cmap,
                 title="Profit rate by campaign type", range_y=[0, 100])
    c2.plotly_chart(style(fig), width="stretch")

    c1, c2 = st.columns(2)
    chl = channel_long(d)
    ch = chl.groupby(["Channel", "Brand"], as_index=False).agg(ROI=("ROI", "median"))
    fig = px.bar(ch, x="Channel", y="ROI", color="Brand", barmode="group", color_discrete_map=cmap,
                 title="Median ROI by channel (multi-channel campaigns count once per channel)")
    c1.plotly_chart(style(fig), width="stretch")

    s = d.sample(min(4000, len(d)), random_state=0)
    fig = px.scatter(s, x="Total_Cost", y="Revenue", color="Profit_Status", opacity=0.45,
                     color_discrete_map=C.PROFIT_COLORS, hover_data=["Campaign_ID", "Brand", "ROI"],
                     title="Spend vs revenue (sample of 4,000)")
    fig.add_trace(go.Scatter(x=[0, s.Total_Cost.max()], y=[0, s.Total_Cost.max()], mode="lines",
                             line=dict(dash="dash", color=C.INK["muted"], width=1), name="break-even"))
    fig.update_traces(marker=dict(size=5), selector=dict(mode="markers"))
    c2.plotly_chart(style(fig), width="stretch")

    bands = pd.cut(d.Acquisition_Cost, [0, 100, 200, 300, 400, 500, 600, 800, 1e6],
                   labels=["<100", "100-200", "200-300", "300-400", "400-500", "500-600", "600-800", ">800"])
    b = (d.groupby(bands, observed=True)["Profit_Flag"].mean() * 100).reset_index()
    b.columns = ["Cost per conversion (₹)", "Profit rate %"]
    fig = px.bar(b, x="Cost per conversion (₹)", y="Profit rate %", range_y=[0, 105],
                 title="Profit rate by acquisition cost per conversion - the strongest profitability driver",
                 color_discrete_sequence=[C.SEQ_BLUE[4]], text_auto=".0f")
    st.plotly_chart(style(fig, 340), width="stretch")

    c1, c2 = st.columns(2)
    cols = ["Campaign_ID", "Brand", "Campaign_Type", "Channel_Used", "Revenue", "Total_Cost", "ROI"]
    c1.markdown("**Top 10 campaigns by ROI**")
    c1.dataframe(d.nlargest(10, "ROI")[cols], hide_index=True, width="stretch")
    c2.markdown("**Bottom 10 campaigns by ROI**")
    c2.dataframe(d.nsmallest(10, "ROI")[cols], hide_index=True, width="stretch")


def page_models(meta):
    st.title("Model Evaluation")
    reg = pd.read_csv(C.REPORTS_DIR / "regression_model_comparison.csv", index_col=0)
    clf = pd.read_csv(C.REPORTS_DIR / "classification_model_comparison.csv", index_col=0)
    rm, cm = meta["regression"]["test_metrics"], meta["classification"]["test_metrics"]

    st.subheader(f"Revenue regression - best: {meta['regression']['best_model']}")
    k = st.columns(4)
    k[0].metric("R²", f"{rm['R2']:.4f}")
    k[1].metric("RMSE", inr(rm["RMSE"]))
    k[2].metric("MAE", inr(rm["MAE"]))
    k[3].metric("MSE", f"{rm['MSE']:.3e}")
    st.dataframe(reg[["R2", "Train_R2", "RMSE", "MAE", "MSE", "Params"]].style.format(
        {"R2": "{:.4f}", "Train_R2": "{:.4f}", "RMSE": "{:,.0f}", "MAE": "{:,.0f}", "MSE": "{:.3e}"}),
        width="stretch")

    st.subheader(f"Profit/Loss classification - best: {meta['classification']['best_model']}")
    k = st.columns(5)
    for col, key in zip(k, ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]):
        col.metric(key.replace("_", "-"), f"{cm[key]:.4f}")
    st.dataframe(clf[["Accuracy", "Train_Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Params"]].style.format(
        {c: "{:.4f}" for c in ["Accuracy", "Train_Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]}),
        width="stretch")

    c1, c2 = st.columns(2)
    for col, img in [(c1, "model_revenue_actual_vs_pred.png"), (c2, "model_profit_confusion_matrix.png"),
                     (c1, "model_revenue_importance.png"), (c2, "model_profit_importance.png")]:
        p = C.FIGURES_DIR / img
        if p.exists():
            col.image(str(p), width="stretch")

    pre = meta["prelaunch_benchmark"]
    st.info(
        f"**Leakage & realism note.** ROI is excluded from the classifier. Without ROI the revenue model reaches "
        f"R² = {pre['revenue_r2_without_roi']:.3f}, and without revenue/ROI the profit model reaches "
        f"accuracy = {pre['profit_accuracy_without_revenue_roi']:.3f} (majority baseline "
        f"{pre['profit_majority_baseline']:.3f}). Revenue per conversion varies independently of every other "
        f"campaign attribute, so that part of revenue cannot be forecast before the campaign runs.")


# --------------------------------------------------------------------------- main
def main():
    try:
        reg, clf, meta = load_models()
    except FileNotFoundError:
        st.error("Models not found. Run `python run_pipeline.py` first.")
        st.stop()
    page = st.sidebar.radio("Navigate", ["🔮 Predict", "📊 Dashboard", "🧪 Model performance"])
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Trained on {meta['train_rows']:,} campaigns · tested on {meta['test_rows']:,}")
    if page.endswith("Predict"):
        page_predict(reg, clf, meta)
    elif page.endswith("Dashboard"):
        page_dashboard(load_data())
    else:
        page_models(meta)


main()
