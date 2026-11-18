"""Model building, tuning, evaluation and persistence.

Regression     : Revenue      <- Linear Regression, Decision Tree, Random Forest, Hist Gradient Boosting
Classification : Profit_Flag  <- Logistic Regression, Decision Tree, Random Forest, Hist Gradient Boosting

Leakage control: ROI is never an input to the classifier (it defines the label), and no
feature equal to Revenue / Total_Cost (= 1 + ROI) is created.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import config as C
from . import features as F
from . import plotting as P

RAW_INPUTS = ["Brand", "Campaign_Type", "Target_Audience", "Language", "Customer_Segment", "Channel_Used",
              "Date", "Duration", "Impressions", "Clicks", "Leads", "Conversions", "Acquisition_Cost",
              "Engagement_Score"]
REG_INPUTS = RAW_INPUTS + ["ROI"]
CLF_INPUTS = RAW_INPUTS + ["Revenue"]          # ROI deliberately excluded
TUNE_SAMPLE = 30_000
RS = C.RANDOM_STATE


def make_pipeline(numeric_cols, model) -> Pipeline:
    return Pipeline([("features", F.CampaignFeatureBuilder()),
                     ("prep", F.build_preprocessor(numeric_cols)),
                     ("model", model)])


def regression_candidates():
    return {
        "Linear Regression": (LinearRegression(), None),
        "Decision Tree": (DecisionTreeRegressor(random_state=RS),
                          {"model__max_depth": [8, 12, 16, 20, None], "model__min_samples_leaf": [1, 5, 10, 20]}),
        # Forest size is capped (trees / leaf size) so the saved model stays < 25 MB (GitHub web-upload limit,
        # needed for Streamlit Cloud deployment). 30 vs 150 trees changes test R² by only ~0.0002.
        "Random Forest": (RandomForestRegressor(random_state=RS, n_jobs=-1),
                          {"model__n_estimators": [30], "model__max_depth": [12, 16, 20],
                           "model__min_samples_leaf": [3, 5], "model__max_features": [0.5, 0.8]}),
        "Hist Gradient Boosting": (HistGradientBoostingRegressor(random_state=RS),
                                   {"model__max_iter": [300, 600, 900], "model__learning_rate": [0.05, 0.1, 0.2],
                                    "model__max_leaf_nodes": [31, 63, 127], "model__l2_regularization": [0, 1.0]}),
    }


def classification_candidates():
    return {
        "Logistic Regression": (LogisticRegression(max_iter=3000),
                                {"model__C": [0.1, 1, 10, 100, 1000]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=RS),
                          {"model__max_depth": [6, 10, 14, 18, None], "model__min_samples_leaf": [1, 5, 10, 20],
                           "model__criterion": ["gini", "entropy"]}),
        "Random Forest": (RandomForestClassifier(random_state=RS, n_jobs=-1),
                          {"model__n_estimators": [60, 100], "model__max_depth": [12, 16, 20],
                           "model__min_samples_leaf": [1, 3], "model__max_features": ["sqrt", 0.5]}),
        "Hist Gradient Boosting": (HistGradientBoostingClassifier(random_state=RS),
                                   {"model__max_iter": [300, 600], "model__learning_rate": [0.05, 0.1, 0.2],
                                    "model__max_leaf_nodes": [31, 63, 127]}),
    }


def _tune(pipe, grid, X, y, scoring):
    """Randomized search on a stratifiable subsample, then return the best params."""
    if grid is None:
        return {}
    n = min(TUNE_SAMPLE, len(X))
    idx = np.random.default_rng(RS).choice(len(X), n, replace=False)
    n_iter = min(8, int(np.prod([len(v) for v in grid.values()])))
    search = RandomizedSearchCV(pipe, grid, n_iter=n_iter, cv=3, scoring=scoring,
                                random_state=RS, n_jobs=-1 if "Forest" not in type(pipe[-1]).__name__ else 1)
    search.fit(X.iloc[idx], y.iloc[idx])
    return search.best_params_


def reg_metrics(y, p):
    mse = mean_squared_error(y, p)
    return {"R2": r2_score(y, p), "RMSE": float(np.sqrt(mse)), "MAE": mean_absolute_error(y, p), "MSE": mse}


def clf_metrics(y, p, proba):
    return {"Accuracy": accuracy_score(y, p), "Precision": precision_score(y, p), "Recall": recall_score(y, p),
            "F1": f1_score(y, p), "ROC_AUC": roc_auc_score(y, proba)}


def run_task(task, Xtr, Xte, ytr, yte, numeric_cols, candidates, log):
    rows, fitted = [], {}
    for name, (model, grid) in candidates.items():
        t0 = time.time()
        pipe = make_pipeline(numeric_cols, model)
        params = _tune(pipe, grid, Xtr, ytr, "r2" if task == "reg" else "accuracy")
        pipe.set_params(**params).fit(Xtr, ytr)
        if task == "reg":
            m = reg_metrics(yte, pipe.predict(Xte))
            m["Train_R2"] = r2_score(ytr, pipe.predict(Xtr))
        else:
            m = clf_metrics(yte, pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1])
            m["Train_Accuracy"] = accuracy_score(ytr, pipe.predict(Xtr))
        m.update(Model=name, Params=json.dumps({k.replace("model__", ""): v for k, v in params.items()}),
                 Seconds=round(time.time() - t0, 1))
        rows.append(m)
        fitted[name] = pipe
        log(f"  {task} | {name:<24} " + "  ".join(f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float)))
    res = pd.DataFrame(rows).set_index("Model")
    key = ["R2"] if task == "reg" else ["Accuracy", "F1"]
    best = res.sort_values(key, ascending=False).index[0]
    return res, fitted, best


def _raw_importance(pipe, X, y, scoring):
    idx = np.random.default_rng(RS).choice(len(X), min(6000, len(X)), replace=False)
    r = permutation_importance(pipe, X.iloc[idx], y.iloc[idx], n_repeats=3, random_state=RS,
                               scoring=scoring, n_jobs=1)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


# --------------------------------------------------------------------------- figures
def _plot_importance(imp: pd.Series, title, fname, xlabel):
    fig, ax = P.plt.subplots(figsize=(7, 4.6))
    s = imp.head(12)[::-1]
    ax.barh(s.index, s.values, color=C.SEQ_BLUE[4], height=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    return P.save(fig, fname)


def _plot_reg(yte, pred, name):
    fig, ax = P.plt.subplots(figsize=(6, 5.2))
    idx = np.random.default_rng(RS).choice(len(yte), min(5000, len(yte)), replace=False)
    ax.scatter(np.asarray(yte)[idx], pred[idx], s=8, alpha=0.35, color=C.BRAND_COLORS["Nykaa"], linewidths=0)
    lim = [0, max(np.max(yte), np.max(pred))]
    ax.plot(lim, lim, color=C.INK["secondary"], lw=1, ls="--", label="Perfect prediction")
    ax.xaxis.set_major_formatter(P.inr)
    ax.yaxis.set_major_formatter(P.inr)
    ax.set_xlabel("Actual revenue")
    ax.set_ylabel("Predicted revenue")
    ax.set_title(f"Revenue: actual vs predicted ({name}, test set)")
    ax.legend(loc="upper left")
    return P.save(fig, "model_revenue_actual_vs_pred.png")


def _plot_cm(cm, name):
    fig, ax = P.plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap=P.matplotlib.colors.LinearSegmentedColormap.from_list("b", C.SEQ_BLUE))
    labels = ["Loss", "Profit"]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else C.INK["primary"], fontsize=12)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
    ax.set_title(f"Profit/Loss confusion matrix\n({name}, test set)")
    return P.save(fig, "model_profit_confusion_matrix.png")


def _plot_comparison(reg: pd.DataFrame, clf: pd.DataFrame):
    fig, axes = P.plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, df, col, title in [(axes[0], reg, "R2", "Revenue regression - test R²"),
                               (axes[1], clf, "Accuracy", "Profit/Loss classification - test accuracy")]:
        s = df[col].sort_values()
        ax.barh(s.index, s.values, color=C.SEQ_BLUE[4], height=0.55)
        ax.axvline(0.95, color=C.INK["secondary"], lw=1, ls="--")
        ax.set_xlabel("dashed line = 0.95 project target")
        for y, v in enumerate(s.values):
            ax.text(min(v, 1) - 0.01, y, f"{v:.4f}", va="center", ha="right", color="white", fontsize=9)
        ax.set_xlim(min(0.5, s.min() - 0.05), 1.0)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    return P.save(fig, "model_comparison.png")


# --------------------------------------------------------------------------- main entry
def train_all(df: pd.DataFrame, log=print) -> dict:
    P.apply_style()
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    y_clf = df["Profit_Flag"]
    tr_idx, te_idx = train_test_split(df.index, test_size=C.TEST_SIZE, random_state=RS, stratify=y_clf)
    tr, te = df.loc[tr_idx], df.loc[te_idx]
    log(f"Train rows: {len(tr):,}  Test rows: {len(te):,}")

    # ---- Regression: Revenue
    reg_res, reg_fit, reg_best = run_task("reg", tr[REG_INPUTS], te[REG_INPUTS], tr["Revenue"], te["Revenue"],
                                          F.REG_NUMERIC, regression_candidates(), log)
    # ---- Classification: Profit_Flag (no ROI)
    clf_res, clf_fit, clf_best = run_task("clf", tr[CLF_INPUTS], te[CLF_INPUTS], tr["Profit_Flag"],
                                          te["Profit_Flag"], F.CLF_NUMERIC, classification_candidates(), log)
    log(f"Best regression model: {reg_best} | Best classification model: {clf_best}")

    # ---- Pre-launch benchmark (no ROI, no Revenue): what is predictable before results are known
    pre = {}
    pr = make_pipeline(F.PRELAUNCH_NUMERIC, HistGradientBoostingRegressor(max_iter=400, random_state=RS))
    pr.fit(tr[RAW_INPUTS], tr["Revenue"])
    pre["revenue_r2_without_roi"] = r2_score(te["Revenue"], pr.predict(te[RAW_INPUTS]))
    pc = make_pipeline(F.PRELAUNCH_NUMERIC, HistGradientBoostingClassifier(max_iter=400, random_state=RS))
    pc.fit(tr[RAW_INPUTS], tr["Profit_Flag"])
    pre["profit_accuracy_without_revenue_roi"] = accuracy_score(te["Profit_Flag"], pc.predict(te[RAW_INPUTS]))
    pre["profit_majority_baseline"] = float(max(y_clf.mean(), 1 - y_clf.mean()))
    log(f"Pre-launch benchmark: {pre}")

    # ---- Persist best models
    reg_model, clf_model = reg_fit[reg_best], clf_fit[clf_best]
    joblib.dump(reg_model, C.MODELS_DIR / "revenue_regressor.joblib", compress=("xz", 9))
    joblib.dump(clf_model, C.MODELS_DIR / "profit_classifier.joblib", compress=("xz", 9))

    # ---- Evaluation artefacts
    reg_pred = reg_model.predict(te[REG_INPUTS])
    clf_pred = clf_model.predict(te[CLF_INPUTS])
    cm = confusion_matrix(te["Profit_Flag"], clf_pred)
    reg_imp = _raw_importance(reg_model, te[REG_INPUTS], te["Revenue"], "r2")
    clf_imp = _raw_importance(clf_model, te[CLF_INPUTS], te["Profit_Flag"], "accuracy")
    _plot_reg(te["Revenue"], reg_pred, reg_best)
    _plot_cm(cm, clf_best)
    _plot_comparison(reg_res, clf_res)
    _plot_importance(reg_imp, f"What drives revenue predictions ({reg_best})", "model_revenue_importance.png",
                     "Drop in R² when the input is shuffled")
    _plot_importance(clf_imp, f"What drives profit/loss predictions ({clf_best})", "model_profit_importance.png",
                     "Drop in accuracy when the input is shuffled")

    reg_res.to_csv(C.REPORTS_DIR / "regression_model_comparison.csv")
    clf_res.to_csv(C.REPORTS_DIR / "classification_model_comparison.csv")

    meta = {
        "regression": {"best_model": reg_best, "inputs": REG_INPUTS, "test_metrics": reg_res.loc[reg_best].to_dict(),
                       "importance": reg_imp.round(5).to_dict()},
        "classification": {"best_model": clf_best, "inputs": CLF_INPUTS,
                           "test_metrics": clf_res.loc[clf_best].to_dict(),
                           "confusion_matrix": cm.tolist(), "importance": clf_imp.round(5).to_dict()},
        "prelaunch_benchmark": pre,
        "train_rows": len(tr), "test_rows": len(te),
        "channels": C.CHANNELS,
        "categories": {c: sorted(df[c].unique().tolist()) for c in C.CATEGORICAL_COLS},
        "input_ranges": {c: [float(df[c].min()), float(df[c].median()), float(df[c].max())]
                         for c in ["Duration", "Impressions", "Clicks", "Leads", "Conversions",
                                   "Acquisition_Cost", "Engagement_Score", "ROI", "Revenue"]},
    }
    with open(C.MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    return {"reg": reg_res, "clf": clf_res, "reg_best": reg_best, "clf_best": clf_best, "cm": cm,
            "reg_imp": reg_imp, "clf_imp": clf_imp, "pre": pre, "meta": meta}
