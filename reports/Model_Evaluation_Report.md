# Model Evaluation Report

**Data:** 165,230 cleaned campaigns, split 80/20 with stratification on Profit_Flag (`random_state=42`).
**Train:** 132,184 rows · **Test (unseen):** 33,046 rows.
**Tuning:** `RandomizedSearchCV`, 3-fold CV on a 30,000-row training subsample. The best parameters are then refit on the full training set.

---

## 1. Revenue regression

**Inputs:** brand, campaign type, target audience, language, customer segment, channels (multi-label), date, duration, impressions, clicks, leads, conversions, acquisition cost, engagement score and ROI. Engineered features are built inside the pipeline: Total_Cost, CTR, Lead_Rate, Conversion_Rate, Cost_per_Click, Channel_Count, the channel flags, Month and Day_of_Week.

| Model | Test R² | Train R² | RMSE | MAE | MSE | Tuned parameters |
|---|---|---|---|---|---|---|
| **Random Forest** ✅ | **0.9990** | 0.9997 | **₹15,335** | **₹4,931** | 2.35 × 10⁸ | 30 trees, max_depth 20, min_samples_leaf 3, max_features 0.8 |
| Decision Tree | 0.9976 | 0.9993 | ₹23,940 | ₹9,302 | 5.73 × 10⁸ | max_depth None, min_samples_leaf 5 |
| Hist Gradient Boosting | 0.9970 | 0.9988 | ₹26,628 | ₹9,491 | 7.09 × 10⁸ | 600 iterations, lr 0.05, 127 leaves |
| Linear Regression | 0.8626 | 0.8633 | ₹180,803 | ₹122,626 | 3.27 × 10¹⁰ | — |

* **Selected model: Random Forest.** R² = 0.9990, well above the 0.95 target. For scale, average revenue is about ₹5.1 L and the MAE is ₹4,931.
* The train/test gap is small (0.9997 vs 0.9990), so the model isn't overfitting.
* Linear Regression scores lower because revenue is a *product* of inputs: `Revenue = Acquisition_Cost × Conversions × (1 + ROI)`. A linear model can't represent the interaction, while tree ensembles learn it.
* Forest size is capped at 30 trees, so the saved model (xz-compressed) is 19 MB. That fits GitHub's 25 MB web-upload limit, which Streamlit Cloud deployment needs. A 150-tree forest scored R² 0.9991, so the smaller model costs only 0.0001.
* **What drives the prediction** (permutation importance = drop in R² when an input is shuffled): Conversions 0.95, ROI 0.86, Acquisition_Cost 0.40, and all other inputs below 0.002.

## 2. Profit / Loss classification

**Target:** `Profit_Flag = 1 if ROI > 0`, giving 76.0% profit and 24.0% loss.
**Inputs:** the same campaign inputs plus Revenue. **ROI is excluded**, and so is any ratio equivalent to it (Revenue / Total_Cost = 1 + ROI).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train accuracy | Tuned parameters |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** ✅ | **0.9996** | **0.9998** | **0.9997** | **0.9998** | 1.0000 | 0.9997 | C = 1000 |
| Hist Gradient Boosting | 0.9988 | 0.9991 | 0.9993 | 0.9992 | 1.0000 | 0.9998 | 300 iterations, lr 0.05, 31 leaves |
| Random Forest | 0.9988 | 0.9990 | 0.9993 | 0.9992 | 1.0000 | 1.0000 | 100 trees, max_depth 16, max_features 0.5 |
| Decision Tree | 0.9985 | 0.9989 | 0.9991 | 0.9990 | 0.9983 | 0.9992 | entropy, min_samples_leaf 5 |

**Confusion matrix (Logistic Regression, test set):**

|  | Predicted Loss | Predicted Profit |
|---|---|---|
| **Actual Loss** | 7,920 | 4 |
| **Actual Profit** | 8 | 25,114 |

* **Selected model: Logistic Regression.** Accuracy = 0.9996, well above the 0.95 target. It misclassified 12 of 33,046 test campaigns.
* The simplest model wins because the profit boundary, `Revenue > Acquisition_Cost × Conversions`, is linear in the standardised Revenue and spend features. The model is also small (< 1 MB), fast, and has interpretable coefficients.
* **Top drivers** (permutation importance = drop in accuracy when shuffled): Acquisition_Cost 0.34, Revenue 0.34, Conversions 0.16, and all other inputs below 0.003.

## 3. Data leakage and realism analysis

The spec requires ROI to be excluded from classification, and it is. The table below also shows how much each model relies on the financial outcome columns:

| Scenario | Inputs | Test result |
|---|---|---|
| Revenue regression (deployed) | campaign attributes + funnel + cost + **ROI** | R² = **0.9990** |
| Revenue regression, **without ROI** | campaign attributes + funnel + cost | R² = 0.770 |
| Profit classification (deployed) | campaign attributes + funnel + cost + **Revenue** (no ROI) | Accuracy = **0.9996** |
| Profit classification, **pre-launch** (no Revenue, no ROI) | campaign attributes + funnel + cost | Accuracy = 0.909 |
| Majority-class baseline | — | Accuracy = 0.760 |

**How to read this:**

* Revenue per conversion varies randomly between ₹200 and ₹800 and is uncorrelated with every campaign attribute. Without ROI, about 23% of revenue variance is therefore unforecastable (R² ceiling ≈ 0.77). With ROI, the model reconstructs revenue almost exactly.
* Even **before a campaign runs**, knowing only its planned funnel and cost per conversion predicts profit vs loss correctly **91% of the time**, against 76% for always guessing "profit". Cost per conversion carries most of that signal.
* The deployed models meet the project targets. The pre-launch figures are the honest estimate of what can be forecast at planning time.

## 4. Key factors affecting profitability (from the models)

1. **Acquisition cost per conversion** is the strongest single driver in both the classifier and the EDA. Below ₹200 campaigns were 100% profitable, and above ₹800 they were 0%.
2. **Conversion volume** drives revenue. Leads and clicks matter only through the conversions they produce.
3. **Revenue per conversion** (order value) decides profit together with cost.
4. Brand, campaign type, channel mix, audience, language, duration and date carry almost no predictive weight (importance < 0.003). This matches the EDA, where all groups fall within about 2 pp of each other.

## 5. Artefacts

| File | Contents |
|---|---|
| `models/revenue_regressor.joblib` | Full pipeline: feature builder → one-hot + scaling → Random Forest |
| `models/profit_classifier.joblib` | Full pipeline: feature builder → one-hot + scaling → Logistic Regression |
| `models/model_metadata.json` | Inputs, metrics, confusion matrix, importances, categories, input ranges |
| `reports/regression_model_comparison.csv` / `classification_model_comparison.csv` | All model results |
| `reports/figures/model_*.png` | Comparison chart, actual vs predicted, confusion matrix, importances |
