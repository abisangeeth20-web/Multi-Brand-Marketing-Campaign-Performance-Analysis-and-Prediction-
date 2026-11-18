# Multi-Brand Marketing Campaign Performance Analysis & Prediction

An end-to-end machine learning pipeline for Nykaa, Purplle and Tira marketing campaigns. It covers CSV ingestion, cleaning, feature engineering (including multi-label channel encoding and a Profit/Loss flag), EDA, revenue regression, profit/loss classification and a Streamlit app.

**Domain:** Marketing Analytics & Business Intelligence · **Stack:** Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit

---

## Results at a glance

| Task | Best model | Test metric | Target |
|---|---|---|---|
| Revenue regression | Random Forest | R² = 0.9990 · RMSE ₹15,335 · MAE ₹4,931 | R² ≥ 0.95 ✅ |
| Profit / Loss classification (ROI excluded) | Logistic Regression | Accuracy 0.9996 · Precision 0.9998 · Recall 0.9997 · F1 0.9998 | Accuracy ≥ 0.95 ✅ |

Exact figures for every model are in [reports/Model_Evaluation_Report.md](reports/Model_Evaluation_Report.md). Insights and recommendations are in [reports/EDA_Report.md](reports/EDA_Report.md).

## Project structure

```
├── nykaa_campaign_data_with_nulls.csv        raw data (as delivered)
├── purplle_campaign_data_with_nulls.csv
├── tira_campaign_data_with_nulls.csv
├── run_pipeline.py                           one command: clean → features → EDA → train → evaluate
├── app.py                                    Streamlit app (predict · dashboard · model performance)
├── requirements.txt
├── src/
│   ├── config.py                             paths, column groups, palette
│   ├── preprocessing.py                      loading + cleaning (ID reconstruction, identity-based recovery, imputation)
│   ├── features.py                           Profit_Flag, multi-label channel encoding, derived features, sklearn transformer
│   ├── eda.py                                EDA figures + summary tables
│   ├── train.py                              model building, tuning, evaluation, persistence
│   └── plotting.py                           shared chart style
├── notebooks/
│   ├── Marketing_Campaign_Analysis.ipynb     executed walk-through notebook
│   └── build_notebook.py                     regenerates the notebook
├── data/processed/
│   ├── cleaned_campaign_data.csv             cleaned dataset
│   └── feature_engineered_campaign_data.csv  + encoded channels, derived features, Profit_Flag
├── models/
│   ├── revenue_regressor.joblib              full sklearn pipeline (feature engineering + preprocessing + model)
│   ├── profit_classifier.joblib
│   └── model_metadata.json                   inputs, metrics, categories, ranges
└── reports/
    ├── EDA_Report.md                         insights & recommendations
    ├── Model_Evaluation_Report.md            metrics, model comparison, leakage analysis
    ├── regression_model_comparison.csv / classification_model_comparison.csv
    ├── eda_tables/*.csv
    └── figures/*.png
```

## How to run

```bash
pip install -r requirements-dev.txt   # requirements.txt alone is enough for the app
python run_pipeline.py          # ~10 min: writes processed data, figures, models and comparison tables
streamlit run app.py            # opens the app at http://localhost:8501
```

To rerun the notebook:
`python -m nbconvert --to notebook --execute --inplace notebooks/Marketing_Campaign_Analysis.ipynb`

## Deploy (permanent public link) — Streamlit Community Cloud

1. Push this folder to a GitHub repository. The two uncompressed CSVs in `data/processed/` are larger than 25 MB, so `.gitignore` excludes them; the `.csv.gz` copies are committed instead.
2. Sign in at https://share.streamlit.io with GitHub and choose **Create app**. Select the repo and branch, and set the main file to `app.py`.
3. Under **Advanced settings**, choose **Python 3.13**. `requirements.txt` pins the exact library versions the models were trained with.
4. Click **Deploy**. The app gets a permanent `https://<name>.streamlit.app` URL.

Free apps sleep after a period with no visitors. Opening the link then shows a **"Yes, get this app back up"** button, and the app wakes in about 30–60 seconds; the URL itself never changes.

## Approach

### 1. Data collection
The three brand CSVs are loaded into one DataFrame (166,665 rows × 16 columns) with a `Brand` column added.

### 2. Data preprocessing
Profiling showed two exact identities in the data:

* `ROI = (Revenue − Acquisition_Cost × Conversions) / (Acquisition_Cost × Conversions)`, so `Acquisition_Cost` is cost **per conversion**.
* `Engagement_Score = (Clicks + Leads + Conversions) / Impressions × 100`

These identities drive the cleaning:

* Missing `Campaign_ID`s are reconstructed (IDs are sequential). Duplicates are checked and removed (none remain).
* **ROI is validated and corrected:** 25 rows broke the identity and were recomputed.
* 57,624 missing cells are **recovered exactly** by solving an identity whenever only one of its members is missing.
* 1,435 rows (0.86%) whose financial values can't be made consistent are dropped.
* Remaining gaps are filled as follows:
  * funnel counts: brand-median conversion ratios
  * Duration: brand median
  * categoricals: brand mode
  * channels: `Unknown`
  * dates: sampled from the brand's observed dates
* Types are converted (dates parsed, counts as integers) and text is standardised. Channel lists are sorted and de-duplicated.

### 3. Feature engineering
* `Profit_Flag = 1 if ROI > 0 else 0` (76.0% profit / 24.0% loss)
* **Multi-label encoding** of `Channel_Used` (`MultiLabelBinarizer`) into `Ch_Email`, `Ch_Facebook`, `Ch_Google`, `Ch_Instagram`, `Ch_WhatsApp`, `Ch_YouTube`, plus `Channel_Count`
* `Total_Cost`, `CTR`, `Lead_Rate`, `Conversion_Rate`, `Cost_per_Click`, `Revenue_per_Conversion`, `Month`, `Day_of_Week`
* One-hot encoding of categoricals and standard scaling of numerics, inside a `ColumnTransformer`

All feature engineering is wrapped in an sklearn transformer (`CampaignFeatureBuilder`) and saved inside each model pipeline. The Streamlit app therefore applies **exactly** the same steps to user input.

### 4. EDA
Brand, campaign-type, channel, audience and language performance; top and bottom campaigns; spend–clicks–revenue–ROI relationships; cost-per-conversion bands. See [reports/EDA_Report.md](reports/EDA_Report.md).

### 5–6. Modelling & evaluation
* 80/20 stratified split (`random_state=42`)
* **Regression (Revenue):** Linear Regression, Decision Tree, Random Forest, Hist Gradient Boosting
* **Classification (Profit_Flag):** Logistic Regression, Decision Tree, Random Forest, Hist Gradient Boosting
* Hyper-parameters tuned with `RandomizedSearchCV` (3-fold CV). The best model is chosen on unseen test data.
* Metrics:
  * regression: R², RMSE, MAE, MSE
  * classification: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix
* **Leakage control:** ROI is never a classifier input. No `Revenue / Total_Cost` ratio (which equals 1 + ROI) is created.

### 7. Insights
See the EDA and model-evaluation reports. In short, **cost per conversion** is the main profitability lever. Revenue scales with conversions, while channel mix, audience and language barely matter.

### 8. Streamlit app
* **Predict:** enter a campaign profile, channels, funnel and costs. The app shows predicted revenue, the Profit/Loss prediction with its probability, a funnel chart, spend vs revenue, and a break-even check.
* The app warns when the planned total spend falls outside the training range (₹50K – ₹3L per campaign), because tree models don't extrapolate.
* **Dashboard:** interactive Plotly EDA with brand, campaign-type, audience and date filters.
* **Model performance:** metrics tables, comparison charts, confusion matrix and feature importance.
