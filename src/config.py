"""Central configuration: paths, column groups and shared constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT                      # raw CSVs live in the project root (as delivered)
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_FILES = {
    "Nykaa": "nykaa_campaign_data_with_nulls.csv",
    "Purplle": "purplle_campaign_data_with_nulls.csv",
    "Tira": "tira_campaign_data_with_nulls.csv",
}
ID_PREFIX = {"Nykaa": "NY-CMP", "Purplle": "PU-CMP", "Tira": "TI-CMP"}
ID_START = 1000                      # IDs are sequential: <prefix>-<1000 + row index>

CLEANED_CSV = PROCESSED_DIR / "cleaned_campaign_data.csv"
FEATURED_CSV = PROCESSED_DIR / "feature_engineered_campaign_data.csv"
# Compact gzip copies (< 25 MB each) - these are what get uploaded to GitHub / Streamlit Cloud
CLEANED_GZ = PROCESSED_DIR / "cleaned_campaign_data.csv.gz"
FEATURED_GZ = PROCESSED_DIR / "feature_engineered_campaign_data.csv.gz"
DASHBOARD_GZ = PROCESSED_DIR / "dashboard_data.csv.gz"
DASHBOARD_COLS = ["Campaign_ID", "Brand", "Campaign_Type", "Target_Audience", "Customer_Segment", "Language",
                  "Channel_Used", "Date", "Impressions", "Clicks", "Leads", "Conversions", "Revenue",
                  "Acquisition_Cost", "Total_Cost", "ROI", "Engagement_Score", "Profit_Flag", "Profit_Status",
                  "Channel_Count"]

DATE_FORMAT = "%d-%m-%Y"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Total spend (Acquisition_Cost x Conversions) seen in training - predictions outside it are extrapolations
TRAIN_SPEND_RANGE = (50_000, 300_000)

CATEGORICAL_COLS =["Brand", "Campaign_Type", "Target_Audience", "Language", "Customer_Segment"]
FUNNEL_COLS = ["Impressions", "Clicks", "Leads", "Conversions"]
CHANNELS = ["Email", "Facebook", "Google", "Instagram", "WhatsApp", "YouTube"]

# Chart palette (validated reference palette - brand slots 1-3, profit/loss diverging pair)
BRAND_COLORS = {"Nykaa": "#2a78d6", "Purplle": "#eb6834", "Tira": "#1baf7a"}
PROFIT_COLORS = {"Profit": "#2a78d6", "Loss": "#e34948"}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
