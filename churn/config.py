from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_FILENAME = "churn_pipeline.joblib"
METADATA_FILENAME = "model_metadata.json"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
METADATA_PATH = MODEL_DIR / METADATA_FILENAME

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

TARGET = "Churn"
ID_COL = "customerID"

# Column groups. Kept explicit on purpose: pandas 3 reads strings as a new "str"
# dtype, so picking columns by dtype is not reliable across versions.
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_NUMERIC_FEATURES = ["SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + BINARY_NUMERIC_FEATURES + CATEGORICAL_FEATURES

# These columns carry a third level ("No internet service") that just repeats
# InternetService == "No". MultipleLines does the same with PhoneService.
NO_INTERNET_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
