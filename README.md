# Telco customer churn: model, pipeline and inference API

Predicts whether a telecom customer will churn, using the public
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset
(7,043 customers, 26.5% churn; the CSV is included in `data/raw/`).

The shipped model is a tuned XGBoost classifier inside one scikit-learn `Pipeline` (cleaning,
imputation, encoding, model). It takes the raw customer JSON and returns a churn probability and a
0/1 decision.

| Shipped model, held-out test set (1,409 customers) | |
|---|---|
| ROC-AUC | **0.850** (95% bootstrap CI 0.828 - 0.870) |
| PR-AUC | **0.670** (0.617 - 0.718), no-skill baseline is 0.265 |
| F1 at the chosen threshold of 0.34 | **0.634** (0.595 - 0.670), precision 0.556, recall 0.738 |

## Quick start

Needs Python 3.10 - 3.14 (I developed on 3.13 and ran the install and test suite in a clean venv on
3.10, 3.11, 3.12, 3.13 and 3.14). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python predict.py examples/sample_customer.json
```

```json
{
  "customerID": "7590-VHVEG",
  "churn_prediction": 1,
  "churn_label": "Yes",
  "churn_probability": 0.7032,
  "decision_threshold": 0.34
}
```

The trained model is already in `models/`, so no training is needed. Versions in `requirements.txt`
are pinned because the model is a joblib pickle, which is only guaranteed to load with the
scikit-learn / xgboost it was saved with. With other versions `predict.py` warns, and if the model then
can't score it stops with a clear message; `python train.py` rebuilds it.

**Payload.** Same shape as a row of the Kaggle CSV (`examples/` has three samples). `customerID` is
optional, extra keys are ignored, and `TotalCharges` may be a string, number, blank or null. A missing
field, unknown category or absurd value is rejected with the field name (exit code 2 / HTTP 422), never
scored silently.

```bash
cat customer.json | python predict.py -            # stdin
python predict.py --json '{"gender": "Female", ...}'   # bash; in Windows PowerShell use the file form
```

**HTTP API** (FastAPI, same code path as the CLI):

```bash
uvicorn churn.api:app --port 8000
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @examples/sample_customer.json
```
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -InFile examples/sample_customer.json
```

`GET /health` shows the loaded model; interactive docs are at `/docs`.

**Using the joblib directly.** `models/churn_pipeline.joblib` is a plain `Pipeline` that accepts the raw
columns as a DataFrame: `predict_proba` gives the churn probability. Its `.predict()` cuts at 0.5, while
the tuned 0.34 threshold lives in `models/model_metadata.json` and is applied by `predict.py` and the
API. Load it from the repo root, because the pickle refers to `churn.preprocessing.TelcoCleaner`.

**Re-train and test.**

```bash
python train.py            # about 2 minutes on 8 cores; rewrites models/ and reports/
python train.py --quick    # smaller searches, writes to models_quick/ and reports_quick/ instead

pip install -r requirements-dev.txt
pytest                     # 61 fast tests
pytest -m slow             # only the end-to-end quick-training test
```

## Model comparison

5-fold stratified CV on the 80% training split (mean +/- sample std across folds), then one evaluation
on the 20% test split. Every model's threshold is tuned on its own out-of-fold predictions (the dummy
uses 0.5). This is `reports/model_comparison.md`, written by `train.py`.

| Model | CV ROC-AUC | CV PR-AUC | Test ROC-AUC | Test PR-AUC | Test F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|---|---|
| dummy_majority | 0.500 +/- 0.000 | 0.265 +/- 0.000 | 0.500 | 0.265 | 0.000 | 0.000 | 0.000 | 0.735 |
| logreg_baseline | 0.846 +/- 0.014 | 0.661 +/- 0.021 | 0.842 | 0.636 | 0.619 | 0.527 | 0.749 | 0.755 |
| logreg_tuned | 0.846 +/- 0.014 | 0.662 +/- 0.021 | 0.842 | 0.632 | 0.618 | 0.534 | 0.733 | 0.759 |
| random_forest_tuned | 0.848 +/- 0.012 | 0.666 +/- 0.024 | 0.846 | 0.660 | 0.637 | 0.536 | 0.786 | 0.762 |
| xgboost_default | 0.823 +/- 0.010 | 0.624 +/- 0.019 | 0.815 | 0.601 | 0.605 | 0.505 | 0.754 | 0.739 |
| **xgboost_tuned** (shipped) | 0.850 +/- 0.013 | 0.671 +/- 0.023 | 0.850 | 0.670 | 0.634 | 0.556 | 0.738 | 0.774 |

- **Baseline:** logistic regression, default regularisation, scaled inputs.
- **Tree ensembles:** random forest and XGBoost. XGBoost is the primary model: library defaults first,
  then tuned with `RandomizedSearchCV` (60 draws, 5-fold CV, scored on PR-AUC). I gave the logistic
  regression (grid over `C`) and the forest (15 draws) some tuning too. Budgets are small on purpose.
- **Selection rule**, coded in `train.py`: highest mean CV PR-AUC among the non-dummy models. The CV
  columns of the tuned models are the search scores on the same folds, so they are slightly optimistic;
  the test columns are the unbiased comparison.
- Shipped model's confusion matrix on the test set (threshold 0.34): 815 true negatives, 220 false
  positives, 98 missed churners, 276 caught. Plots are in `reports/figures/` (ROC/PR, confusion matrix,
  calibration, permutation importance).

**Reading the table honestly.** The three tuned models are not meaningfully different: their CV scores
are within about half a fold standard deviation of each other, and the test-set PR-AUC interval alone
is about +/- 0.05. XGBoost wins on the rule I set up front; if interpretability mattered more I would
ship the logistic regression at almost no cost. The clear result is that tuning XGBoost matters (CV
PR-AUC 0.624 to 0.671). I also tried three hand-made features (number of add-ons, average monthly
spend, gap between current and average spend) in repeated CV: no improvement, so they are not in.

## How it works

**Data issues** (plots in `notebooks/01_eda.ipynb`):
- `TotalCharges` is text because 11 rows hold a single space. All have `tenure == 0` (signed up, not
  billed yet) and none churned, so "missing" means "zero so far". I fill it with
  `tenure * MonthlyCharges` (0 for those rows; they correlate at 0.9996 elsewhere) rather than a median,
  which would invent customers who paid ~$1,400 in month zero. "No phone/internet service" just repeats
  another column, so it is folded into "No".
- Churn concentrates in short-tenure, month-to-month customers: 43% on month-to-month contracts vs 3% on
  two-year, 53% in the first six months vs under 10% after four years, fibre 42% vs DSL 19%, electronic
  check 45% vs 15-19% for other payment methods. Permutation importance agrees on tenure, contract and
  internet service; payment method and the support add-ons look strong alone but add little once those
  are known.

**Pipeline.** `TelcoCleaner` (types, blanks, merged levels; nothing learned) -> `ColumnTransformer`
(imputation, scaling where the model needs it, one-hot) -> model, all in one `Pipeline`, so the saved
artifact accepts raw input and there is no separate preprocessing code to drift from training.

**Avoiding leakage.**
- The stratified 80/20 split happens before anything is fitted. Nothing the model learns (imputer
  medians, scaler, encoder levels, hyper-parameters, model choice, threshold) uses the test rows; the EDA
  looks at all rows but only motivated fixed, row-wise cleaning rules.
- Imputers, scalers and encoders are fitted inside the pipeline, so in cross-validation they only see the
  training folds. The threshold comes from out-of-fold predictions on the training split.
- Test predictions are computed for every candidate and reported, but no choice depends on them.
- `customerID` is dropped, and there is no cancellation date or reason column.

**Class imbalance.** With 27% churners I stratify the split and folds, judge models on ROC-AUC and PR-AUC,
and choose the decision threshold explicitly instead of assuming 0.5. As a check I also tried class
weighting (`scale_pos_weight`) on the same folds and hyper-parameters:

| Shipped XGBoost, 5-fold CV on the training split | PR-AUC | Brier (lower = better calibrated) | F1 at best threshold | Best threshold |
|---|---|---|---|---|
| No class weights (shipped) | 0.671 | **0.133** | 0.636 | 0.34 |
| Class weighted | 0.668 | 0.161 | 0.633 | 0.58 |

Weighting doesn't improve ranking or tuned-threshold F1; it only moves the threshold and inflates the
probabilities (`imbalance_check` in `reports/metrics.json`). The API returns a probability, so I kept the
model unweighted and let the threshold absorb the imbalance.

**Threshold.** Best F1 on out-of-fold predictions, on a 0.01 grid with the F1 curve lightly smoothed so an
isolated spike can't win (here it equals the plain argmax, and F1 stays within 0.01 of its best from about
0.30 to 0.40). F1 treats a missed churner and a wasted offer as equally bad, which a real business would
not; see the last section.

**Why accuracy isn't enough.** 73.5% of customers don't churn, so a model that says "stays" for everyone is
73.5% accurate and catches zero churners (the `dummy_majority` row). Accuracy also weights a missed churner
and a false alarm equally and depends on an arbitrary cut-off. Instead I use:
- **ROC-AUC**: how well churners are ranked above non-churners, independent of the threshold.
- **PR-AUC (average precision)**: focused on the churn class and sensitive to how well churners are ranked
  near the top, which matters at 27% prevalence. The searches optimise this one.
- **Precision, recall, F1** at the chosen threshold: what a retention team would actually see.
- **Brier score / calibration**, because the API exposes a probability.

## Limitations

- One snapshot with no dates, so the split is random rather than "train on the past, test on the
  future", and drift can't be assessed. Also, for churned customers `tenure` and `TotalCharges` stop at
  the exit date, while active customers are still counting, so these features carry some outcome
  information that a live scoring system would not have in the same form.
- The test set has about 1,400 customers: roughly +/- 0.02 of noise on ROC-AUC and +/- 0.05 on PR-AUC.
- The threshold maximises F1 rather than real costs, and numbers can move in the third decimal across
  platforms (the untuned XGBoost row is the most sensitive to library versions).
- The shipped model is trained on the 80% split, i.e. exactly the artifact the metrics describe.

## If I had two more days, the top 3 things I would do

1. **Monitoring and a retraining loop (~7h).** The biggest gap between this and something that keeps
   working after launch. Log each request and score, run scheduled PSI/KS drift checks on the inputs
   and the score distribution, and track calibration and PR-AUC as labels arrive. Retrain on a schedule
   and promote a new model only if it beats the current one on a time-based backtest (this needs a
   signup or snapshot date, which this dataset lacks).
2. **Deployable and reproducible (~5h).** A Dockerfile for the API and CI running `pytest` plus
   `python train.py --quick` on every push. `model_metadata.json` already records data hash and library
   versions; I would add the git commit, and bake the tuned threshold into the model artifact so it
   can't be separated from it (today it sits in a sidecar JSON).
3. **Make the decision actionable (~4h).** Replace the F1 threshold with an expected-profit one (offer
   cost and customer lifetime value as config), and return the top SHAP reasons with each prediction so
   a retention agent knows what to talk about.

Also worth doing: a batch endpoint and a load test, repeated or nested CV with a paired test (the
current candidates are within noise), and uplift modelling if campaign history existed.

## Repository layout

```
train.py, predict.py     training entry point, CLI inference
churn/                   config, data, preprocessing, modeling, evaluation, schemas, inference, api
models/                  churn_pipeline.joblib, model_metadata.json
reports/                 metrics.json, model_comparison.md, figures/
notebooks/01_eda.ipynb   exploratory analysis (already executed)
examples/, tests/        sample payloads, pytest suite
```
