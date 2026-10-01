from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

YesNo = Literal["Yes", "No"]


class CustomerProfile(BaseModel):
    """One customer, in the same shape as a row of the raw Kaggle CSV.

    Unknown keys are ignored (so a stray "Churn" label can't sneak in), but a
    missing feature or an unseen category is rejected instead of being silently
    scored, which is what a one-hot encoder would otherwise do.
    """

    # coerce_numbers_to_str: a numeric customerID (12345) is fine, it is only echoed back
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True, coerce_numbers_to_str=True)

    customerID: Optional[str] = None
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    # generous sanity bounds (training data: tenure 0-72, charges under ~9k); they keep absurd
    # values from overflowing the TotalCharges fill further down
    tenure: int = Field(ge=0, le=1200, description="months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=100_000, allow_inf_nan=False)
    # The CSV ships this as a string and leaves it blank for brand-new customers,
    # so accept a number, a numeric string, a blank string or null.
    TotalCharges: Optional[float] = Field(default=None, ge=0, le=10_000_000, allow_inf_nan=False)

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def blank_total_is_missing(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value


class PredictionResponse(BaseModel):
    customerID: Optional[str] = None
    churn_prediction: int = Field(description="1 = predicted to churn, 0 = predicted to stay")
    churn_label: Literal["Yes", "No"]
    churn_probability: float = Field(description="model's churn probability, between 0 and 1")
    decision_threshold: float = Field(description="probability cut-off used for churn_prediction")
