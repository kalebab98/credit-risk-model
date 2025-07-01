from pydantic import BaseModel

class PredictionRequest(BaseModel):
    # Use features from your processed dataset
    log_amount: float
    log_value: float
    TransactionHour: float
    TransactionDay: float
    TransactionMonth: float
    TransactionYear: float
    amount_sum: float
    amount_mean: float
    amount_std: float
    amount_count: float
    # Add encoded categorical features
    ProviderId_ProviderId_2: float
    ProviderId_ProviderId_3: float
    ProviderId_ProviderId_4: float
    ProviderId_ProviderId_5: float
    ProviderId_ProviderId_6: float
    PricingStrategy_0: float
    PricingStrategy_1: float
    PricingStrategy_2: float
    PricingStrategy_4: float
