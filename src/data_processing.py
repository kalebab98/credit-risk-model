# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Feature Engineering - Extract Time Features and Log Transform
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Convert to datetime
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
        
        # Extract time features
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year
        
        # Cap Amount and Value at 1st and 99th percentile to limit outliers
        df['Amount_capped'] = df['Amount'].clip(lower=df['Amount'].quantile(0.01),
                                               upper=df['Amount'].quantile(0.99))
        df['Value_capped'] = df['Value'].clip(lower=df['Value'].quantile(0.01),
                                             upper=df['Value'].quantile(0.99))
        
        # Log transform (use abs for Amount to avoid log of negative)
        df['log_amount'] = np.log1p(df['Amount_capped'].abs())
        df['log_value'] = np.log1p(df['Value_capped'])
        
        return df

# 2. Aggregate Features per Customers
class CustomerAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Group by CustomerId and aggregate
        agg = df.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'std', 'count'],
            'Value': ['sum', 'mean', 'std']
        })
        
        agg.columns = [
            'amount_sum', 'amount_mean', 'amount_std', 'amount_count',
            'value_sum', 'value_mean', 'value_std'
        ]
        agg.reset_index(inplace=True)
        
        # Merge aggregate features back to original df
        df = df.merge(agg, on='CustomerId', how='left')
        return df

# 3. Pipeline Builder
def build_preprocessing_pipeline():
    # Categorical features for encoding
    categorical_features = ['ProductCategory', 'ChannelId', 'ProviderId', 'PricingStrategy']
    
    # Numerical features to scale (including engineered + aggregated)
    numerical_features = [
        'log_amount', 'log_value',
        'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
        'amount_sum', 'amount_mean', 'amount_std', 'amount_count',
        'value_sum', 'value_mean', 'value_std'
    ]
    
    # Numerical pipeline: impute missing with mean + standardize
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: impute missing with most frequent + one-hot encode
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    # Full pipeline chaining feature engineering, aggregation, and preprocessing
    full_pipeline = Pipeline([
        ('feature_engineer', FeatureEngineer()),
        ('customer_agg', CustomerAggregator()),
        ('preprocessing', preprocessor)
    ])
    
    return full_pipeline
