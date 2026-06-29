# train_pipeline.py
import os
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib
from dataclasses import dataclass, asdict

# Configs
SPLIT_DATE_ID = 8800

@dataclass
class LGBMParameters:
    n_estimators: int = 5000
    learning_rate: float = 0.01
    num_leaves: int = 31
    max_depth: int = 5
    n_jobs: int = -1
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1

def run_training_pipeline():
    print("🚀 Initializing Training Pipeline via Polars...")
    if not os.path.exists("train.csv"):
        print("❌ Error: 'train.csv' missing.")
        return
        
    df = pl.read_csv("train.csv")
    df = df.rename({'market_forward_excess_returns': 'target'})
    
    # Lag Features
    df = df.with_columns([
        pl.col('target').shift(1).alias("lagged_market_forward_excess_returns"), 
        pl.col('risk_free_rate').shift(1).alias("lagged_risk_free_rate"),
        pl.col('forward_returns').shift(1).alias("lagged_forward_returns"),
    ]).drop(['forward_returns', 'risk_free_rate'])
    
    df = df.with_columns(pl.exclude('date_id').cast(pl.Float64, strict=False)).drop_nulls()
    
    # Feature Engineering
    df = df.with_columns([
        (pl.col("I2") - pl.col("I1")).alias("U1"),
        (pl.col("M11") / (pl.col("I2") + pl.col("I9") + pl.col("I7") + 1e-6)).alias("U2"), 
        (pl.col("S1") / (pl.col("S2") + 1e-6)).alias("S_Ratio"), 
        (pl.col("lagged_forward_returns") - pl.col("lagged_risk_free_rate")).alias("Lagged_Excess_Return")
    ])
    
    # Select Features
    features = [c for c in df.columns if c not in ['date_id', 'target']]
    
    # EWM Fill
    for col in features:
        df = df.with_columns(pl.col(col).fill_null(pl.col(col).ewm_mean(span=100)))
    df = df.drop_nulls()
    
    # Split
    train_df = df.filter(pl.col('date_id') < SPLIT_DATE_ID)
    val_df = df.filter(pl.col('date_id') >= SPLIT_DATE_ID)
    
    X_train = train_df.select(features).to_numpy()
    y_train = train_df.get_column('target').to_numpy()
    X_val = val_df.select(features).to_numpy()
    y_val = val_df.get_column('target').to_numpy()
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print(f"Training Samples: {X_train_scaled.shape[0]}, Validation Samples: {X_val_scaled.shape[0]}")
    
    # Fit Model
    params = LGBMParameters()
    model = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42, verbose=-1, **asdict(params))
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], callbacks=[lgb.early_stopping(200, verbose=False)])
    
    # Export
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(features, 'features_list.pkl')
    print("📦 Export Complete! saved: 'model.pkl', 'scaler.pkl', 'features_list.pkl'")

if __name__ == "__main__":
    run_training_pipeline()
