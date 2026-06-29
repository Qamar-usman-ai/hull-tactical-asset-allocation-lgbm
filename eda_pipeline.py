# eda_pipeline.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (15, 5), 'font.size': 11})

def run_standalone_eda():
    print("⏳ Loading Dataset for Comprehensive EDA...")
    # Read the data from your local folder structure
    if not os.path.exists("train.csv"):
        print("❌ Error: 'train.csv' not found. Make sure it is in the same directory.")
        return
        
    df = pd.read_csv("train.csv")
    print(f"✅ Data Loaded. Shape: {df.shape[0]} Rows, {df.shape[1]} Columns")
    
    # Create directory for saving charts
    os.makedirs("./plots", exist_ok=True)
    
    # 1. Target Diagnostics
    target = 'market_forward_excess_returns'
    if target in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        sns.histplot(df[target].dropna(), kde=True, ax=axes[0], color='#2b5c8f', bins=100)
        axes[0].set_title("Distribution of Market Excess Returns")
        
        axes[1].plot(df['date_id'], df[target], color='#e67e22', alpha=0.4, linewidth=0.5)
        axes[1].set_title("Target Values Timeline (Chronological)")
        plt.tight_layout()
        plt.savefig("./plots/target_diagnostics.png")
        print("💾 Saved: ./plots/target_diagnostics.png")
        plt.close()

    # 2. Market Regime Volatility
    if target in df.columns:
        df['vol_30'] = df[target].rolling(window=30).std()
        df['vol_90'] = df[target].rolling(window=90).std()
        
        plt.figure(figsize=(15, 5))
        plt.plot(df['date_id'], df['vol_30'], color='#9b59b6', alpha=0.7, label='30-Day Risk')
        plt.plot(df['date_id'], df['vol_90'], color='#2c3e50', alpha=0.9, label='90-Day Trend')
        plt.title("Systemic Volatility Dynamics (Market Regimes)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("./plots/market_volatility_regimes.png")
        print("💾 Saved: ./plots/market_volatility_regimes.png")
        plt.close()

    # 3. Features Group Snapshot
    feature_cols = [c for c in df.columns if c not in ['date_id', 'forward_returns', 'risk_free_rate', target]][:12]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    for idx, col in enumerate(feature_cols):
        sns.histplot(df[col].dropna(), ax=axes[idx], color='seagreen', bins=30)
        axes[idx].set_title(f"Dist: {col}")
    plt.tight_layout()
    plt.savefig("./plots/feature_distributions.png")
    print("💾 Saved: ./plots/feature_distributions.png")
    plt.close()
    
    print("🏁 EDA Pipeline finished running successfully!")

if __name__ == "__main__":
    run_standalone_eda()
