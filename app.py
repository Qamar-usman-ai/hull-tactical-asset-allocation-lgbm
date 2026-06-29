# app.py
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Hull Tactical Dashboard", page_icon="🥉", layout="wide")
sns.set_theme(style="whitegrid")

st.title("🥉 Hull Tactical: Causal LGBM Asset Allocation Dashboard")
st.markdown("### **Developed by: Qamar Usman (Machine Learning Engineer)**")
st.write("Upload a tracking test dataset to run the LightGBM production pipeline and generate volatility-controlled trading signals.")

st.divider()

# Load Artifacts Safely
@st.cache_resource
def load_production_artifacts():
    if os.path.exists("model.pkl") and os.path.exists("scaler.pkl") and os.path.exists("features_list.pkl"):
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        features = joblib.load("features_list.pkl")
        return model, scaler, features
    return None, None, None

model, scaler, trained_features = load_production_artifacts()

if model is None:
    st.error("❌ Production core files ('model.pkl', 'scaler.pkl', 'features_list.pkl') are missing from the root folder.")
else:
    st.sidebar.header("📁 Step 1: Upload Test Data")
    uploaded_file = st.sidebar.file_uploader("Upload test.csv file", type=["csv"])
    
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
        st.success(f"Successfully uploaded {test_df.shape[0]} trading entries!")
        
        # Data Pipeline logic inside Streamlit
        with st.spinner("Processing tactical transformations..."):
            proc_df = test_df.copy()
            
            # Recreate structural elements missing from single rows
            if 'target' not in proc_df.columns:
                proc_df['target'] = 0.0
                
            # Alignment interactions
            proc_df["U1"] = proc_df["I2"] - proc_df["I1"]
            proc_df["U2"] = proc_df["M11"] / (proc_df["I2"] + proc_df["I9"] + proc_df["I7"] + 1e-6)
            proc_df["S_Ratio"] = proc_df["S1"] / (proc_df["S2"] + 1e-6)
            
            # Map lags safely from incoming streaming names
            if 'lagged_forward_returns' in proc_df.columns and 'lagged_risk_free_rate' in proc_df.columns:
                proc_df["Lagged_Excess_Return"] = proc_df["lagged_forward_returns"] - proc_df["lagged_risk_free_rate"]
            else:
                proc_df["Lagged_Excess_Return"] = 0.0
            
            # Ensure all features match trained setup exactly
            for col in trained_features:
                if col not in proc_df.columns:
                    proc_df[col] = np.nan
            
            # Fill missing entries & scale
            X_eval = proc_df[trained_features].fillna(0.0).to_numpy()
            X_eval_scaled = scaler.transform(X_eval)
            
            # Inference Execution
            raw_preds = model.predict(X_eval_scaled)
            # Betting engine conversion
            final_signals = np.clip(raw_preds * 150.0 + 1, 0.0, 2.0)
            
            proc_df['Predicted_Return'] = raw_preds
            proc_df['Allocation_Signal'] = final_signals

        # Dashboard Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Average Predicted Return", f"{raw_preds.mean():.6f}")
        m2.metric("Maximum Leverage Advised", f"{final_signals.max():.2f}x")
        m3.metric("Minimum Leverage Advised", f"{final_signals.min():.2f}x")
        
        st.divider()
        
        # Visualizations Charts Area
        st.header("📊 Tactical Prediction & Allocation Plots")
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(proc_df['date_id'], proc_df['Allocation_Signal'], marker='o', color='#2ecc71', linewidth=1.5)
            ax1.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='100% Capital')
            ax1.set_title("Recommended Asset Allocation Timeline (0x to 2x)", fontweight='bold')
            ax1.set_xlabel("Date ID")
            ax1.set_ylabel("Allocation Scale")
            ax1.legend()
            st.pyplot(fig1)
            
        with col_plot2:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.histplot(proc_df['Predicted_Return'], bins=15, kde=True, color='#3498db', ax=ax2)
            ax2.axvline(0, color='red', linestyle='--')
            ax2.set_title("Distribution of Raw Returns Forecast", fontweight='bold')
            ax2.set_xlabel("Predicted Returns")
            st.pyplot(fig2)
            
        # Download Data Output Section
        st.divider()
        st.subheader("📥 Export Tactical Allocations Log")
        export_df = proc_df[['date_id', 'Predicted_Return', 'Allocation_Signal']]
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Predictions (CSV)",
            data=csv_data,
            file_name="hull_tactical_allocations.csv",
            mime="text/csv"
        )
        st.dataframe(export_df)
