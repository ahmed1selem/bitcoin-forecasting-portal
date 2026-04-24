# ₿ Bitcoin Price Forecasting Portal

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

An interactive, production-quality web application built with Streamlit designed specifically for financial time-series analysis. This tool allows users to upload standard Kaggle-style historical Bitcoin (BTC) datasets to seamlessly analyze and forecast price trends using mathematically rigorous, research-backed Machine Learning pipelines.

---

## 🎯 Purpose of the Project

The objective of this portal is to provide an accessible, purely data-driven forecasting tool that relies strictly on structural historical data. Rather than relying on simple extrapolations, the app compares three highly distinct mathematical approaches (Statistical, Additive, and Tree-based Ensembles) to model Bitcoin's extreme volatility. 

The application is entirely **self-contained**. It requires no external API connections, executing all feature engineering and mathematical transformations purely on the user's uploaded `.csv` file.

---

## 🤖 Models & Implementations

The portal dynamically evaluates and compares three forecasting paradigms:

1. **ARIMA (AutoRegressive Integrated Moving Average)**:
   - *Implementation:* Uses `pmdarima.auto_arima` to dynamically search for the optimal $(p,d,q)$ parameters.
   - *Role:* Serves as the classic econometric baseline. Assumes linear relationships and struggles with Bitcoin's non-stationary regime shifts, but provides a grounded statistical benchmark.

2. **Prophet**:
   - *Implementation:* Meta's generalized additive model configured with weekly/yearly seasonality and automatic changepoint detection.
   - *Role:* Highly robust to missing data and structural shifts, making it exceptionally powerful for capturing the cyclical nature of cryptocurrency markets.

3. **XGBoost (Extreme Gradient Boosting)**:
   - *Implementation:* An advanced non-linear regression model driven by an extensively engineered feature matrix (lags, rolling volatility, volume, and technical indicators).
   - *Role:* The research-driven powerhouse of the app, designed specifically to capture complex, multi-dimensional momentum shifts that linear models cannot perceive.

---

## 📚 Academic Research & Integration

Our implementation of the XGBoost pipeline was heavily refined by integrating findings from several top-tier financial research papers to prevent standard ML pitfalls (like exponential target leakage and heteroskedasticity).

### 1. Khedr et al. (Survey) / Bouri et al. (2019) / Badenhorst (2018)
* **Finding:** These papers definitively proved Granger-causality between Bitcoin trading volume and subsequent price volatility. 
* **Implementation:** We updated our ingestion engine to automatically parse `Volume` columns from Kaggle datasets. We engineered a `volume_lag_1` feature, effectively feeding the XGBoost model crucial data regarding market participation and liquidity to predict imminent price breakouts.

### 2. Chen et al. (2020) — "Bitcoin Price Prediction Using Machine Learning"
* **Finding:** Tree-based ensembles (like XGBoost) inherently outperform ARIMA on cryptocurrency data because they can capture non-linear regime transitions, provided the sample dimension (feature matrix) is correctly engineered.
* **Implementation:** We implemented an array of non-collinear technical indicators strictly shifted to $(t-1)$ to avoid leakage:
  * **RSI (14-day)**: Captures overbought/oversold momentum.
  * **MACD**: Captures moving average convergence/divergence.
  * **Bollinger Bands**: Captures volatility expansions.

### 3. Quantitative Finance Standards (Stationarity Fixes)
* **Finding:** Tree-based models fail to extrapolate absolute values higher than their training data maximums. 
* **Implementation:** 
  * **Log Returns:** The XGBoost `.fit()` method was mathematically updated to predict the *Log Return* ($\ln(P_t / P_{t-1})$) rather than the absolute dollar price. This normalizes variance and strictly forces the model to predict momentum.
  * **1-Step Walk-Forward Backtesting:** During evaluation, the model uses actual ground truth $P_{t-1}$ to predict $P_t$, exactly mirroring how quantitative trading models are evaluated out-of-sample in the real world. 

---

## ⚙️ Installation & Usage

### 1. Requirements

Ensure you have Python 3.9+ installed. Install the necessary dependencies via the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Running the Portal

Launch the Streamlit portal from your terminal:

```bash
streamlit run app.py
```

### 3. How to Use
1. **Upload Data:** Drag and drop a standard BTC `.csv` file (e.g., from Binance or Kaggle). The app will auto-detect the date format (including `Open time` / `Close time` formats) and price columns.
2. **Configure:** Use the sidebar to select which models to train, define the forecasting horizon, and set the confidence intervals.
3. **Train Automatically:** After upload and configuration, the app preprocesses the data, trains the selected models, and runs the backtest automatically.
4. **Generate Forecast:** Click `Generate Forecast` to create the future projections for the selected horizon.
5. **Analyze:** Review the interactive Plotly chart and the detailed metrics table (MAE, RMSE, MAPE, MDA) to determine the best-performing model.
