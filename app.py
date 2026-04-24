"""Streamlit app for comparing BTC forecasting models."""

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import detect_columns, engineer_features, load_data, preprocess, train_test_split_ts
from evaluation import compare_models, select_winner
from models import ARIMAModel, ProphetModel, XGBoostModel


SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "sample_btc.csv")

MODEL_COLORS = {
    "ARIMA": "#00d2ff",
    "Prophet": "#ff6ec7",
    "XGBoost": "#ffd700",
}

MODEL_COLORS_RGBA = {
    "ARIMA": "rgba(0, 210, 255, 0.15)",
    "Prophet": "rgba(255, 110, 199, 0.15)",
    "XGBoost": "rgba(255, 215, 0, 0.15)",
}


st.set_page_config(
    page_title="₿ Bitcoin Forecasting Portal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    /* Dark-themed cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        text-align: center;
    }
    .metric-card h3 {
        color: #e94560;
        margin: 0;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card p {
        color: #eee;
        font-size: 28px;
        font-weight: 700;
        margin: 8px 0 0 0;
    }

    /* Winner banner */
    .winner-banner {
        background: linear-gradient(135deg, #0f3460 0%, #533483 50%, #e94560 100%);
        border-radius: 16px;
        padding: 24px 32px;
        text-align: center;
        margin: 16px 0;
        animation: glow 2s ease-in-out infinite alternate;
    }
    .winner-banner h2 {
        color: #fff;
        margin: 0;
        font-size: 24px;
    }
    .winner-banner .trophy {
        font-size: 48px;
    }

    @keyframes glow {
        from { box-shadow: 0 0 10px rgba(233, 69, 96, 0.3); }
        to { box-shadow: 0 0 25px rgba(233, 69, 96, 0.6); }
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #e94560, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 28px;
        font-weight: 800;
        margin-top: 32px;
    }

    /* Sidebar styling *
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #1a1a2e 100%);
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


def create_plotly_layout(title: str, yaxis_title: str = "Price (USD)") -> dict:
    """Shared chart styling."""
    return {
        "title": {"text": title, "font": {"size": 20, "color": "#e94560"}},
        "template": "plotly_dark",
        "paper_bgcolor": "#0a0a1a",
        "plot_bgcolor": "#0a0a1a",
        "xaxis": {
            "title": "Date",
            "gridcolor": "#1a1a2e",
            "showgrid": True,
            "rangeslider": {"visible": True, "bgcolor": "#0a0a1a"},
        },
        "yaxis": {
            "title": yaxis_title,
            "gridcolor": "#1a1a2e",
            "showgrid": True,
            "tickformat": "$,.0f",
        },
        "legend": {
            "bgcolor": "rgba(10,10,26,0.8)",
            "bordercolor": "#0f3460",
            "borderwidth": 1,
            "font": {"color": "#eee"},
        },
        "hovermode": "x unified",
        "height": 550,
        "margin": {"l": 60, "r": 30, "t": 60, "b": 40},
    }


def add_confidence_band(fig, x_values, lower_values, upper_values, fill_color, upper_name, lower_name):
    """Add the usual upper/lower filled band to a Plotly figure."""
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=upper_values,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name=upper_name,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=lower_values,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=fill_color,
            showlegend=False,
            name=lower_name,
            hoverinfo="skip",
        )
    )


@st.cache_data
def cached_load(file_content, file_name):
    """Cache uploaded CSV content so widgets do not reload it every time."""
    import io

    return load_data(io.BytesIO(file_content))


@st.cache_data
def cached_load_from_path(path):
    """Cache loading the bundled sample file."""
    return pd.read_csv(path)


with st.sidebar:
    st.markdown("## ₿ **Bitcoin Forecasting**")
    st.markdown("---")

    st.markdown("### 📂 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload BTC CSV (Kaggle format)",
        type=["csv"],
        help="Expected columns: Date/Timestamp + OHLC prices",
    )

    use_sample = False
    if os.path.exists(SAMPLE_DATA_PATH):
        use_sample = st.checkbox(
            "📊 Use built-in sample data",
            value=False,
            help="Load a pre-generated BTC dataset for quick demo",
        )

    st.markdown("---")


st.markdown(
    """
<div style="text-align: center; padding: 20px 0;">
    <h1 style="background: linear-gradient(90deg, #00d2ff, #e94560, #ffd700);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               font-size: 42px; font-weight: 900; margin-bottom: 4px;">
        ₿ Bitcoin Price Forecasting Portal
    </h1>
    <p style="color: #888; font-size: 16px; margin-top: 0;">
        Multi-model comparison • Interactive visualization • Research-driven analysis
    </p>
</div>
""",
    unsafe_allow_html=True,
)


has_data = uploaded_file is not None or use_sample

if not has_data:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <h3>📊 3 Models</h3>
            <p>ARIMA · Prophet · XGBoost</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <h3>🔬 Research-Driven</h3>
            <p>Paper-backed features</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <h3>🏆 Auto Comparison</h3>
            <p>MAE · RMSE · Winner</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.info(
        "👈 **Upload a BTC CSV file** in the sidebar to get started.\n\n"
        "Expected format: Kaggle Bitcoin dataset with Date + OHLC columns."
    )
    st.stop()


if uploaded_file is not None:
    raw_df = cached_load(uploaded_file.getvalue(), uploaded_file.name)
elif use_sample:
    raw_df = cached_load_from_path(SAMPLE_DATA_PATH)
    raw_df.columns = raw_df.columns.str.strip()
else:
    st.stop()


try:
    date_col, price_cols = detect_columns(raw_df)
except ValueError as error:
    st.error(f"❌ Column detection failed: {error}")
    st.stop()


with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    price_col_select = st.selectbox(
        "Price Column",
        options=price_cols,
        index=0,
        help="Which price to forecast (Close is most common)",
    )

    st.markdown("### 🤖 Models & Hyperparameters")
    selected_models = st.multiselect(
        "Select models to compare:",
        options=["ARIMA", "Prophet", "XGBoost"],
        default=["Prophet", "XGBoost"],
    )

    forecast_horizon = st.slider(
        "Forecast Horizon (days)",
        min_value=7,
        max_value=90,
        value=30,
        step=1,
        help="How many days into the future to forecast",
    )

    confidence_level = st.slider(
        "Confidence Interval (%)",
        min_value=80,
        max_value=99,
        value=95,
        step=1,
        help="Width of the prediction confidence band",
    ) / 100.0

    st.markdown("---")
    st.markdown("### 📅 Training Date Range")

    temp_dates = pd.to_datetime(raw_df[date_col], infer_datetime_format=True, utc=True)
    temp_dates = temp_dates.dt.tz_localize(None)
    min_date = temp_dates.min().date()
    max_date = temp_dates.max().date()

    date_range = st.slider(
        "Select training window",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD",
        help="Model trains ONLY on data within this range",
    )

    st.caption(f"📊 Dataset: **{min_date}** to **{max_date}** ({(max_date - min_date).days} days)")

    st.markdown("---")
    st.markdown("### 🛈 About")
    st.caption(
        "Built with Streamlit + Plotly.\n\n"
        "Models: ARIMA (pmdarima), Prophet, XGBoost.\n\n"
        "Research basis: Chen et al. (2020), Khedr et al. (Survey)"
    )


if not selected_models:
    st.warning("⚠️ Please select at least one model from the sidebar.")
    st.stop()


with st.spinner("🔄 Preprocessing data..."):
    df = preprocess(
        raw_df,
        date_col,
        price_col_select,
        start_date=pd.Timestamp(date_range[0]),
        end_date=pd.Timestamp(date_range[1]),
    )

if len(df) < 60:
    st.error(f"❌ Only {len(df)} data points in selected range. Need at least 60 for reliable modeling.")
    st.stop()


st.markdown('<p class="section-header">📊 Dataset Overview</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Data Points", f"{len(df):,}")
with col2:
    st.metric("Date Range", f"{df['ds'].iloc[0].strftime('%Y-%m-%d')} → {df['ds'].iloc[-1].strftime('%Y-%m-%d')}")
with col3:
    st.metric(f"Latest {price_col_select}", f"${df['y'].iloc[-1]:,.2f}")
with col4:
    pct_change = ((df["y"].iloc[-1] / df["y"].iloc[0]) - 1) * 100
    st.metric("Total Change", f"{pct_change:+.1f}%")


train_df, test_df = train_test_split_ts(df, test_ratio=0.2)
train_feat = engineer_features(train_df)
confidence = confidence_level

st.caption(f"🔀 Train: {len(train_df)} points | Test: {len(test_df)} points (80/20 split, chronological)")


st.markdown('<p class="section-header">🤖 Model Training</p>', unsafe_allow_html=True)

results = {}
forecasts = {}

progress_bar = st.progress(0, text="Training models...")
total_models = len(selected_models)

for index, model_name in enumerate(selected_models):
    progress_bar.progress(
        index / total_models,
        text=f"Training {model_name}... ({index + 1}/{total_models})",
    )

    with st.spinner(f"⏳ Training **{model_name}**..."):
        try:
            if model_name == "ARIMA":
                model = ARIMAModel()
                backtest_result = model.backtest(train_df, test_df, confidence)
                model.fit(df)

                future_forecast = model.predict(forecast_horizon, confidence)
                future_forecast["ds"] = pd.date_range(
                    start=df["ds"].iloc[-1] + pd.Timedelta(days=1),
                    periods=forecast_horizon,
                    freq="D",
                )

            elif model_name == "Prophet":
                model = ProphetModel()
                backtest_result = model.backtest(train_df, test_df, confidence)
                model.fit(df, confidence)
                future_forecast = model.predict(forecast_horizon, confidence)

            elif model_name == "XGBoost":
                model = XGBoostModel()
                backtest_result = model.backtest(train_df, test_df, train_feat, confidence)
                full_features = engineer_features(df)
                model.fit(full_features, confidence)
                future_forecast = model.predict_recursive(df, forecast_horizon)

            else:
                continue

            forecasts[model_name] = future_forecast
            results[model_name] = backtest_result
            results[model_name]["model_obj"] = model

        except Exception as error:
            st.error(f"❌ {model_name} failed: {str(error)}")
            continue

progress_bar.progress(1.0, text="✅ All models trained!")


if not results:
    st.error("❌ No models completed successfully. Check your data and try again.")
    st.stop()


st.markdown('<p class="section-header">📈 Forecast Chart</p>', unsafe_allow_html=True)

fig_main = go.Figure()
fig_main.add_trace(
    go.Scatter(
        x=df["ds"],
        y=df["y"],
        mode="lines",
        name="Historical Price",
        line=dict(color="#ffffff", width=2),
        opacity=0.8,
    )
)

# add_vline can be a little inconsistent with timestamps, so a manual line is safer here
split_date = test_df["ds"].iloc[0]
split_str = str(split_date)
fig_main.update_layout(
    shapes=[
        dict(
            type="line",
            x0=split_str,
            x1=split_str,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#888", width=1.5, dash="dash"),
        )
    ],
    annotations=[
        dict(
            x=split_str,
            y=1.05,
            yref="paper",
            text="Train | Test",
            showarrow=False,
            font=dict(color="#888", size=11),
        )
    ],
)

for model_name in selected_models:
    if model_name not in forecasts:
        continue

    forecast_df = forecasts[model_name]
    line_color = MODEL_COLORS.get(model_name, "#fff")
    fill_color = MODEL_COLORS_RGBA.get(model_name, "rgba(255,255,255,0.1)")

    add_confidence_band(
        fig_main,
        forecast_df["ds"],
        forecast_df["yhat_lower"],
        forecast_df["yhat_upper"],
        fill_color,
        f"{model_name} Upper CI",
        f"{model_name} Lower CI",
    )
    fig_main.add_trace(
        go.Scatter(
            x=forecast_df["ds"],
            y=forecast_df["yhat"],
            mode="lines",
            name=f"{model_name} Forecast",
            line=dict(color=line_color, width=2.5, dash="dot"),
        )
    )

fig_main.update_layout(
    **create_plotly_layout(f"BTC {price_col_select} Price — {forecast_horizon}-Day Forecast")
)

st.plotly_chart(fig_main, use_container_width=True)


st.markdown('<p class="section-header">🔥 Model Competition — Test Set</p>', unsafe_allow_html=True)
st.caption("All models evaluated on the same held-out test set. Toggle models on/off in the legend.")

fig_comp = go.Figure()
test_dates = test_df["ds"].values

fig_comp.add_trace(
    go.Scatter(
        x=test_dates,
        y=test_df["y"].values,
        mode="lines",
        name="Actual Price",
        line=dict(color="#ffffff", width=2.5),
    )
)

for model_name in selected_models:
    if model_name not in results:
        continue

    result = results[model_name]
    line_color = MODEL_COLORS.get(model_name, "#fff")
    fill_color = MODEL_COLORS_RGBA.get(model_name, "rgba(255,255,255,0.1)")

    add_confidence_band(
        fig_comp,
        result["dates"],
        result["y_lower"],
        result["y_upper"],
        fill_color,
        f"{model_name} Upper CI",
        f"{model_name} Lower CI",
    )
    fig_comp.add_trace(
        go.Scatter(
            x=result["dates"],
            y=result["y_pred"],
            mode="lines",
            name=f"{model_name} (RMSE: {result['metrics']['RMSE']:,.0f})",
            line=dict(color=line_color, width=2),
        )
    )

fig_comp.update_layout(
    **create_plotly_layout("Model Competition — Predictions vs. Actual on Test Set")
)

st.plotly_chart(fig_comp, use_container_width=True)


st.markdown('<p class="section-header">🏆 Results & Winner</p>', unsafe_allow_html=True)

comparison_df = compare_models(results)
winner_name = select_winner(results)
winner_metrics = results[winner_name]["metrics"]

st.markdown(
    f"""
<div class="winner-banner">
    <div class="trophy">🏆</div>
    <h2>Winner: {winner_name}</h2>
    <p style="color: #ddd; margin: 8px 0 0 0; font-size: 18px;">
        MAE: ${winner_metrics['MAE']:,.2f} &nbsp;|&nbsp; RMSE: ${winner_metrics['RMSE']:,.2f} <br>
        MAPE: {winner_metrics.get('MAPE', 'N/A')}% &nbsp;|&nbsp; MDA: {winner_metrics.get('MDA', 'N/A')}%
    </p>
</div>
""",
    unsafe_allow_html=True,
)

metric_columns = st.columns(len(results))
for index, (model_name, result) in enumerate(results.items()):
    with metric_columns[index]:
        is_winner = model_name == winner_name
        border_color = "#e94560" if is_winner else "#0f3460"
        badge = " 🏆" if is_winner else ""

        st.markdown(
            f"""
        <div class="metric-card" style="border-color: {border_color}; border-width: {'2px' if is_winner else '1px'};">
            <h3>{model_name}{badge}</h3>
            <p style="font-size: 16px;">
                MAE: ${result['metrics']['MAE']:,.2f}<br>
                RMSE: ${result['metrics']['RMSE']:,.2f}<br>
                MAPE: {result['metrics'].get('MAPE', 'N/A')}%<br>
                MDA: {result['metrics'].get('MDA', 'N/A')}%
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

st.markdown("### Detailed Metrics Table")
st.dataframe(
    comparison_df.style.format(
        {
            "MAE": "${:,.2f}",
            "RMSE": "${:,.2f}",
            "MAPE (%)": "{:.2f}%",
            "MDA (%)": "{:.2f}%",
        }
    ),
    use_container_width=True,
)


fig_bar = go.Figure()
fig_bar.add_trace(
    go.Bar(
        x=comparison_df["Model"],
        y=comparison_df["MAE"],
        name="MAE",
        marker_color="#00d2ff",
        text=comparison_df["MAE"].apply(lambda value: f"${value:,.0f}"),
        textposition="auto",
    )
)
fig_bar.add_trace(
    go.Bar(
        x=comparison_df["Model"],
        y=comparison_df["RMSE"],
        name="RMSE",
        marker_color="#e94560",
        text=comparison_df["RMSE"].apply(lambda value: f"${value:,.0f}"),
        textposition="auto",
    )
)

fig_bar.update_layout(
    title=dict(text="Model Comparison — Error Metrics", font=dict(size=18, color="#e94560")),
    template="plotly_dark",
    paper_bgcolor="#0a0a1a",
    plot_bgcolor="#0a0a1a",
    barmode="group",
    yaxis_title="Error ($)",
    height=400,
    legend=dict(bgcolor="rgba(10,10,26,0.8)", bordercolor="#0f3460"),
)

st.plotly_chart(fig_bar, use_container_width=True)


st.markdown('<p class="section-header">🔍 Model Details</p>', unsafe_allow_html=True)

for model_name in selected_models:
    if model_name not in results:
        continue

    with st.expander(f"📋 {model_name} — Details & Parameters", expanded=False):
        model_obj = results[model_name].get("model_obj")
        if model_obj:
            st.markdown(model_obj.get_info())

        st.markdown(f"**Backtest parameters:** {results[model_name].get('params', 'N/A')}")

        if model_name == "XGBoost" and model_obj and model_obj.feature_importances_:
            feature_importances = model_obj.feature_importances_
            sorted_importances = dict(
                sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
            )

            fig_fi = go.Figure(
                go.Bar(
                    x=list(sorted_importances.values()),
                    y=list(sorted_importances.keys()),
                    orientation="h",
                    marker_color="#ffd700",
                )
            )
            fig_fi.update_layout(
                title="Feature Importances",
                template="plotly_dark",
                paper_bgcolor="#0a0a1a",
                plot_bgcolor="#0a0a1a",
                height=350,
                yaxis=dict(autorange="reversed"),
                xaxis_title="Importance",
                margin=dict(l=120),
            )
            st.plotly_chart(fig_fi, use_container_width=True)


st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #555; font-size: 13px; padding: 16px 0;">
    <p>🛈 About<br>
    Built with Streamlit + Plotly.<br><br>
    Models: ARIMA (pmdarima), Prophet, XGBoost.<br><br>
    Main paper used in this project:<br>
    Chen et al. (2020), "Bitcoin Price Prediction Using Machine Learning:
    An Approach to Sample Dimension Engineering."<br><br>
    Additional background: Khedr et al. (Survey)</p>
</div>
""",
    unsafe_allow_html=True,
)
