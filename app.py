import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# ==============================
# 🔹 PAGE CONFIG
# ==============================
st.set_page_config(page_title="Energy Forecast Dashboard", layout="wide")
st.title("🔋 Energy Consumption Forecast Dashboard")

# ==============================
#  LOAD MODEL + SCALERS
# ==============================
@st.cache_resource
def load_model_and_scalers():
    model = load_model("energy_lstm_model.h5", compile=False)

    with open("scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)

    with open("scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)

    return model, scaler_X, scaler_y


@st.cache_data
def load_data():
    df = pd.read_csv("sample_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


model, scaler_X, scaler_y = load_model_and_scalers()
df = load_data()

# ==============================
# 🔹 SIDEBAR
# ==============================
st.sidebar.header("⚙️ Controls")

building = st.sidebar.selectbox(
    "Select Building", df["building_id"].unique()
)

num_points = st.sidebar.slider("Past Data Points", 50, 500, 150)
future_steps = st.sidebar.slider("Future Steps", 10, 100, 30)

# ==============================
# 🔹 FILTER DATA
# ==============================
df_building = df[df["building_id"] == building].copy()
df_building = df_building.sort_values("timestamp")

# ==============================
# 🔹 FEATURE ENGINEERING (ONLY BASIC)
# ==============================
df_building["hour"] = df_building["timestamp"].dt.hour
df_building["weekday"] = df_building["timestamp"].dt.weekday

# ==============================
# 🔹 FEATURES (MATCH TRAINING)
# ==============================
features = [
    "air_temperature",
    "hour",
    "weekday",
    "sqft",
    "meter_gas",
    "meter_water"
]

X = df_building[features]
X_scaled = scaler_X.transform(X)

# ==============================
# 🔹 LSTM SEQUENCE
# ==============================
last_sequence = X_scaled[-24:]
last_sequence = last_sequence.reshape(1, 24, X_scaled.shape[1])

# ==============================
# 🔹 FUTURE PREDICTION
# ==============================
future_predictions = []
current_seq = last_sequence.copy()

for _ in range(future_steps):
    pred = model.predict(current_seq, verbose=0)[0][0]
    future_predictions.append(pred)

    next_step = current_seq[0, -1].copy()
    current_seq = np.append(current_seq[:, 1:, :], [[next_step]], axis=1)

# Inverse scaling
future_predictions = scaler_y.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
)

# ==============================
# 🔹 FUTURE DATES
# ==============================
future_dates = pd.date_range(
    start=df_building["timestamp"].iloc[-1],
    periods=future_steps,
    freq="H"
)

# ==============================
# 🔹 METRICS
# ==============================
col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest Consumption",
    f"{df_building['meter_reading'].iloc[-1]:.2f}"
)

col2.metric(
    "Next Prediction",
    f"{future_predictions[0][0]:.2f}"
)

col3.metric(
    "Average Consumption",
    f"{df_building['meter_reading'].mean():.2f}"
)

# ==============================
# 🔹 PLOT
# ==============================
fig = go.Figure()

# Past
fig.add_trace(go.Scatter(
    x=df_building["timestamp"].tail(num_points),
    y=df_building["meter_reading"].tail(num_points),
    mode='lines',
    name='Actual (Past)',
    line=dict(width=3)
))

# Future
fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_predictions.flatten(),
    mode='lines',
    name='Predicted (Future)',
    line=dict(dash='dash', width=3)
))

fig.update_layout(
    title="📈 Energy Consumption Forecast",
    xaxis_title="Time",
    yaxis_title="Energy Usage",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# 🔹 DOWNLOAD
# ==============================
future_df = pd.DataFrame({
    "timestamp": future_dates,
    "prediction": future_predictions.flatten()
})

csv = future_df.to_csv(index=False)

st.download_button(
    "📥 Download Predictions",
    data=csv,
    file_name="future_predictions.csv",
    mime="text/csv"
)

# ==============================
# 🔹 DATA VIEW
# ==============================
with st.expander("📊 View Data"):
    st.dataframe(df_building.tail(50))