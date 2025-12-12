import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# 1️⃣ Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Tunisair Flight Delay Predictor",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Tunisair Flight Delay & Duration Predictor")
st.markdown("""
Predict expected **delay duration (in minutes)** for Tunisair flights using historical data and machine learning.
""")

# -----------------------------
# 2️⃣ Load Model and Data
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("tunisair_delay_regressor.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("Tunisair_flights_cleaned_data.csv")

try:
    regressor = load_model()
    df = load_data()
    st.success("✅ Model and data loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model or data: {e}")
    st.stop()

# -----------------------------
# 3️⃣ Prepare Dropdown Options
# -----------------------------
# Extract origin and destination from the "Route" column ("ABJ → TUN")
df[['Origin', 'Destination']] = df['Route'].str.split('→', expand=True)
df['Origin'] = df['Origin'].str.strip()
df['Destination'] = df['Destination'].str.strip()

origins = sorted(df['Origin'].dropna().unique().tolist())
destinations = sorted(df['Destination'].dropna().unique().tolist())
aircraft_codes = sorted(df['Aircraft_code'].dropna().unique().tolist())

months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# -----------------------------
# 4️⃣ Input Form
# -----------------------------
st.subheader("🧭 Input Flight Details")

with st.form("delay_prediction_form"):
    origin = st.selectbox("Origin Airport", origins, index=0)
    destination_options = [d for d in destinations if d != origin]
    destination = st.selectbox("Destination Airport", destination_options, index=0)
    dep_dayofweek = st.selectbox("Departure Day of Week", days_of_week)
    dep_hour = st.number_input("Departure Hour (0–23)", min_value=0, max_value=23, value=10)
    dep_month = st.selectbox("Departure Month", months)
    aircraft_code = st.selectbox("Aircraft Code", aircraft_codes, index=0)
    submitted = st.form_submit_button("🔍 Predict Delay")

# -----------------------------
# 5️⃣ Handle Prediction
# -----------------------------
if submitted:
    try:
        # Construct correct route string (exactly as in dataset)
        route = f"{origin} → {destination}"

        # Compute rolling averages for that route
        route_stats = df[df["Route"] == route][["route_rolling_7", "route_rolling_30"]].dropna()
        if not route_stats.empty:
            avg_7 = np.round(route_stats["route_rolling_7"].mean(), 2)
            avg_30 = np.round(route_stats["route_rolling_30"].mean(), 2)
        else:
            avg_7, avg_30 = np.nan, np.nan  # better than forcing 0.0 to expose missing data

        st.markdown("### 📊 Route Historical Averages:")
        if not np.isnan(avg_7) and not np.isnan(avg_30):
            st.write(f"**7-Day Rolling Average Delay:** {avg_7} minutes")
            st.write(f"**30-Day Rolling Average Delay:** {avg_30} minutes")
        else:
            st.warning("⚠️ No historical delay data found for this route in the dataset.")

        # Prepare model input — includes rolling averages
        input_data = {
            "Route": [route],
            "dep_dayofweek": [dep_dayofweek],
            "dep_hour": [dep_hour],
            "dep_month": [dep_month],
            "Aircraft_code": [aircraft_code],
            "route_rolling_7": [avg_7 if not np.isnan(avg_7) else 0],
            "route_rolling_30": [avg_30 if not np.isnan(avg_30) else 0]
        }
        input_df = pd.DataFrame(input_data)

        # Ensure correct dtypes
        cat_cols = ['Route', 'dep_dayofweek', 'dep_month', 'Aircraft_code']
        num_cols = ['dep_hour', 'route_rolling_7', 'route_rolling_30']

        for col in cat_cols:
            input_df[col] = input_df[col].astype(str)
        for col in num_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        st.markdown("### ✅ Processed Input Sample:")
        st.dataframe(input_df)

        # Predict delay
        prediction = regressor.predict(input_df)
        delay_minutes = float(np.round(prediction[0], 2))

        # Display prediction
        st.success(f"🕓 **Predicted Delay:** {delay_minutes} minutes")

        if delay_minutes < 5:
            st.info("🟢 Flight likely to be on time or with minimal delay.")
        elif delay_minutes < 30:
            st.warning("🟡 Minor delay expected.")
        else:
            st.error("🔴 Significant delay likely.")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# -----------------------------
# 6️⃣ Footer
# -----------------------------
st.markdown("---")
st.caption("Developed for Tunisair Flight Analytics Project | Powered by Streamlit & Scikit-learn")
