import json
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Sri Lanka Flood Risk Ensemble Dashboard", layout="wide")

@st.cache_resource
def load_artifacts():
    bagging = joblib.load("bagging_model.joblib")
    boosting = joblib.load("boosting_model.joblib")
    stacking = joblib.load("stacking_model.joblib")
    with open("metrics.json") as f:
        metrics = json.load(f)
    with open("feature_stats.json") as f:
        stats = json.load(f)
    return bagging, boosting, stacking, metrics, stats

@st.cache_data
def load_data():
    return pd.read_csv("sri_lanka_flood_data.csv")

bagging_model, boosting_model, stacking_model, metrics, stats = load_artifacts()
df = load_data()

st.title("Sri Lanka Flood Risk Prediction — Ensemble Learning Dashboard")
st.markdown("Comparing **Bagging (Random Forest)**, **Boosting (XGBoost)**, and **Stacking** across Sri Lankan hydrological and infrastructure indicators.")

tab1, tab2, tab3 = st.tabs([
    "1. Live Flood Risk Simulator",
    "2. Ensemble Model Comparison",
    "3. Sri Lanka Dataset Explorer"
])

# --- TAB 1: LIVE PREDICTOR ---
with tab1:
    st.subheader("Simulate Sri Lankan Location Conditions")
    col_inputs, col_results = st.columns([1.2, 1])

    user_input = {}
    with col_inputs:
        c1, c2 = st.columns(2)
        feature_keys = list(stats.keys())
        for idx, feat in enumerate(feature_keys):
            target_col = c1 if idx % 2 == 0 else c2
            s = stats[feat]
            user_input[feat] = target_col.slider(
                label=feat.replace("_", " ").title(),
                min_value=float(s["min"]),
                max_value=float(s["max"]),
                value=float(s["median"])
            )

    input_df = pd.DataFrame([user_input])

    with col_results:
        st.subheader("Real-Time Ensemble Predictions")
        models = {
            "Bagging (Random Forest)": bagging_model,
            "Boosting (XGBoost)": boosting_model,
            "Stacking (Meta-Ensemble)": stacking_model
        }
        for name, mdl in models.items():
            prob = float(mdl.predict_proba(input_df)[0][1])
            pred = "HIGH FLOOD RISK" if prob >= 0.5 else "LOW FLOOD RISK"
            st.metric(label=name, value=pred, delta=f"Flood Probability: {prob*100:.1f}%")
            st.progress(prob)

# --- TAB 2: MODEL COMPARISON ---
with tab2:
    st.subheader("Bagging vs. Boosting vs. Stacking Performance")
    metrics_df = pd.DataFrame(metrics).T.drop(columns=["Confusion_Matrix", "Feature_Importances"])
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="#1f4e3d"), use_container_width=True)

    melted = metrics_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
    fig_bar = px.bar(
        melted, x="Metric", y="Score", color="index", barmode="group",
        labels={"index": "Ensemble Method"}, title="Evaluation Metrics Comparison"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        selected_model = st.selectbox("Select Model for Confusion Matrix", list(metrics.keys()))
        cm = metrics[selected_model]["Confusion_Matrix"]
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=["Predicted No Flood", "Predicted Flood"],
            y=["Actual No Flood", "Actual Flood"],
            color_continuous_scale="Blues",
            title=f"Confusion Matrix — {selected_model}"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with c_right:
        fi = metrics["Boosting (XGBoost)"]["Feature_Importances"]
        fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())}).sort_values("Importance")
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", title="Key Flood Risk Drivers (XGBoost)")
        st.plotly_chart(fig_fi, use_container_width=True)

# --- TAB 3: DATA EXPLORER ---
with tab3:
    st.subheader("Sri Lanka Dataset Overview")
    st.write(f"Total Records: **{len(df):,}**")
    st.dataframe(df.head(50), use_container_width=True)
    fig_scatter = px.scatter(
        df.sample(min(2000, len(df)), random_state=42),
        x="rainfall_7d_mm", y="elevation_m", color="flood_occurrence_current_event",
        title="7-Day Rainfall vs. Elevation by Flood Occurrence"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)