import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import datetime

# ------------------------------------
# PAGE CONFIG + THEME
# ------------------------------------
st.set_page_config(page_title="PowerBI Insurance Dashboard", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
body { background-color:#0f1117; color:#e6eef6; }
.header { text-align:center; margin-bottom:20px; }
.kpi { background:#151718; padding:16px; border-radius:10px; border:1px solid #23262b; text-align:center; }
.card { background:#151718; padding:16px; border-radius:12px; border:1px solid #23262b; margin-bottom:18px; }
.small { color:#9aa0a6; font-size:13px; }
.stButton>button { background:#2d66f4 !important; color:white !important; border-radius:8px; padding:10px 18px; }
input, select { background:#181a1d !important; color:#e6eef6 !important; border-radius:8px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'><h1>Insurance Claim – Anomaly Detection </h1>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------
# MODEL LOADING + FIX for datetime
# ------------------------------------
@st.cache_resource
def load_model_and_meta():
    df = pd.read_csv("insurance_claim.csv")

    # ❗ IMPORTANT FIX: Create date only for visual, never for model
    df["claim_date"] = pd.to_datetime("2023-01-01") + pd.to_timedelta(np.random.randint(0, 365, len(df)), unit="D")

    # Encode categoricals for model
    df_enc = pd.get_dummies(df, columns=["claim_type", "region", "income_bracket"], drop_first=False)

    # ❗ FIX: Remove anomaly + datetime before scaling
    drop_cols = ["anomaly", "claim_date"]
    X = df_enc.drop(columns=drop_cols, errors="ignore")

    REQUIRED_COLS = X.columns.tolist()

    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.04, random_state=42)
    model.fit(X_scaled)

    # Score ranges
    scores = model.decision_function(X_scaled)
    SCORE_MIN, SCORE_MAX = float(scores.min()), float(scores.max())

    return df, REQUIRED_COLS, scaler, model, SCORE_MIN, SCORE_MAX

# Load model and training meta
DF_TRAIN, REQUIRED_COLS, SCALER, MODEL, SCORE_MIN, SCORE_MAX = load_model_and_meta()

# ------------------------------------
# UTILITY HELPERS
# ------------------------------------

def align_dataframe(df_in):
    df = df_in.copy()

    # One-hot encode if columns present
    cat_cols = [c for c in ["claim_type","region","income_bracket"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # Ensure all required model cols exist
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = 0

    # Keep correct order
    return df[REQUIRED_COLS]

def score_to_percent(s, smin, smax):
    if smin == smax:
        return 50
    scaled = (s - smin) / (smax - smin)
    pct = (1 - scaled) * 100
    return float(np.clip(pct, 0, 100))

def compute_risk(df_enc):
    X_scaled = SCALER.transform(df_enc)
    raw_score = MODEL.decision_function(X_scaled)
    risk = [score_to_percent(s, SCORE_MIN, SCORE_MAX) for s in raw_score]
    return raw_score, risk

# ------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Analytics", "Single Claim", "Batch Upload"])

# ------------------------------------
# FILTERS
# ------------------------------------
claim_types = sorted(DF_TRAIN["claim_type"].unique())
regions = sorted(DF_TRAIN["region"].unique())
income_list = sorted(DF_TRAIN["income_bracket"].unique())

f_type = st.sidebar.multiselect("Claim Type", claim_types, default=claim_types)
f_region = st.sidebar.multiselect("Region", regions, default=regions)
f_inc = st.sidebar.multiselect("Income Bracket", income_list, default=income_list)

min_amt, max_amt = st.sidebar.slider(
    "Claim Amount Range",
    float(DF_TRAIN["claim_amount"].min()),
    float(DF_TRAIN["claim_amount"].max()),
    (float(DF_TRAIN["claim_amount"].min()), float(DF_TRAIN["claim_amount"].max()))
)

# Filtered training data for visuals
DF_VIS = DF_TRAIN[
    (DF_TRAIN["claim_type"].isin(f_type)) &
    (DF_TRAIN["region"].isin(f_region)) &
    (DF_TRAIN["income_bracket"].isin(f_inc)) &
    (DF_TRAIN["claim_amount"].between(min_amt, max_amt))
]

# ------------------------------------
# PAGE 1 – HOME
# ------------------------------------
if page == "Home":
    # KPIs
    total = len(DF_VIS)
    ana = int(DF_VIS["anomaly"].sum())
    pct = round(100 * ana / total, 2)
    avg = round(DF_VIS["claim_amount"].mean(), 2)
    maxc = round(DF_VIS["claim_amount"].max(), 2)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='kpi'>Total Claims<br><b>{total}</b></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'>Anomalies<br><b>{ana}</b></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'>Anomaly %<br><b>{pct}%</b></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'>Avg Claim<br><b>₹{avg:,.0f}</b></div>", unsafe_allow_html=True)

    st.markdown("")

    # VISUALS
    v1, v2 = st.columns([1.2, 1])

    with v1:
        st.subheader("Claim Type Distribution")
        ct = DF_VIS["claim_type"].value_counts()
        fig, ax = plt.subplots(figsize=(5,4))
        ax.pie(ct.values, labels=ct.index, autopct="%1.1f%%", wedgeprops=dict(width=0.4))
        ax.axis("equal")
        st.pyplot(fig)

    with v2:
        st.subheader("Region: Normal vs Anomaly")
        pivot = DF_VIS.groupby("region")["anomaly"].value_counts().unstack().fillna(0)
        if 0 not in pivot.columns: pivot[0] = 0
        if 1 not in pivot.columns: pivot[1] = 0
        fig2, ax2 = plt.subplots(figsize=(6,4))
        pivot[[0,1]].plot(kind="bar", stacked=True, ax=ax2, color=["#7fb3ff","#ff6b6b"])
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Region-wise Anomaly Intensity")
    heat = DF_VIS.groupby("region")["anomaly"].mean().reindex(regions).fillna(0)
    fig3, ax3 = plt.subplots(figsize=(6,2))
    sns.heatmap(heat.to_frame().T, annot=True, fmt=".2f", cmap="YlOrRd", cbar=False, ax=ax3)
    st.pyplot(fig3)

# ------------------------------------
# PAGE 2 – ANALYTICS
# ------------------------------------
elif page == "Analytics":
    st.subheader("Analytics & Trends")

    DF_VIS["month"] = DF_VIS["claim_date"].dt.to_period("M").dt.to_timestamp()
    trend = DF_VIS.groupby("month")["claim_amount"].mean()

    st.markdown("### Monthly Average Claim Amount")
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(trend.index, trend.values, marker="o", color="#7fb3ff")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.markdown("### Amount vs Risk (Training Sample)")
    sample = DF_VIS.sample(1200, random_state=42)
    df_enc = align_dataframe(sample)
    _, risk_vals = compute_risk(df_enc)
    sample["risk_pct"] = risk_vals

    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.scatter(sample["claim_amount"], sample["risk_pct"], alpha=0.5, s=20, c=sample["risk_pct"], cmap="coolwarm")
    ax2.set_xlabel("Claim Amount")
    ax2.set_ylabel("Risk %")
    st.pyplot(fig2)

# ------------------------------------
# PAGE 3 – SINGLE CLAIM
# ------------------------------------
elif page == "Single Claim":
    st.subheader("Single Claim Evaluation")

    with st.form("single"):
        c1, c2 = st.columns(2)
        with c1:
            cid = st.number_input("Claim ID", min_value=1, value=101)
            age = st.number_input("Customer Age", min_value=18, value=35)
            tenure = st.number_input("Policy Tenure (years)", value=5.0)
            amount = st.number_input("Claim Amount", value=20000.0)
            duration = st.number_input("Claim Duration (days)", min_value=1, value=6)
        with c2:
            hosp = st.number_input("Hospital Stay Days", min_value=0, value=2)
            prev = st.number_input("Previous Claims", min_value=0, value=0)
            ctype = st.selectbox("Claim Type", claim_types)
            region = st.selectbox("Region", regions)
            income = st.selectbox("Income Bracket", income_list)

        submit = st.form_submit_button("Evaluate")

    if submit:
        row = pd.DataFrame([{
            "claim_id": cid, "customer_age": age, "policy_tenure_years": tenure,
            "claim_amount": amount, "claim_duration_days": duration,
            "hospital_stay_days": hosp, "previous_claims": prev,
            "claim_type": ctype, "region": region, "income_bracket": income
        }])

        df_enc = align_dataframe(row)
        raw, risk = compute_risk(df_enc)
        score = risk[0]

        if score >= 65:
            st.error(f" High Risk — {score:.1f}%")
        elif score >= 35:
            st.warning(f" Medium Risk — {score:.1f}%")
        else:
            st.success(f" Low Risk — {score:.1f}%")

        st.write("Input Summary:")
        st.table(row.T)

# ------------------------------------
# PAGE 4 – BATCH UPLOAD
# ------------------------------------
elif page == "Batch Upload":
    st.subheader("Batch Upload & Prediction")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown("### Preview")
        st.dataframe(df.head())

        # Remove claim_date if exists
        if "claim_date" in df:
            df.drop(columns=["claim_date"], inplace=True)

        df_enc = align_dataframe(df)
        raw, risk = compute_risk(df_enc)
        df["risk_pct"] = risk
        df["prediction"] = (df["risk_pct"] >= 50).astype(int)

        st.markdown("### Results")
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", csv, "predictions.csv")

        st.subheader("Normal vs Anomaly")
        fig, ax = plt.subplots(figsize=(5,3))
        sns.countplot(x=df["prediction"], palette=["#7fb3ff","#ff6b6b"], ax=ax)
        ax.set_xticklabels(["Normal","Anomaly"])
        st.pyplot(fig)

