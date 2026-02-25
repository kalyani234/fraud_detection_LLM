import streamlit as st
import pandas as pd
from src.data.loader import load_data
from src.agent.agent import run_fraud_analysis

# ───────────────────────── PAGE CONFIG ─────────────────────────
st.set_page_config(
    page_title="PaySim Fraud Agent",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ PaySim Fraud Detection Agent")
st.markdown("**4-Signal Weighted Framework** • Production-Ready")

# ───────────────────────── SIDEBAR ─────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    sample_pct = st.slider("Data Sample (%)", 1, 100, 5)
    show_raw = st.checkbox("Show raw JSON")
    show_truth = st.checkbox("Show ground truth", value=True)

# ───────────────────────── LOAD DATA ─────────────────────────
@st.cache_resource  # Changed: cache_resource is better for data
def get_data(sample):
    return load_data(sample_frac=sample / 100)

df = get_data(sample_pct)

if df.empty:
    st.error("No data loaded.")
    st.stop()

st.success(f"✅ Loaded {len(df):,} rows ({sample_pct}%)")

# ───────────────────────── TRANSACTION SELECT ─────────────────────────
st.subheader("📊 Transaction Selection")

if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0

col1, col2, col3 = st.columns([5, 2, 2])

with col1:
    st.session_state.current_idx = st.number_input(
        "Row index", 0, len(df)-1, st.session_state.current_idx
    )

with col2:
    if st.button("🔀 Random"):
        st.session_state.current_idx = int(df.sample(1).index[0])
        st.rerun()

with col3:
    frauds = df[df["isFraud"] == 1]
    if not frauds.empty and st.button("⚠️ Fraud"):
        st.session_state.current_idx = int(frauds.sample(1).index[0])
        st.rerun()

idx = st.session_state.current_idx
tx = df.iloc[idx].to_dict()

# ───────────────────────── DISPLAY ─────────────────────────
st.divider()
st.subheader(f"Transaction #{idx:,}")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("**📋 Main Fields**")
    st.dataframe(pd.DataFrame({
        "Field": ["Type", "Amount", "Origin", "Destination"],
        "Value": [
            tx.get("type"),
            f"{tx.get('amount', 0):,.0f}",
            tx.get("nameOrig"),
            tx.get("nameDest")
        ]
    }), hide_index=True, width='stretch')

with col2:
    st.markdown("**💰 Balances**")
    st.dataframe(pd.DataFrame({
        "Type": ["Old", "New"],
        "Origin": [
            f"{tx.get('oldbalanceOrg', 0):,.0f}",
            f"{tx.get('newbalanceOrig', 0):,.0f}"
        ],
        "Dest": [
            f"{tx.get('oldbalanceDest', 0):,.0f}",
            f"{tx.get('newbalanceDest', 0):,.0f}"
        ]
    }), hide_index=True, width='stretch')

# Quick Metrics
st.markdown("**📊 Quick Stats**")
c1, c2, c3, c4 = st.columns(4)

ratio = tx.get('amount', 0) / max(tx.get('oldbalanceOrg', 1), 1)
c1.metric("Ratio", f"{ratio:.1f}x")

merchant = "Yes ✓" if str(tx.get("nameDest", "")).startswith("M") else "No"
c2.metric("Merchant", merchant)

risk = "High ⚠️" if tx.get("type") in ["TRANSFER", "CASH_OUT"] else "Safe ✓"
c3.metric("Type Risk", risk)

truth = "FRAUD 🔴" if tx.get("isFraud") else "LEGIT 🟢"
c4.metric("Truth", truth)

if show_raw:
    with st.expander("📄 Raw JSON"):
        st.json(tx)

# ───────────────────────── ANALYSIS ─────────────────────────
st.divider()

if st.button("🔍 ANALYZE TRANSACTION", type="primary", width='stretch'):
    with st.spinner("Running 4-signal analysis..."):
        result = run_fraud_analysis(tx, mode="production")
        st.subheader("🤖 Agent Analysis")
        st.markdown(result)

        if show_truth:
            st.divider()
            truth = "🔴 FRAUD" if tx.get("isFraud") else "🟢 LEGITIMATE"
            color = "red" if tx.get("isFraud") else "green"
            st.markdown(
                f"**Ground Truth:** <span style='color:{color}'>{truth}</span>",
                unsafe_allow_html=True
            )

# ───────────────────────── FOOTER ─────────────────────────
st.divider()

col1, col2, col3 = st.columns(3)

col1.metric("Transactions", f"{len(df):,}")

fraud_count = df["isFraud"].sum()
fraud_pct = fraud_count / len(df) * 100

col2.metric("Fraud Cases", f"{fraud_count:,}", f"{fraud_pct:.2f}%")
col3.metric("Sample Size", f"{sample_pct}%")

st.caption("**PaySim Fraud Detection** • 4-Signal Framework • LangGraph + GPT-4o-mini")