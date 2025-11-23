# app.py (transformed for robust deploy)
# CellGuard.AI - Streamlit app (robust, tolerant, deploy-ready)
# Based on user's original file. Minor defensive and UX fixes.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="CellGuard.AI - Robust", layout="wide")

# --------------------------
# Utilities
# --------------------------
def normalize_bms_columns(df):
    df = df.copy()
    simplified = {col: "".join(ch for ch in col.lower() if ch.isalnum()) for col in df.columns}
    patterns = {
        "voltage": ["volt", "vcell", "cellv", "packv"],
        "current": ["curr", "amp", "amps", "ichg", "idis", "current"],
        "temperature": ["temp", "temperature", "celltemp", "packtemp"],
        "soc": ["soc", "stateofcharge"],
        "cycle": ["cycle", "cyclecount", "chargecycle"],
        "time": ["time", "timestamp", "t", "index"],
    }
    col_map = {}
    used = set()
    for target, keys in patterns.items():
        for orig, s in simplified.items():
            if orig in used:
                continue
            if any(k in s for k in keys):
                col_map[target] = orig
                used.add(orig)
                break
    # rename original column -> canonical name
    rename = {orig: targ for targ, orig in col_map.items()}
    df = df.rename(columns=rename)
    return df, col_map

def ensure_columns(df, required):
    for c in required:
        if c not in df.columns:
            df[c] = np.nan
    return df

def generate_sample_bms_data(n=800, seed=42):
    np.random.seed(seed)
    t = np.arange(n)
    voltage = 3.7 + 0.05 * np.sin(t / 50) + np.random.normal(0, 0.005, n)
    current = 1.5 + 0.3 * np.sin(t / 30) + np.random.normal(0, 0.05, n)
    temperature = 30 + 3 * np.sin(t / 60) + np.random.normal(0, 0.3, n)
    soc = np.clip(80 + 10 * np.sin(t / 80) + np.random.normal(0, 1, n), 0, 100)
    cycle = t // 50
    idx = np.random.choice(n, size=25, replace=False)
    voltage[idx] -= np.random.uniform(0.04, 0.12, size=len(idx))
    temperature[idx] += np.random.uniform(3, 7, size=len(idx))
    return pd.DataFrame({"time": t, "voltage": voltage, "current": current, "temperature": temperature, "soc": soc, "cycle": cycle})

# --------------------------
# Feature engineering & models (defensive)
# --------------------------
def feature_engineering(df, window=10):
    df = df.copy()
    df = ensure_columns(df, ["voltage", "current", "temperature", "soc", "cycle", "time"])
    # Voltage features
    if df["voltage"].notna().sum() > 0:
        df["voltage_ma"] = df["voltage"].rolling(window, min_periods=1).mean()
        df["voltage_roc"] = df["voltage"].diff().fillna(0)
        df["voltage_var"] = df["voltage"].rolling(window, min_periods=1).var().fillna(0)
    else:
        df["voltage_ma"] = np.nan
        df["voltage_roc"] = np.nan
        df["voltage_var"] = np.nan

    # Temperature features
    if df["temperature"].notna().sum() > 0:
        df["temp_ma"] = df["temperature"].rolling(window, min_periods=1).mean()
        df["temp_roc"] = df["temperature"].diff().fillna(0)
        temp_mean = df["temperature"].mean()
        temp_std = df["temperature"].std()
        temp_threshold = temp_mean + 2 * temp_std if not np.isnan(temp_mean) and not np.isnan(temp_std) else np.nan
    else:
        df["temp_ma"] = np.nan
        df["temp_roc"] = np.nan
        temp_threshold = np.nan

    # Risk label from combined heuristics
    if df["voltage"].notna().sum() > 0:
        volt_drop_threshold = -0.03
        conditions = pd.Series(False, index=df.index)
        if df["temperature"].notna().sum() > 0 and not np.isnan(temp_threshold):
            conditions = conditions | (df["temperature"] > temp_threshold)
        # safe access to voltage_roc
        if "voltage_roc" in df.columns:
            conditions = conditions | (df["voltage_roc"] < volt_drop_threshold)
        df["risk_label"] = np.where(conditions, 1, 0)
    else:
        df["risk_label"] = 0

    return df

def build_models_and_scores(df, contamination=0.05):
    df = df.copy()
    possible = ["voltage", "current", "temperature", "soc", "voltage_ma", "voltage_roc", "temp_roc", "voltage_var", "temp_ma", "cycle"]
    anomaly_features = [f for f in possible if f in df.columns and df[f].notna().sum() > 0]
    df["anomaly_flag"] = 0
    df["risk_pred"] = 0
    df["battery_health_score"] = 50.0

    # Anomaly detection (IsolationForest) if enough valid rows
    if len(anomaly_features) >= 2 and df[anomaly_features].dropna().shape[0] >= 30:
        try:
            iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
            X = df[anomaly_features].fillna(df[anomaly_features].median())
            iso.fit(X)
            df["anomaly_flag"] = iso.predict(X).map({1: 0, -1: 1})
        except Exception:
            df["anomaly_flag"] = 0

    # Risk prediction (decision tree) if labels exist
    if "risk_label" in df.columns and df["risk_label"].nunique() > 1:
        clf_features = [f for f in anomaly_features if f in df.columns]
        if len(clf_features) >= 2:
            try:
                Xc = df[clf_features].fillna(df[clf_features].median())
                yc = df["risk_label"]
                tree = DecisionTreeClassifier(max_depth=4, random_state=42)
                tree.fit(Xc, yc)
                df["risk_pred"] = tree.predict(Xc)
            except Exception:
                df["risk_pred"] = df["risk_label"]
        else:
            df["risk_pred"] = df["risk_label"]
    else:
        # fallback rule-based risk_pred
        temp_series = df.get("temperature", pd.Series(np.nan, index=df.index))
        temp_mean = temp_series.mean() if hasattr(temp_series, "mean") else np.nan
        temp_std = temp_series.std() if hasattr(temp_series, "std") else np.nan
        thresh = temp_mean + 2 * temp_std if not np.isnan(temp_mean) and not np.isnan(temp_std) else np.nan
        cond_temp = (temp_series > thresh) if not np.isnan(thresh) else False
        df["risk_pred"] = np.where((df.get("anomaly_flag", 0) == 1) | cond_temp, 1, 0)

    # Health base score computation
    base = pd.Series(0.0, index=df.index)
    if "voltage_ma" in df.columns and df["voltage_ma"].notna().sum() > 0:
        vm = df["voltage_ma"].fillna(method="ffill").fillna(df["voltage"].median() if "voltage" in df.columns else 3.7)
        base += (vm.max() - vm)
    elif "voltage" in df.columns:
        v = df["voltage"].fillna(df["voltage"].median())
        base += (v.max() - v)
    else:
        base += 0.5

    if "temperature" in df.columns and df["temperature"].notna().sum() > 0:
        t = df["temperature"].fillna(df["temperature"].median())
        base += (t - t.min()) / 10.0

    base = base + df.get("anomaly_flag", 0)*1.0 + df.get("risk_pred", 0)*0.8

    trend_features = [f for f in ["voltage_ma", "voltage_var", "temp_ma", "cycle", "anomaly_flag"] if f in df.columns]
    if len(trend_features) >= 2 and df[trend_features].dropna().shape[0] >= 20:
        try:
            Xtr = df[trend_features].fillna(0)
            reg = LinearRegression()
            reg.fit(Xtr, base)
            hp = reg.predict(Xtr)
        except Exception:
            hp = base.values
    else:
        hp = base.values

    hp = np.array(hp, dtype=float)
    # normalize and compose final score
    hp_norm = (hp - hp.min()) / (hp.max() - hp.min() + 1e-9)
    health_component = 1 - hp_norm
    score = (0.6 * health_component) + (0.25 * (1 - df.get("risk_pred", 0))) + (0.15 * (1 - df.get("anomaly_flag", 0)))
    df["battery_health_score"] = (score * 100).clip(0, 100)
    return df

def recommend_action(row):
    score = row.get("battery_health_score", 50)
    rp = row.get("risk_pred", 0)
    an = row.get("anomaly_flag", 0)
    if score > 85 and rp == 0 and an == 0:
        return "Battery healthy. Normal operation."
    elif 70 < score <= 85:
        return "Monitor battery. Avoid deep discharge & fast charging."
    elif 50 < score <= 70:
        return "Limit fast charging. Allow cooling intervals."
    else:
        return "High risk! Reduce load & schedule maintenance."

def pack_health_label(score):
    if score >= 85:
        return "HEALTHY", "green"
    elif score >= 60:
        return "WATCH", "orange"
    else:
        return "CRITICAL", "red"

# --------------------------
# UI helpers
# --------------------------
def info_button(key, title, text):
    # use a deterministic unique key per info control
    btn_key = f"info_btn_{key}"
    if st.button("ℹ️ " + title, key=btn_key):
        st.session_state.setdefault(f"info_{key}", False)
        st.session_state[f"info_{key}"] = not st.session_state[f"info_{key}"]
    if st.session_state.get(f"info_{key}", False):
        st.info(text)

def gradient_progress(score):
    pct = float(np.clip(score, 0, 100))
    color = "green" if pct >= 85 else "orange" if pct >= 60 else "red"
    bar = f"""<div style='padding:6px;border-radius:8px;'>
      <div style='height:20px;width:100%;background:#111;border-radius:6px;'>
        <div style='width:{pct}%;height:20px;background:{color};border-radius:6px;'></div>
      </div>
      <div style='font-weight:700;padding-top:6px;'>{pct:.1f} / 100</div>
    </div>"""
    st.markdown(bar, unsafe_allow_html=True)

# --------------------------
# Main
# --------------------------
def main():
    st.title("CELLGUARD.AI - Robust Dashboard")
    st.write("A tolerant dashboard that adapts to messy or incomplete CSVs.")

    st.sidebar.header("Configuration")
    data_mode = st.sidebar.radio("Data source", ["Sample data", "Upload CSV"])
    contamination = st.sidebar.slider("Anomaly sensitivity", 0.01, 0.2, 0.05, 0.01)
    window = st.sidebar.slider("Rolling window", 5, 30, 10)
    st.sidebar.markdown("CSV column names can vary. App will adapt where possible.")

    if data_mode == "Sample data":
        df_raw = generate_sample_bms_data()
        st.success("Using simulated BMS data.")
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded is None:
            st.warning("Please upload a CSV file or use Sample data.")
            st.stop()
        # be defensive with encoding & read errors
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception:
            try:
                df_raw = pd.read_csv(uploaded, encoding="latin1")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                st.stop()
        st.success("CSV loaded.")

    df_raw, col_map = normalize_bms_columns(df_raw)
    if col_map:
        mapped_text = ", ".join([f'"{orig}" -> {target}' for target, orig in col_map.items()])
        st.info("Auto-mapped columns: " + mapped_text)
    else:
        st.warning("Could not auto-map standard columns. Proceeding with available fields.")

    required_logical = ["voltage", "current", "temperature", "soc", "cycle", "time"]
    df_raw = ensure_columns(df_raw, required_logical)

    with st.expander("Raw preview"):
        st.dataframe(df_raw.head(10))

    df_fe = feature_engineering(df_raw, window=window)
    df_out = build_models_and_scores(df_fe, contamination=contamination)
    df_out["recommendation"] = df_out.apply(recommend_action, axis=1)

    avg_score = float(df_out["battery_health_score"].mean()) if not df_out["battery_health_score"].isnull().all() else 50.0
    anomaly_pct = float(df_out["anomaly_flag"].mean() * 100) if "anomaly_flag" in df_out.columns else 0.0
    label, color = pack_health_label(avg_score)

    c1, c2 = st.columns([3,1])
    with c1:
        st.subheader("Pack Health Overview")
        gradient_progress(avg_score)
        st.markdown(f"**Status:** {label}")
        st.write(f"Avg Health Score: **{avg_score:.1f}** • Anomalies: **{anomaly_pct:.1f}%**")
    with c2:
        st.subheader("Quick actions")
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download processed CSV", csv_bytes, "CellGuardAI_Output.csv", "text/csv")

    st.markdown("---")

    tab_trad, tab_ai, tab_compare, tab_table = st.tabs(["Traditional BMS", "CellGuard.AI", "Compare", "Data"])

    with tab_trad:
        st.header("Traditional BMS Analysis")
        info_button("trad_overview", "Traditional BMS overview", "Traditional BMS shows instantaneous voltage, raw temperature, and SOC. It alarms on hard thresholds.")
        if df_out["voltage"].notna().sum() > 0:
            fig_v = px.line(df_out, x="time", y="voltage", labels={"time":"Time","voltage":"Voltage (V)"})
            st.plotly_chart(fig_v, use_container_width=True)
            info_button("trad_volt", "What voltage shows", "Voltage trend indicates sag under load and cell imbalance.")
        else:
            st.warning("Voltage data missing - cannot show voltage chart.")
        if df_out["temperature"].notna().sum() > 0:
            fig_t = px.line(df_out, x="time", y="temperature", labels={"time":"Time","temperature":"Temperature (C)"})
            st.plotly_chart(fig_t, use_container_width=True)
            info_button("trad_temp", "What temperature shows", "Temperature trend flags thermal drift/overheating.")
        else:
            st.warning("Temperature data missing - cannot show temperature chart.")
        if df_out["soc"].notna().sum() > 0:
            fig_soc = px.line(df_out, x="time", y="soc", labels={"time":"Time","soc":"State of Charge (%)"})
            st.plotly_chart(fig_soc, use_container_width=True)
            info_button("trad_soc", "What SOC shows", "SOC shows charge level and drift.")
        else:
            st.info("SOC not provided - traditional SOC-based alarms not available.")

    with tab_ai:
        st.header("CellGuard.AI - Pattern & Anomaly Analysis")
        info_button("ai_overview", "CellGuard AI overview", "Detects micro-patterns, trend deviations, and multivariate anomalies.")
        colL, colR = st.columns([2.5, 1])
        with colL:
            fig_health = px.area(df_out, x="time", y="battery_health_score", labels={"time":"Time","battery_health_score":"Health Score"})
            st.plotly_chart(fig_health, use_container_width=True)
            info_button("ai_health", "What the health chart says", "Shows a normalized health score (0-100). Drops indicate worsening combined signals.")
            if df_out["anomaly_flag"].sum() > 0:
                a_df = df_out[df_out["anomaly_flag"]==1]
                fig_h2 = go.Figure()
                fig_h2.add_trace(go.Scatter(x=df_out["time"], y=df_out["battery_health_score"], mode="lines", name="Health"))
                fig_h2.add_trace(go.Scatter(x=a_df["time"], y=a_df["battery_health_score"], mode="markers", name="Anomaly", marker=dict(color="red", size=8, symbol="x")))
                st.plotly_chart(fig_h2, use_container_width=True)
        with colR:
            st.subheader("Top Risks")
            worst = df_out.nsmallest(8, "battery_health_score")[["time","voltage","temperature","battery_health_score","anomaly_flag","risk_pred"]]
            st.table(worst.fillna("N/A"))

    with tab_compare:
        st.header("Compare Traditional vs CellGuard.AI")
        left, right = st.columns(2)
        with left:
            st.subheader("Traditional - threshold & instantaneous")
            st.markdown("- Relies on thresholds (low voltage, high temp).\\n- Deterministic and simple.")
            if df_out["voltage"].notna().sum() > 0:
                st.metric("Voltage mean", f"{df_out['voltage'].mean():.3f} V")
        with right:
            st.subheader("CellGuard.AI - trend-aware & multivariate")
            st.markdown("- Detects micro-patterns across signals.\\n- Produces continuous health score.")
            st.metric("Avg Health Score", f"{avg_score:.1f}")

    with tab_table:
        st.header("Processed Data & Export")
        st.dataframe(df_out.head(400), use_container_width=True)

    st.caption("CellGuard.AI - tolerant to messy files, clearer UI, and helpful explanations.")

if __name__ == "__main__":
    main()
