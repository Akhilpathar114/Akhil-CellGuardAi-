# app.py
# CellGuard.AI - Enhanced Dashboard (includes Health Gauge, Alerts, Interpretations)
# Based on user's transformed file (defensive + deploy-ready). See original reference. 1

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="CellGuard.AI - Dashboard", layout="wide")

# --------------------------
# Utilities (kept from prior robust version)
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
    if df["voltage"].notna().sum() > 0:
        df["voltage_ma"] = df["voltage"].rolling(window, min_periods=1).mean()
        df["voltage_roc"] = df["voltage"].diff().fillna(0)
        df["voltage_var"] = df["voltage"].rolling(window, min_periods=1).var().fillna(0)
    else:
        df["voltage_ma"] = np.nan
        df["voltage_roc"] = np.nan
        df["voltage_var"] = np.nan

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

    if df["voltage"].notna().sum() > 0:
        volt_drop_threshold = -0.03
        conditions = pd.Series(False, index=df.index)
        if df["temperature"].notna().sum() > 0 and not np.isnan(temp_threshold):
            conditions = conditions | (df["temperature"] > temp_threshold)
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

    if len(anomaly_features) >= 2 and df[anomaly_features].dropna().shape[0] >= 30:
        try:
            iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
            X = df[anomaly_features].fillna(df[anomaly_features].median())
            iso.fit(X)
            df["anomaly_flag"] = iso.predict(X).map({1: 0, -1: 1})
        except Exception:
            df["anomaly_flag"] = 0

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
        temp_series = df.get("temperature", pd.Series(np.nan, index=df.index))
        temp_mean = temp_series.mean() if hasattr(temp_series, "mean") else np.nan
        temp_std = temp_series.std() if hasattr(temp_series, "std") else np.nan
        thresh = temp_mean + 2 * temp_std if not np.isnan(temp_mean) and not np.isnan(temp_std) else np.nan
        cond_temp = (temp_series > thresh) if not np.isnan(thresh) else False
        df["risk_pred"] = np.where((df.get("anomaly_flag", 0) == 1) | cond_temp, 1, 0)

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
# UI helpers & visual widgets
# --------------------------
def make_gauge(score):
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(score),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Battery Health Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 60], 'color': "lightcoral"},
                {'range': [60, 85], 'color': "gold"},
                {'range': [85, 100], 'color': "lightgreen"},
            ],
        }
    ))
    gauge.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0))
    return gauge

def anomaly_marker_trace(df):
    # return scatter trace of anomalies (time vs health) for overlay
    a_df = df[df["anomaly_flag"] == 1]
    if a_df.empty:
        return None
    return go.Scatter(x=a_df["time"], y=a_df["battery_health_score"], mode="markers", name="Anomaly", marker=dict(color="red", size=8, symbol="x"))

def simple_alerts_from_df(df):
    alerts = []
    # thermal drift alert
    if "temperature" in df.columns and df["temperature"].notna().sum()>0:
        temp_mean = df["temperature"].mean()
        temp_std = df["temperature"].std()
        recent_temp = df["temperature"].iloc[-1]
        if recent_temp > (temp_mean + 2*temp_std):
            alerts.append(("Thermal drift", "Recent temperature >> historical mean. Hotspot risk."))
    # imbalance pattern (look at voltage_roc negative trend / increasing variance)
    if "voltage_roc" in df.columns and "voltage_var" in df.columns:
        last_roc = df["voltage_roc"].rolling(5).mean().iloc[-1]
        last_var = df["voltage_var"].rolling(10).mean().iloc[-1]
        if last_roc < -0.01:
            alerts.append(("Voltage sag pattern", "Sustained negative voltage ROC — internal resistance may be rising."))
        if last_var > df["voltage_var"].mean() + df["voltage_var"].std():
            alerts.append(("Voltage variance rising", "Cell-to-cell variance increasing — watch for imbalance."))
    # anomaly percent
    if "anomaly_flag" in df.columns:
        p = df["anomaly_flag"].mean()
        if p > 0.05:
            alerts.append(("Anomaly rate high", f"{p*100:.1f}% of recent readings flagged as anomalies."))
    # risk_pred
    if "risk_pred" in df.columns and df["risk_pred"].iloc[-1]==1:
        alerts.append(("Immediate risk prediction", "Model predicts elevated risk at last timestep."))
    return alerts

# --------------------------
# Main UI
# --------------------------
def main():
    st.title("CELLGUARD.AI — Dashboard")
    st.write("Predictive battery intelligence: health score, early alerts, anomalies, and clear actions.")

    # Sidebar config
    st.sidebar.header("Configuration")
    data_mode = st.sidebar.radio("Data source", ["Sample data", "Upload CSV"])
    contamination = st.sidebar.slider("Anomaly sensitivity", 0.01, 0.2, 0.05, 0.01)
    window = st.sidebar.slider("Rolling window", 5, 30, 10)
    st.sidebar.markdown("Tip: upload CSV with columns like voltage, temperature, current, soc, time.")

    # Data load
    if data_mode == "Sample data":
        df_raw = generate_sample_bms_data()
        st.sidebar.success("Using simulated data")
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded is None:
            st.warning("Upload a CSV or choose Sample data.")
            st.stop()
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception:
            try:
                df_raw = pd.read_csv(uploaded, encoding="latin1")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                st.stop()
        st.sidebar.success("CSV loaded.")

    df_raw, col_map = normalize_bms_columns(df_raw)
    required_logical = ["voltage", "current", "temperature", "soc", "cycle", "time"]
    df_raw = ensure_columns(df_raw, required_logical)

    # small header row: health gauge + status + download
    df_fe = feature_engineering(df_raw, window=window)
    df_out = build_models_and_scores(df_fe, contamination=contamination)
    df_out["recommendation"] = df_out.apply(recommend_action, axis=1)

    avg_score = float(df_out["battery_health_score"].mean()) if not df_out["battery_health_score"].isnull().all() else 50.0
    anomaly_pct = float(df_out["anomaly_flag"].mean() * 100) if "anomaly_flag" in df_out.columns else 0.0
    label, color = pack_health_label(avg_score)

    # Top header layout
    top_left, top_mid, top_right = st.columns([1.4, 1.4, 1])
    with top_left:
        st.markdown("### Battery Health")
        gauge = make_gauge(avg_score)
        st.plotly_chart(gauge, use_container_width=True)
    with top_mid:
        st.markdown("### Pack Status")
        st.markdown(f"**{label}**")
        st.metric("Avg Health Score", f"{avg_score:.1f}/100", delta=f"{(avg_score-85):.1f} vs ideal")
        st.markdown("#### Quick summary")
        st.write(f"- Anomalies: **{anomaly_pct:.1f}%**")
        st.write(f"- Data points: **{len(df_out)}**")
        st.write(f"- Mapped columns: {', '.join(list(col_map.keys())) if col_map else 'auto-map not found'}")
    with top_right:
        st.markdown("### Actions")
        st.download_button("⬇️ Download processed CSV", df_out.to_csv(index=False).encode("utf-8"), "CellGuardAI_Output.csv", "text/csv")
        st.button("📄 Generate PDF report (placeholder)")  # placeholder; can implement PDF later
        st.markdown("### Predictive alerts")
        alerts = simple_alerts_from_df(df_out)
        if alerts:
            for a in alerts:
                st.warning(f"**{a[0]}** — {a[1]}")
        else:
            st.success("No immediate alerts from AI")

    st.markdown("---")

    # Summary metrics row for quick glance
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.metric("Avg Temp (°C)", f"{df_out['temperature'].mean():.2f}" if df_out['temperature'].notna().sum()>0 else "N/A")
    with s2:
        st.metric("Voltage Var (mean)", f"{df_out['voltage_var'].mean():.4f}" if "voltage_var" in df_out.columns else "N/A")
    with s3:
        st.metric("Cycle Count (max)", f"{int(df_out['cycle'].max())}" if df_out['cycle'].notna().sum()>0 else "N/A")
    with s4:
        st.metric("Anomaly %", f"{anomaly_pct:.2f}%")
    with s5:
        st.metric("Last Risk Pred", "HIGH" if df_out["risk_pred"].iloc[-1]==1 else "NORMAL")

    st.markdown("---")

    # Main tabs: Traditional, CellGuard.AI, Compare, Data (expanded visuals + interpretation)
    tab_trad, tab_ai, tab_compare, tab_table = st.tabs(["Traditional BMS", "CellGuard.AI", "Compare", "Data"])

    with tab_trad:
        st.header("Traditional BMS — raw signals")
        cols = st.columns(1)
        # Voltage chart with anomaly markers
        if df_out["voltage"].notna().sum()>0:
            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(x=df_out["time"], y=df_out["voltage"], mode="lines", name="Voltage (V)"))
            m = anomaly_marker_trace(df_out)
            if m is not None:
                fig_v.add_trace(m)
            fig_v.update_layout(height=320, margin=dict(t=30))
            st.plotly_chart(fig_v, use_container_width=True)
            st.caption("Interpretation: voltage oscillations and sudden dips often indicate rising internal resistance or cell imbalance.")
        else:
            st.warning("Voltage data not available.")

        # Temperature chart
        if df_out["temperature"].notna().sum()>0:
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(x=df_out["time"], y=df_out["temperature"], mode="lines", name="Temperature (°C)"))
            m = anomaly_marker_trace(df_out)
            if m is not None:
                fig_t.add_trace(m)
            fig_t.update_layout(height=280, margin=dict(t=10))
            st.plotly_chart(fig_t, use_container_width=True)
            st.caption("Interpretation: rising baseline or spikes can mean hotspots; sustained rises are especially concerning.")
        else:
            st.info("Temperature data not available.")

    with tab_ai:
        st.header("CellGuard.AI — predictions, trends, and alerts")
        left, right = st.columns([2, 1])
        with left:
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(x=df_out["time"], y=df_out["battery_health_score"], mode="lines", name="Health Score"))
            m = anomaly_marker_trace(df_out)
            if m is not None:
                fig_h.add_trace(m)
            fig_h.update_layout(height=360, margin=dict(t=30))
            st.plotly_chart(fig_h, use_container_width=True)
            st.caption("Interpretation: declines in health score are composite signals — often a combination of voltage sag, temp drift, and anomalies.")
            # show some detected anomalies in a listed form
            detected = df_out[df_out["anomaly_flag"]==1].tail(8)
            if not detected.empty:
                st.subheader("Recent Anomalies (sample)")
                st.table(detected[["time","voltage","temperature","battery_health_score"]].fillna("N/A"))
        with right:
            st.subheader("Top Risks & Actions")
            worst = df_out.nsmallest(8, "battery_health_score")[["time","voltage","temperature","battery_health_score","anomaly_flag","risk_pred","recommendation"]]
            st.table(worst.fillna("N/A"))
            if not worst.empty:
                # quick recommended combined action (aggregate)
                rec = worst["recommendation"].value_counts().idxmax()
                st.markdown("### Combined suggestion")
                st.info(rec)

    with tab_compare:
        st.header("Traditional vs CellGuard.AI — why upgrade?")
        left, right = st.columns(2)
        with left:
            st.subheader("Traditional BMS (instant thresholds)")
            st.markdown("- Raw readings, on/off alarms\n- Requires manual log review")
            if df_out["voltage"].notna().sum()>0:
                st.metric("Voltage mean", f"{df_out['voltage'].mean():.3f} V")
        with right:
            st.subheader("CellGuard.AI (trend-aware & predictive)")
            st.markdown("- Detects micro-patterns and anomalies\n- Produces continuous health score and recommendations")
            st.metric("Avg Health Score", f"{avg_score:.1f}")

    with tab_table:
        st.header("Processed Data & Export")
        st.download_button("⬇️ Download full report CSV", df_out.to_csv(index=False).encode("utf-8"), "CellGuardAI_FullReport.csv", "text/csv")
        st.dataframe(df_out.head(500), use_container_width=True)

    st.caption("CellGuard.AI — shows health score, predictive alerts, anomaly timeline, and simple actions. Use the CSV export to archive results or send to technicians.")

if __name__ == "__main__":
    main()
