# app.py
# CellGuard.AI — Dashboard with demo dataset selector (EV / Drone / Phone presets)
# All prior features included: health gauge, alerts, extra charts, colored warnings, PDF export.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

st.set_page_config(page_title="CellGuard.AI - Dashboard", layout="wide")

# --------------------------
# Data generators (scenarios)
# --------------------------
def generate_sample_bms_data(n=800, seed=42, scenario="Generic"):
    """
    Scenarios:
    - Generic: balanced synthetic signals (default)
    - EV: higher cycles, gradual thermal drift, slow voltage sag
    - Drone: frequent high current spikes, abrupt voltage dips
    - Phone: low variance, stable small-signal behavior
    """
    np.random.seed(seed)
    t = np.arange(n)
    base_voltage = 3.7
    base_current = 1.5
    base_temp = 30.0
    soc_base = 80.0

    if scenario == "Generic":
        voltage = base_voltage + 0.05 * np.sin(t / 50) + np.random.normal(0, 0.005, n)
        current = base_current + 0.3 * np.sin(t / 30) + np.random.normal(0, 0.05, n)
        temperature = base_temp + 3 * np.sin(t / 60) + np.random.normal(0, 0.3, n)
        soc = np.clip(soc_base + 10 * np.sin(t / 80) + np.random.normal(0, 1, n), 0, 100)
        cycle = t // 50
        # occasional anomalies
        idx = np.random.choice(n, size=18, replace=False)
        voltage[idx] -= np.random.uniform(0.03, 0.08, size=len(idx))
        temperature[idx] += np.random.uniform(2, 5, size=len(idx))

    elif scenario == "EV":
        # EV packs: high cycles, gradual thermal drift, slow voltage sag & higher variance
        voltage = base_voltage + 0.03 * np.sin(t / 40) - 0.0005 * t/ n + np.random.normal(0, 0.008, n)
        current = 2.5 + 0.4 * np.sin(t / 20) + np.random.normal(0, 0.07, n)
        temperature = base_temp + 4 * np.sin(t / 120) + 0.01 * (t / n) * 10 + np.random.normal(0, 0.5, n)
        soc = np.clip(90 - 20*(t/n) + np.random.normal(0, 1.5, n), 0, 100)
        cycle = t // 10  # faster cycling
        idx = np.random.choice(n, size=35, replace=False)
        voltage[idx] -= np.random.uniform(0.04, 0.12, size=len(idx))
        temperature[idx] += np.random.uniform(3, 8, size=len(idx))

    elif scenario == "Drone":
        # Drone packs: short bursts, frequent high current spikes, abrupt voltage dips
        voltage = base_voltage + 0.04 * np.sin(t / 30) + np.random.normal(0, 0.006, n)
        current = base_current + 0.6 * np.sin(t / 10) + np.random.normal(0, 0.2, n)
        temperature = base_temp + 2 * np.sin(t / 80) + np.random.normal(0, 0.4, n)
        soc = np.clip(85 + 6 * np.sin(t / 40) + np.random.normal(0, 2, n), 0, 100)
        cycle = t // 30
        # more spikes
        spikes = np.random.choice(n, size=60, replace=False)
        current[spikes] += np.random.uniform(2.0, 6.0, size=len(spikes))
        # abrupt dips
        dips = np.random.choice(n, size=30, replace=False)
        voltage[dips] -= np.random.uniform(0.06, 0.18, size=len(dips))

    elif scenario == "Phone":
        # Phone packs: highly stable, low variance, few anomalies
        voltage = base_voltage + 0.02 * np.sin(t / 80) + np.random.normal(0, 0.002, n)
        current = 0.8 + 0.1 * np.sin(t / 60) + np.random.normal(0, 0.02, n)
        temperature = base_temp + 1.5 * np.sin(t / 120) + np.random.normal(0, 0.15, n)
        soc = np.clip(95 + 3 * np.sin(t / 160) + np.random.normal(0, 0.5, n), 0, 100)
        cycle = t // 200
        idx = np.random.choice(n, size=6, replace=False)
        voltage[idx] -= np.random.uniform(0.01, 0.03, size=len(idx))

    else:
        # fallback to generic
        return generate_sample_bms_data(n=n, seed=seed, scenario="Generic")

    return pd.DataFrame({
        "time": t,
        "voltage": voltage,
        "current": current,
        "temperature": temperature,
        "soc": soc,
        "cycle": cycle
    })


# --------------------------
# (The rest of the code is same as previous app version, unchanged logic)
# For brevity I include the full app code but keep the same helper functions, ML, UI,
# PDF generation, warnings, recommendations, and tabs.
# --------------------------

# ... (reusing functions: normalize_bms_columns, ensure_columns, feature_engineering,
# build_models_and_scores, recommend_action, pack_health_label, make_gauge,
# anomaly_marker_trace, simple_alerts_from_df, build_recommendations_from_df,
# render_colored_badge, generate_pdf_report)
#
# To keep this snippet focused, paste the rest of your existing app code here
# exactly as you had it (feature engineering, UI helpers, PDF generator, main()).
#
# But since you asked to "do it in app.py", below is the complete continuation:
# (Full app code continues — identical to the previous feature-complete version,
# but uses generate_sample_bms_data(scenario=...) where Sample data is selected.)

# --------------------------
# Reusing the helper functions from prior app (copy/paste)
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
    else:
        df["temp_ma"] = np.nan
        df["temp_roc"] = np.nan

    if df["soc"].notna().sum() > 0:
        df["soc_ma"] = df["soc"].rolling(window, min_periods=1).mean()
        df["soc_roc"] = df["soc"].diff().fillna(0)
    else:
        df["soc_ma"] = np.nan
        df["soc_roc"] = np.nan

    if df["voltage"].notna().sum() > 0:
        volt_drop_threshold = -0.03
        conditions = pd.Series(False, index=df.index)
        if df["temperature"].notna().sum() > 0:
            temp_mean = df["temperature"].mean()
            temp_std = df["temperature"].std()
            temp_threshold = temp_mean + 2 * temp_std if not np.isnan(temp_mean) and not np.isnan(temp_std) else np.nan
            if not np.isnan(temp_threshold):
                conditions = conditions | (df["temperature"] > temp_threshold)
        if "voltage_roc" in df.columns:
            conditions = conditions | (df["voltage_roc"] < volt_drop_threshold)
        if "soc_roc" in df.columns:
            conditions = conditions | (df["soc_roc"] < -5)
        df["risk_label"] = np.where(conditions, 1, 0)
    else:
        df["risk_label"] = 0

    return df

def build_models_and_scores(df, contamination=0.05):
    df = df.copy()
    possible = ["voltage", "current", "temperature", "soc", "voltage_ma", "voltage_roc", "soc_roc", "voltage_var", "temp_ma", "cycle"]
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
        return "Healthy — normal operation."
    elif 70 < score <= 85:
        return "Watch — avoid deep discharge & fast-charge this cycle."
    elif 50 < score <= 70:
        return "Caution — restrict fast charging; allow cooling intervals."
    else:
        return "Critical — reduce load, stop fast charging, schedule inspection."

def pack_health_label(score):
    if score >= 85:
        return "HEALTHY", "green"
    elif score >= 60:
        return "WATCH", "orange"
    else:
        return "CRITICAL", "red"

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
    a_df = df[df["anomaly_flag"] == 1]
    if a_df.empty:
        return None
    return go.Scatter(x=a_df["time"], y=a_df["battery_health_score"], mode="markers", name="Anomaly", marker=dict(color="red", size=8, symbol="x"))

def simple_alerts_from_df(df):
    alerts = []
    if "temperature" in df.columns and df["temperature"].notna().sum()>0:
        temp_mean = df["temperature"].mean()
        temp_std = df["temperature"].std()
        recent_temp = df["temperature"].iloc[-1]
        if recent_temp > (temp_mean + 2*temp_std):
            alerts.append({"title":"Thermal drift", "detail":"Temp well above normal — hotspot risk. Cool & inspect.", "severity":"high"})
    if "voltage_roc" in df.columns and "voltage_var" in df.columns:
        last_roc = df["voltage_roc"].rolling(5).mean().iloc[-1]
        last_var = df["voltage_var"].rolling(10).mean().iloc[-1]
        if last_roc < -0.01:
            alerts.append({"title":"Voltage sag pattern", "detail":"Sustained negative voltage change — internal resistance rising.", "severity":"medium"})
        if last_var > df["voltage_var"].mean() + df["voltage_var"].std():
            alerts.append({"title":"Voltage variance rising", "detail":"Cell-to-cell variance increasing — imbalance risk.", "severity":"medium"})
    if "current" in df.columns and df["current"].notna().sum()>0:
        spike_pct = (df["current"] > (df["current"].mean() + 2*df["current"].std())).mean()
        if spike_pct > 0.02:
            alerts.append({"title":"Current spikes", "detail":"Frequent high current spikes — mechanical/connection stress likely.", "severity":"medium"})
    if "anomaly_flag" in df.columns:
        p = df["anomaly_flag"].mean()
        if p > 0.05:
            alerts.append({"title":"Anomaly rate high", "detail":f"{p*100:.1f}% readings flagged — investigate.", "severity":"medium"})
    if "risk_pred" in df.columns and df["risk_pred"].iloc[-1]==1:
        alerts.append({"title":"Immediate risk", "detail":"Model predicts elevated risk on latest measurement.", "severity":"high"})
    return alerts

def build_recommendations_from_df(df, n=5):
    recs = []
    try:
        worst = df.nsmallest(n, "battery_health_score")
        rec_counts = worst["recommendation"].value_counts()
        for rec, cnt in rec_counts.items():
            recs.append({"recommendation": rec, "count": int(cnt)})
    except Exception:
        pass
    return recs

def render_colored_badge(text, color_hex="#eeeeee", text_color="#000000"):
    html = f"<span style='background:{color_hex};color:{text_color};padding:6px 10px;border-radius:8px;font-weight:600'>{text}</span>"
    return html

def generate_pdf_report(df_out, avg_score, anomaly_pct, alerts, recs, verdict_text):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    margin = 18*mm
    x = margin
    y = height - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "CellGuard.AI — Diagnostic Report")
    y -= 8*mm

    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"Avg Health Score: {avg_score:.1f}/100")
    c.drawString(x + 80*mm, y, f"Anomaly Rate: {anomaly_pct:.2f}%")
    y -= 6*mm
    c.drawString(x, y, f"Data points: {len(df_out)}")
    y -= 8*mm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Combined Verdict:")
    c.setFont("Helvetica", 11)
    c.drawString(x + 30, y, verdict_text)
    y -= 10*mm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(x, y, "Top AI Alerts:")
    y -= 6*mm
    c.setFont("Helvetica", 10)
    if alerts:
        for a in alerts[:6]:
            c.drawString(x + 6, y, f"- {a['title']}: {a['detail']}")
            y -= 5*mm
            if y < margin + 40*mm:
                c.showPage()
                y = height - margin
                c.setFont("Helvetica", 10)
    else:
        c.drawString(x + 6, y, "- None")
        y -= 6*mm

    y -= 4*mm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x, y, "Top Recommendations:")
    y -= 6*mm
    c.setFont("Helvetica", 10)
    if recs:
        for r in recs[:6]:
            c.drawString(x + 6, y, f"- {r['recommendation']} (observed {r['count']} times)")
            y -= 5*mm
            if y < margin + 20*mm:
                c.showPage()
                y = height - margin
                c.setFont("Helvetica", 10)
    else:
        c.drawString(x + 6, y, "- None")
        y -= 6*mm

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(margin, margin, "Generated by CellGuard.AI")
    c.save()
    buf.seek(0)
    return buf.read()

# --------------------------
# Main UI
# --------------------------
def main():
    st.title("CELLGUARD.AI — Dashboard")
    st.write("Predictive battery intelligence: health score, early alerts, anomaly timeline, and actionable recommendations.")

    # Sidebar config
    st.sidebar.header("Configuration")
    data_mode = st.sidebar.radio("Data source", ["Sample data", "Upload CSV"])
    # New: scenario picker when using sample data
    scenario = st.sidebar.selectbox("Demo scenario (if Sample data)", ["Generic", "EV", "Drone", "Phone"])
    contamination = st.sidebar.slider("Anomaly sensitivity", 0.01, 0.2, 0.05, 0.01)
    window = st.sidebar.slider("Rolling window", 5, 30, 10)
    st.sidebar.markdown("Tip: upload CSV with columns like voltage, temperature, current, soc, time.")

    # Data load
    if data_mode == "Sample data":
        df_raw = generate_sample_bms_data(n=800, seed=42, scenario=scenario)
        st.sidebar.success(f"Using simulated data: {scenario}")
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

    # process
    df_fe = feature_engineering(df_raw, window=window)
    df_out = build_models_and_scores(df_fe, contamination=contamination)
    df_out["recommendation"] = df_out.apply(recommend_action, axis=1)

    avg_score = float(df_out["battery_health_score"].mean()) if not df_out["battery_health_score"].isnull().all() else 50.0
    anomaly_pct = float(df_out["anomaly_flag"].mean() * 100) if "anomaly_flag" in df_out.columns else 0.0
    label, color = pack_health_label(avg_score)

    alerts = simple_alerts_from_df(df_out)
    recs = build_recommendations_from_df(df_out, n=8)

    # Top header layout
top_left, top_mid, top_right = st.columns([1.4, 1.4, 1])
with top_left:
    st.markdown("### Battery Health")
    gauge = make_gauge(avg_score)
    st.plotly_chart(gauge, use_container_width=True, key="gauge_main")

with top_mid:
    st.markdown("### Pack Status")
    badge_color = "#2ecc71" if label == "HEALTHY" else ("#f39c12" if label == "WATCH" else "#e74c3c")
    st.markdown(render_colored_badge(label, color_hex=badge_color, text_color="#ffffff"), unsafe_allow_html=True)
    st.metric("Avg Health Score", f"{avg_score:.1f}/100", delta=f"{(avg_score-85):.1f} vs ideal")
    st.markdown("#### Quick summary")
    st.write(f"- Scenario: **{scenario}**")
    st.write(f"- Anomalies: **{anomaly_pct:.1f}%**")
    st.write(f"- Data points: **{len(df_out)}**")
    st.write(f"- Mapped columns: {', '.join(list(col_map.keys())) if col_map else 'auto-map not found'}")

with top_right:
    st.markdown("### Actions")
    st.download_button("⬇️ Download processed CSV", df_out.to_csv(index=False).encode("utf-8"), "CellGuardAI_Output.csv", "text/csv")

    high_alerts = [a for a in alerts if a.get("severity") == "high"]
    if high_alerts or avg_score < 60:
        verdict_text = "Immediate action required: reduce load, stop fast charging, schedule inspection."
        st.error("Combined verdict: Immediate action required.")
    elif avg_score < 75:
        verdict_text = "Monitor closely: use conservative charge/discharge limits and inspect soon."
        st.warning("Combined verdict: Monitor closely.")
    else:
        verdict_text = "Pack healthy: continue normal operation and keep monitoring trends."
        st.success("Combined verdict: Pack healthy.")

    pdf_bytes = generate_pdf_report(df_out, avg_score, anomaly_pct, alerts, recs, verdict_text)
    st.download_button("📄 Download PDF report", pdf_bytes, "CellGuardAI_Report.pdf", "application/pdf")

    st.markdown("### Predictive Alerts (main)")
    if alerts:
        for i, a in enumerate(alerts):
            sev = a.get("severity", "info")
            bg = "#fdecea" if sev=="high" else ("#fff4e5" if sev=="medium" else "#eef7ff")
            icon = "🔴" if sev=="high" else ("🟠" if sev=="medium" else "🔵")
            st.markdown(
                f"<div style='background:{bg};padding:8px;border-radius:8px;margin-bottom:6px'><b>{icon} {a['title']}</b> — {a['detail']}</div>",
                unsafe_allow_html=True
            )
    else:
        st.success("No immediate AI alerts")

st.markdown("---")

# Recommendations and Warnings
rcol1, rcol2 = st.columns([1, 1])
with rcol1:
    st.subheader("Recommendations (if any)")
    if recs:
        for idx, r in enumerate(recs):
            st.markdown(f"✅ **{r['recommendation']}** — seen in {r['count']} risky rows")
    else:
        st.write("No specific recommendations at this time.")

with rcol2:
    st.subheader("Top Warnings Snapshot")
    if alerts:
        for idx, a in enumerate(alerts):
            st.markdown(f"• **{a['title']}** — {a['detail']}")
    else:
        st.write("No warnings.")

st.markdown("---")

# Summary Metrics
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
    st.metric("Last Risk Pred", "HIGH" if df_out["risk_pred"].iloc[-1] == 1 else "NORMAL")

st.markdown("---")

# Tabs
tab_ai, tab_trad, tab_compare, tab_table = st.tabs(["CellGuard.AI", "Traditional BMS", "Compare (Combined)", "Data"])

############################
# CellGuard.AI TAB
############################
with tab_ai:
    st.header("CellGuard.AI — predictions, trends, and alerts (expanded)")

    # Health Trend
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=df_out["time"], y=df_out["battery_health_score"], mode="lines", name="Health Score"))
    m = anomaly_marker_trace(df_out)
    if m is not None:
        fig_h.add_trace(m)
    st.plotly_chart(fig_h, use_container_width=True, key="health_trend_chart")

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        if "voltage_var" in df_out.columns:
            fig_vv = px.line(df_out, x="time", y="voltage_var")
            st.plotly_chart(fig_vv, use_container_width=True, key="voltage_variance_chart")

        if "soc" in df_out.columns:
            fig_soc_ai = px.line(df_out, x="time", y="soc")
            st.plotly_chart(fig_soc_ai, use_container_width=True, key="soc_ai_chart")

    with pcol2:
        if "current" in df_out.columns:
            fig_cur = px.line(df_out, x="time", y="current")
            st.plotly_chart(fig_cur, use_container_width=True, key="current_chart")

        if "temperature" in df_out.columns:
            fig_temp_hist = px.histogram(df_out, x="temperature", nbins=30)
            st.plotly_chart(fig_temp_hist, use_container_width=True, key="temp_hist_chart")

    # Correlation matrix
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
    corr_cols = [c for c in ["voltage","current","temperature","soc","voltage_var","voltage_roc","battery_health_score"] if c in numeric_cols]
    if len(corr_cols) >= 2:
        corr = df_out[corr_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True, key="corr_matrix_chart")

    with st.expander("Open / Close — Top 8 Riskiest Moments (toggle)"):
        worst = df_out.nsmallest(8, "battery_health_score")[["time","voltage","temperature","battery_health_score","anomaly_flag","risk_pred","recommendation"]]
        st.table(worst.fillna("N/A"))

############################
# TRADITIONAL BMS TAB
############################
with tab_trad:
    st.header("Traditional BMS — raw signals")

    if df_out["voltage"].notna().sum()>0:
        fig_v = px.line(df_out, x="time", y="voltage")
        st.plotly_chart(fig_v, use_container_width=True, key="trad_voltage_chart")

    if df_out["temperature"].notna().sum()>0:
        fig_t = px.line(df_out, x="time", y="temperature")
        st.plotly_chart(fig_t, use_container_width=True, key="trad_temp_chart")

    fig_soc_trad = px.line(df_out, x="time", y="soc")
    st.plotly_chart(fig_soc_trad, use_container_width=True, key="trad_soc_chart")

############################
# COMPARE TAB
############################
with tab_compare:
    st.header("Compare — Combined result")

    st.markdown("### CellGuard.AI (Predictive View)")
    st.write(f"- Health Score: **{avg_score:.1f}/100**")
    st.write(f"- AI Anomaly %: **{anomaly_pct:.1f}%**")

    if alerts:
        st.write("- Current AI warnings:")
        for a in alerts:
            st.write(f"  - **{a['title']}** — {a['detail']}")
    else:
        st.write("- No AI warnings")

    st.markdown("### Traditional BMS (Instant View)")
    trad_cols = st.columns(3)
    with trad_cols[0]:
        st.metric("Voltage (mean)", f"{df_out['voltage'].mean():.3f} V")
    with trad_cols[1]:
        st.metric("Temperature (mean)", f"{df_out['temperature'].mean():.2f} °C")
    with trad_cols[2]:
        st.metric("SOC (last)", f"{df_out['soc'].iloc[-1]:.1f}%")

    st.markdown("---")
    st.subheader("Combined Recommendation")

    if high_alerts or avg_score < 60:
        st.error("Immediate action required.")
    elif avg_score < 75:
        st.warning("Monitor closely.")
    else:
        st.success("Pack healthy.")

############################
# DATA TAB
############################
with tab_table:
    st.header("Processed Data & Export")
    st.download_button("⬇️ Download full report CSV", df_out.to_csv(index=False).encode("utf-8"), "CellGuardAI_FullReport.csv", "text/csv")
    st.dataframe(df_out.head(500), use_container_width=True)

st.caption("CellGuard.AI — now with fixed Streamlit IDs, stable charts, and robust UI.")

            
