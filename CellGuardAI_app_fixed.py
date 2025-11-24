
# CellGuardAI_app_fixed.py
# Fixed and cleaned single-file Streamlit dashboard for CellGuard.AI
# This file is derived from the uploaded file and corrected for indentation issues.
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
import re

st.set_page_config(page_title="CellGuard.AI - Dashboard", layout="wide")

# --------------------------
# Data generators (scenarios)
# --------------------------
def generate_sample_bms_data(n=800, seed=42, scenario="Generic"):
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
        idx = np.random.choice(n, size=18, replace=False)
        voltage[idx] -= np.random.uniform(0.03, 0.08, size=len(idx))
        temperature[idx] += np.random.uniform(2, 5, size=len(idx))

    elif scenario == "EV":
        voltage = base_voltage + 0.03 * np.sin(t / 40) - 0.0005 * t / n + np.random.normal(0, 0.008, n)
        current = 2.5 + 0.4 * np.sin(t / 20) + np.random.normal(0, 0.07, n)
        temperature = base_temp + 4 * np.sin(t / 120) + 0.01 * (t / n) * 10 + np.random.normal(0, 0.5, n)
        soc = np.clip(90 - 20 * (t / n) + np.random.normal(0, 1.5, n), 0, 100)
        cycle = t // 10
        idx = np.random.choice(n, size=35, replace=False)
        voltage[idx] -= np.random.uniform(0.04, 0.12, size=len(idx))
        temperature[idx] += np.random.uniform(3, 8, size=len(idx))

    elif scenario == "Drone":
        voltage = base_voltage + 0.04 * np.sin(t / 30) + np.random.normal(0, 0.006, n)
        current = base_current + 0.6 * np.sin(t / 10) + np.random.normal(0, 0.2, n)
        temperature = base_temp + 2 * np.sin(t / 80) + np.random.normal(0, 0.4, n)
        soc = np.clip(85 + 6 * np.sin(t / 40) + np.random.normal(0, 2, n), 0, 100)
        cycle = t // 30
        spikes = np.random.choice(n, size=60, replace=False)
        current[spikes] += np.random.uniform(2.0, 6.0, size=len(spikes))
        dips = np.random.choice(n, size=30, replace=False)
        voltage[dips] -= np.random.uniform(0.06, 0.18, size=len(dips))

    elif scenario == "Phone":
        voltage = base_voltage + 0.02 * np.sin(t / 80) + np.random.normal(0, 0.002, n)
        current = 0.8 + 0.1 * np.sin(t / 60) + np.random.normal(0, 0.02, n)
        temperature = base_temp + 1.5 * np.sin(t / 120) + np.random.normal(0, 0.15, n)
        soc = np.clip(95 + 3 * np.sin(t / 160) + np.random.normal(0, 0.5, n), 0, 100)
        cycle = t // 200
        idx = np.random.choice(n, size=6, replace=False)
        voltage[idx] -= np.random.uniform(0.01, 0.03, size=len(idx))

    else:
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
# Helper functions (cleaned)
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
    a_df = df[df.get("anomaly_flag", 0) == 1]
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
        if "battery_health_score" in df.columns:
            worst = df.nsmallest(n, "battery_health_score")
            if "recommendation" in worst.columns:
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


# Safe wrapper for plotly charts
def safe_plotly_chart(fig, key, **kwargs):
    try:
        st.plotly_chart(fig, use_container_width=True, key=key, **kwargs)
    except Exception as e:
        st.warning(f"Chart '{key}' failed to render: {e}")


# Strong sanitizer for uploaded CSVs
def strong_sanitize_df_for_cellguard(df):
    """
    Aggressively clean df in-place and return cleaned copy.
    - Force-convert known columns to numeric
    - Remove additional header rows (rows containing column names repeated)
    - Strip units like 'V', 'A', '%' and commas
    - Report up to 50 offending unique strings per numeric column
    - Fill NaNs with median / forward-fill to keep downstream aggregations safe
    """
    df = df.copy()
    st.write("### Raw sample (first 6 rows)")
    st.write(df.head(6))
    st.write("### Raw dtypes")
    st.write(df.dtypes)

    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    # Quick heuristic: remove rows that look like repeated header rows
    def is_header_like(row):
        cnt = 0
        for col in df.columns:
            try:
                if str(row.get(col, "")).strip().lower() == str(col).strip().lower():
                    cnt += 1
            except Exception:
                pass
        return cnt >= max(2, len(df.columns)//4)
    header_mask = df.apply(is_header_like, axis=1)
    if header_mask.any():
        st.warning(f"Removed {header_mask.sum()} header-like rows that looked like repeated headings.")
        df = df.loc[~header_mask].reset_index(drop=True)

    # columns we care about (force these numeric)
    force_numeric = ["voltage","current","temperature","temp","soc","cycle","time"]
    # normalize column names to lowercase for matching
    col_map_lower = {c.lower(): c for c in df.columns}
    mapped = {}
    for want in force_numeric:
        for cname_lower, orig in col_map_lower.items():
            if want in cname_lower or (any(k in cname_lower for k in ["volt","vcell","packv"]) and "volt" in want):
                if want not in mapped:
                    mapped[want] = orig

    for core in ["voltage","current","temperature","soc","cycle","time"]:
        if core in df.columns and core not in mapped:
            mapped[core] = core

    st.write("Column mapping (for force-numeric):", mapped)

    # cleaning function for numeric-like strings
    def clean_num_str(s):
        if pd.isnull(s):
            return s
        s = str(s).strip()
        s = re.sub(r'(?i)\b(v|volts?|a|amps?|%|degc|c|khz|hz|mv|ma)\b', '', s)
        s = re.sub(r'\(.*?\)', '', s)
        s = re.sub(r'[^\d\.\-eE,]+', '', s)
        s = s.replace(',', '')
        s = s.strip()
        return s if s != '' else None

    for key, orig_col in mapped.items():
        try:
            raw_series = df[orig_col].astype(object)
        except Exception:
            continue

        cleaned = raw_series.map(clean_num_str)
        coerced = pd.to_numeric(cleaned, errors='coerce')

        bad_mask = coerced.isna() & raw_series.notna()
        if bad_mask.any():
            bad_vals = raw_series[bad_mask].astype(str).unique()[:50]
            st.write(f"Non-numeric examples in column `{orig_col}` (showing up to 50):")
            st.write(list(bad_vals))

        df[orig_col] = coerced

        if df[orig_col].notna().sum() > 0:
            median = df[orig_col].median(skipna=True)
            df[orig_col] = df[orig_col].fillna(median).ffill().bfill()
        else:
            df[orig_col] = df[orig_col].astype(float)

    st.write("### After force-conversion dtypes")
    st.write(df.dtypes)
    st.write("### Cleaned sample (first 6 rows)")
    st.write(df.head(6))
    return df


# --------------------------
# Main UI
# --------------------------
def main():
    st.title("CELLGUARD.AI — Dashboard")
    st.write("Predictive battery intelligence: health score, early alerts, anomaly timeline, and actionable recommendations.")

    # Sidebar config
    st.sidebar.header("Configuration")
    data_mode = st.sidebar.radio("Data source", ["Sample data", "Upload CSV"])
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
        # Apply the stronger sanitizer to uploaded CSV
        df_raw = strong_sanitize_df_for_cellguard(df_raw)

    # Normalize and feature engineering
    df_raw, col_map = normalize_bms_columns(df_raw)
    required_logical = ["voltage", "current", "temperature", "soc", "cycle", "time"]
    df_raw = ensure_columns(df_raw, required_logical)

    df_fe = feature_engineering(df_raw, window=window)
    df_out = build_models_and_scores(df_fe, contamination=contamination)
    df_out["recommendation"] = df_out.apply(recommend_action, axis=1)

    avg_score = float(df_out["battery_health_score"].mean()) if not df_out["battery_health_score"].isnull().all() else 50.0
    anomaly_pct = float(df_out["anomaly_flag"].mean() * 100) if "anomaly_flag" in df_out.columns else 0.0
    label, color = pack_health_label(avg_score)

    # Ensure expected UI variables exist
    alerts = simple_alerts_from_df(df_out)
    recs = build_recommendations_from_df(df_out, n=8)
    pdf_bytes = generate_pdf_report(df_out, avg_score, anomaly_pct, alerts, recs, "Auto-generated verdict")

    # Top header layout
    top_left, top_mid, top_right = st.columns([1.4, 1.4, 1])
    with top_left:
        st.markdown("### Battery Health")
        gauge = make_gauge(avg_score)
        safe_plotly_chart(gauge, key="gauge_health")
    with top_mid:
        st.markdown("### Pack Status")
        badge_color = "#2ecc71" if label=="HEALTHY" else ("#f39c12" if label=="WATCH" else "#e74c3c")
        st.markdown(render_colored_badge(label, color_hex=badge_color, text_color="#ffffff"), unsafe_allow_html=True)
        st.metric("Avg Health Score", f"{avg_score:.1f}/100", delta=f"{(avg_score-85):.1f} vs ideal")
        st.write(f"- Scenario: **{scenario}**")
        st.write(f"- Anomalies: **{anomaly_pct:.1f}%**")
        st.write(f"- Data points: **{len(df_out)}**")
        st.write(f"- Mapped columns: {', '.join(list(col_map.keys())) if col_map else 'auto-map not found'}")
    with top_right:
        st.markdown("### Actions")
        st.download_button("⬇️ Download processed CSV", df_out.to_csv(index=False).encode("utf-8"),
                           "CellGuardAI_Output.csv", "text/csv", key="download_processed_csv_header")

    # Combined verdict & PDF (placed after df_out is created)
    st.subheader("Combined Verdict and PDF")
    st.write("### Final Verdict")
    if avg_score < 60:
        st.error("Combined verdict: Immediate action required.")
    elif avg_score < 75:
        st.warning("Combined verdict: Monitor closely.")
    else:
        st.success("Combined verdict: Pack is healthy.")

    # Downloadable Reports
    st.download_button(
        label="⬇️ Download Processed CSV",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="CellGuardAI_Processed.csv",
        mime="text/csv",
        key="download_processed_csv"
    )
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name="CellGuardAI_Report.pdf",
        mime="application/pdf",
        key="download_pdf_report"
    )
    st.download_button(
        label="⬇️ Download Full Raw CSV",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="CellGuardAI_FullOutput.csv",
        mime="text/csv",
        key="download_full_csv"
    )

    st.markdown("### Predictive Alerts (main)")
    if alerts:
        for a in alerts:
            sev = a.get("severity", "info")
            if sev == "high":
                st.markdown(f"<div style='background:#fdecea;padding:8px;border-radius:8px;margin-bottom:6px'><b>🔴 {a.get('title','')}</b> — {a.get('detail','')}</div>", unsafe_allow_html=True)
            elif sev == "medium":
                st.markdown(f"<div style='background:#fff4e5;padding:8px;border-radius:8px;margin-bottom:6px'><b>🟠 {a.get('title','')}</b> — {a.get('detail','')}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background:#eef7ff;padding:8px;border-radius:8px;margin-bottom:6px'><b>🔵 {a.get('title','')}</b> — {a.get('detail','')}</div>", unsafe_allow_html=True)
    else:
        st.success("No immediate AI alerts")

    st.markdown("---")

    # Recommendations list
    rcol1, rcol2 = st.columns([1, 1])
    with rcol1:
        st.subheader("Recommendations (if any)")
        if recs:
            for r in recs:
                st.markdown(f"✅ **{r['recommendation']}** — seen in ({r['count']}) risky rows")
        else:
            st.write("No specific recommendations at this time.")
    with rcol2:
        st.subheader("Top Warnings Snapshot")
        if alerts:
            for a in alerts:
                st.markdown(f"- **{a['title']}** — {a['detail']}")
        else:
            st.write("No warnings.")

    st.markdown("---")

    # Summary metrics
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

    # Tabs
    tab_ai, tab_trad, tab_compare, tab_table = st.tabs(["CellGuard.AI", "Traditional BMS", "Compare (Combined)", "Data"])

    with tab_ai:
        st.subheader("AI-Based Battery Insights")
        # Health Score Timeline
        fig_h = px.line(df_out, x="time", y="battery_health_score", labels={"time":"Time","battery_health_score":"Health Score"}, title="Health Score Over Time")
        safe_plotly_chart(fig_h, key="ai_health_timeline")
        # voltage var
        fig_vv = px.line(df_out, x="time", y="voltage_var", labels={"time":"Time","voltage_var":"Voltage Variance"}, title="Voltage Variance")
        safe_plotly_chart(fig_vv, key="ai_voltage_var")
        # soc
        fig_soc_ai = px.line(df_out, x="time", y="soc", labels={"time":"Time","soc":"SOC (%)"}, title="SOC Trend")
        safe_plotly_chart(fig_soc_ai, key="ai_soc_chart")
        # current
        fig_cur = px.line(df_out, x="time", y="current", labels={"time":"Time","current":"Current (A)"}, title="Current Flow Over Time")
        safe_plotly_chart(fig_cur, key="ai_current_plot")
        # temp hist
        fig_temp_hist = px.histogram(df_out, x="temperature", labels={"temperature":"Temperature (°C)"}, title="Temperature Distribution")
        safe_plotly_chart(fig_temp_hist, key="ai_temp_hist")
        # corr
        corr = df_out[["voltage","current","temperature","soc","battery_health_score"]].corr()
        fig_corr = px.imshow(corr, text_auto=True, title="Parameter Correlation Heatmap")
        safe_plotly_chart(fig_corr, key="ai_corr_heatmap")

    with tab_trad:
        st.subheader("Traditional BMS Insights")
        fig_v = px.line(df_out, x="time", y="voltage", labels={"time":"Time","voltage":"Voltage (V)"}, title="Voltage Over Time")
        safe_plotly_chart(fig_v, key="trad_voltage_chart")
        fig_t = px.line(df_out, x="time", y="temperature", labels={"time":"Time","temperature":"Temperature (°C)"}, title="Temperature Over Time")
        safe_plotly_chart(fig_t, key="trad_temp_chart")
        fig_soc_trad = px.line(df_out, x="time", y="soc", labels={"time":"Time","soc":"SOC (%)"}, title="SOC Over Time (Traditional BMS)")
        safe_plotly_chart(fig_soc_trad, key="trad_soc_chart")

    with tab_compare:
        st.header("Compare — Combined result (CellGuard.AI first, then Traditional BMS)")
        st.markdown("### CellGuard.AI (Predictive view)")
        st.write(f"- Health Score: **{avg_score:.1f}/100**")
        st.write(f"- AI Anomaly %: **{anomaly_pct:.1f}%**")
        if alerts:
            st.write("- Current AI warnings:")
            for a in alerts:
                st.write(f"  - **{a['title']}** — {a['detail']}")
        else:
            st.write("- No AI warnings detected.")
        st.markdown("### Traditional BMS (Instant/raw view)")
        trad_cols = st.columns(3)
        with trad_cols[0]:
            if "voltage" in df_out.columns and df_out["voltage"].notna().sum()>0:
                st.metric("Voltage (mean)", f"{df_out['voltage'].mean():.3f} V")
            else:
                st.write("Voltage: N/A")
        with trad_cols[1]:
            if "temperature" in df_out.columns and df_out["temperature"].notna().sum()>0:
                st.metric("Temperature (mean)", f"{df_out['temperature'].mean():.2f} °C")
            else:
                st.write("Temperature: N/A")
        with trad_cols[2]:
            if "soc" in df_out.columns and df_out['soc'].notna().sum()>0:
                st.metric("SOC (last)", f"{df_out['soc'].iloc[-1]:.1f}%")
            else:
                st.write("SOC: N/A")
        st.markdown("---")
        st.subheader("Combined Recommendation")
        high_alerts = [a for a in alerts if a.get("severity")=="high"]
        if high_alerts or avg_score < 60:
            st.error("Combined verdict: Immediate action required. Reduce load, avoid fast charging, and schedule inspection.")
        elif avg_score < 75:
            st.warning("Combined verdict: Monitor closely. Apply conservative charge/discharge limits.")
        else:
            st.success("Combined verdict: Pack is healthy. Continue normal operation but monitor trends.")

    with tab_table:
        st.header("Processed Data & Export")
        st.download_button("⬇️ Download full report CSV", df_out.to_csv(index=False).encode("utf-8"), "CellGuardAI_FullReport.csv", "text/csv", key="download_full_report")
        st.dataframe(df_out.head(500), use_container_width=True)

    st.caption("CellGuard.AI — demo scenarios added: Generic, EV, Drone, Phone. Toggle scenarios in the sidebar to simulate different field conditions for judges and testing.")

if __name__ == "__main__":
    main()
