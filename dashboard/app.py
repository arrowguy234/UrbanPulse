import os
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta

# Optional (only used on Risk page)
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

DB_PATH = "urbanpulse.db"
DEMO_CSV = "data/demo/urbanpulse_demo.csv"

st.set_page_config(page_title="UrbanPulse • Smart City Platform", layout="wide")

# -----------------------------
# Database helpers
# -----------------------------
def read_demo_csv():
    if not os.path.exists(DEMO_CSV):
        st.error("Demo dataset not found. Please add: data/demo/urbanpulse_demo.csv")
        st.stop()
    df = pd.read_csv(DEMO_CSV)
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    return df

def expand_demo_if_too_small(df):
    # If demo file has too few sensors, create "virtual sensors" for a richer demo
    sensors = df["sensor_id"].unique().tolist()
    if len(sensors) >= 6:
        return df

    base = df.copy()
    out = [base]

    # create 9 sensors total
    target_sensors = ["S_0","S_1","S_2","S_3","S_4","S_5","S_6","S_7","S_8"]
    base_sensor = sensors[0] if len(sensors) > 0 else "S_0"

    for i, sid in enumerate(target_sensors):
        if sid == base_sensor:
            continue
        tmp = base.copy()
        tmp["sensor_id"] = sid

        # add small noise so leaderboards look realistic
        # keep non-negative
        noise = np.random.normal(loc=0.0, scale=1.0 + (i * 0.2), size=len(tmp))
        tmp["value"] = tmp["value"].values + noise

        # metric-specific cleanup
        tmp.loc[tmp["metric"] == "speed", "value"] = tmp.loc[tmp["metric"] == "speed", "value"].clip(lower=0)
        tmp.loc[tmp["metric"] == "demand", "value"] = tmp.loc[tmp["metric"] == "demand", "value"].clip(lower=0)
        tmp.loc[tmp["metric"] == "inflow", "value"] = tmp.loc[tmp["metric"] == "inflow", "value"].clip(lower=0)

        out.append(tmp)

    big = pd.concat(out, ignore_index=True)
    # cap size to keep app fast
    big = big.sample(n=min(len(big), 250000), random_state=42).reset_index(drop=True)
    return big

def ensure_db():
    if os.path.exists(DB_PATH):
        return

    df = read_demo_csv()
    df = expand_demo_if_too_small(df)

    con = sqlite3.connect(DB_PATH)
    df.to_sql("fact_traffic_long", con, if_exists="replace", index=False)

    cur = con.cursor()

    # Congestion view (speed-based)
    cur.execute("""
    CREATE VIEW IF NOT EXISTS v_congestion_index AS
    WITH s AS (
      SELECT sensor_id,
             time_index,
             value AS speed_value,
             MAX(value) OVER (PARTITION BY sensor_id) AS sensor_max_speed
      FROM fact_traffic_long
      WHERE metric = 'speed'
    )
    SELECT
      sensor_id,
      time_index,
      speed_value,
      CASE
        WHEN sensor_max_speed IS NULL OR sensor_max_speed = 0 THEN NULL
        ELSE (1.0 - (speed_value / sensor_max_speed))
      END AS congestion_index
    FROM s;
    """)

    # Anomaly view (z-score per metric+sensor)
    cur.execute("""
    CREATE VIEW IF NOT EXISTS v_anomalies AS
    WITH base AS (
      SELECT
        metric,
        sensor_id,
        time_index,
        value,
        AVG(value) OVER (PARTITION BY metric, sensor_id) AS mu,
        AVG(value*value) OVER (PARTITION BY metric, sensor_id) AS mu2
      FROM fact_traffic_long
    ),
    stats AS (
      SELECT
        metric,
        sensor_id,
        time_index,
        value,
        mu,
        CASE
          WHEN (mu2 - mu*mu) <= 0 THEN NULL
          ELSE SQRT(mu2 - mu*mu)
        END AS sigma
      FROM base
    )
    SELECT
      metric,
      sensor_id,
      time_index,
      value,
      CASE
        WHEN sigma IS NULL OR sigma = 0 THEN NULL
        ELSE (value - mu) / sigma
      END AS z_score,
      CASE
        WHEN sigma IS NULL OR sigma = 0 THEN 0
        WHEN ABS((value - mu) / sigma) >= 3 THEN 1
        ELSE 0
      END AS is_anomaly
    FROM stats;
    """)

    con.commit()
    con.close()

@st.cache_data
def query_df(sql):
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, con)
    con.close()
    return df

ensure_db()

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("UrbanPulse")
page = st.sidebar.radio("Page", ["Overview", "Time Insights", "Risk & Zones", "Incidents"], index=0)

metric = st.sidebar.selectbox("Metric", ["speed", "demand", "inflow"], index=0)

sensor_list = query_df(f"""
SELECT DISTINCT sensor_id
FROM fact_traffic_long
WHERE metric='{metric}'
ORDER BY sensor_id;
""")["sensor_id"].tolist()

if len(sensor_list) == 0:
    st.error("No sensors found in demo dataset.")
    st.stop()

sensor_id = st.sidebar.selectbox("Sensor", sensor_list, index=0)

max_time = int(query_df(f"""
SELECT MAX(time_index) AS m
FROM fact_traffic_long
WHERE metric='{metric}';
""")["m"].iloc[0])

t0, t1 = st.sidebar.slider("Time window (time_index)", 0, max_time, (0, min(2000, max_time)))

# -----------------------------
# Common KPIs
# -----------------------------
total_rows = int(query_df("SELECT COUNT(*) AS n FROM fact_traffic_long;")["n"].iloc[0])
total_anom = int(query_df("SELECT COUNT(*) AS n FROM v_anomalies WHERE is_anomaly=1;")["n"].iloc[0])
avg_cong = query_df("SELECT AVG(congestion_index) AS a FROM v_congestion_index WHERE congestion_index IS NOT NULL;")["a"].iloc[0]

st.title("🚦 UrbanPulse: Smart City Traffic Intelligence Platform (Demo)")

c1, c2, c3 = st.columns(3)
c1.metric("Total Records", f"{total_rows:,}")
c2.metric("Total Anomalies (z≥3)", f"{total_anom:,}")
c3.metric("Avg Congestion Index (speed)", f"{avg_cong:.3f}" if avg_cong is not None else "N/A")

st.caption(
    "UrbanPulse turns raw traffic signals into a warehouse + KPI views + anomaly detection + risk ranking. "
    "Use the pages on the left to explore time patterns, risk segmentation, and incident-style alerts."
)

st.divider()

# -----------------------------
# Page: Overview
# -----------------------------
if page == "Overview":
    left, right = st.columns(2)

    with left:
        st.subheader("🏁 Most Congested Sensors")
        cong = query_df("""
        SELECT sensor_id, AVG(congestion_index) AS avg_congestion
        FROM v_congestion_index
        WHERE congestion_index IS NOT NULL
        GROUP BY sensor_id
        ORDER BY avg_congestion DESC
        LIMIT 10;
        """)
        st.dataframe(cong, use_container_width=True)

    with right:
        st.subheader("⚠️ Most Anomalous Sensors")
        anom = query_df("""
        SELECT metric, sensor_id, SUM(is_anomaly) AS anomaly_count
        FROM v_anomalies
        GROUP BY metric, sensor_id
        ORDER BY anomaly_count DESC
        LIMIT 15;
        """)
        st.dataframe(anom, use_container_width=True)

    st.subheader(f"📈 Drilldown • {metric} • {sensor_id}")
    series = query_df(f"""
    SELECT time_index, value
    FROM fact_traffic_long
    WHERE metric='{metric}' AND sensor_id='{sensor_id}'
      AND time_index BETWEEN {t0} AND {t1}
    ORDER BY time_index;
    """)
    fig = px.line(series, x="time_index", y="value")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Anomaly points (z-score ≥ 3)")
    an_pts = query_df(f"""
    SELECT time_index, value, z_score
    FROM v_anomalies
    WHERE metric='{metric}' AND sensor_id='{sensor_id}'
      AND is_anomaly=1
      AND time_index BETWEEN {t0} AND {t1}
    ORDER BY time_index;
    """)
    st.dataframe(an_pts, use_container_width=True)

# -----------------------------
# Page: Time Insights
# -----------------------------
elif page == "Time Insights":
    st.subheader("🕒 Time Insights (Simulated timestamps)")

    step_minutes = st.sidebar.selectbox("Assume each step is", [5, 15, 60], index=0)
    start_date = datetime(2025, 1, 1, 0, 0, 0)

    # Pull only what we need for speed
    df = query_df(f"""
    SELECT time_index, sensor_id, metric, value
    FROM fact_traffic_long
    WHERE metric='{metric}';
    """)

    # Create timestamp features
    # (lightweight mapping)
    df = df.copy()
    df["timestamp"] = df["time_index"].apply(lambda t: start_date + timedelta(minutes=step_minutes * int(t)))
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.weekday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    by_hour = df.groupby("hour", as_index=False)["value"].mean()
    by_dow = df.groupby("day_of_week", as_index=False)["value"].mean()
    by_weekend = df.groupby("is_weekend", as_index=False)["value"].mean()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average by Hour")
        st.plotly_chart(px.line(by_hour, x="hour", y="value"), use_container_width=True)

    with col2:
        st.subheader("Average by Day of Week (0=Mon)")
        st.plotly_chart(px.line(by_dow, x="day_of_week", y="value"), use_container_width=True)

    st.subheader("Weekday vs Weekend")
    by_weekend["label"] = by_weekend["is_weekend"].apply(lambda x: "Weekend" if x == 1 else "Weekday")
    st.plotly_chart(px.bar(by_weekend, x="label", y="value"), use_container_width=True)

    st.caption("Tip: These time KPIs are what make dashboards feel operational (rush hours, weekend patterns, etc.).")

# -----------------------------
# Page: Risk & Zones
# -----------------------------
elif page == "Risk & Zones":
    st.subheader("🧠 Risk & Zones (Segmentation + Prioritization)")

    if not SKLEARN_OK:
        st.warning("scikit-learn not available. Add scikit-learn to requirements.txt and redeploy.")
        st.stop()

    # Features per sensor
    speed_cong = query_df("""
    SELECT sensor_id, AVG(congestion_index) AS avg_congestion
    FROM v_congestion_index
    WHERE congestion_index IS NOT NULL
    GROUP BY sensor_id;
    """)

    demand_avg = query_df("""
    SELECT sensor_id, AVG(value) AS avg_demand
    FROM fact_traffic_long
    WHERE metric='demand'
    GROUP BY sensor_id;
    """)

    inflow_avg = query_df("""
    SELECT sensor_id, AVG(value) AS avg_inflow
    FROM fact_traffic_long
    WHERE metric='inflow'
    GROUP BY sensor_id;
    """)

    anom_counts = query_df("""
    SELECT sensor_id, SUM(is_anomaly) AS anomaly_count
    FROM v_anomalies
    GROUP BY sensor_id;
    """)

    feat = speed_cong.merge(demand_avg, on="sensor_id", how="left")
    feat = feat.merge(inflow_avg, on="sensor_id", how="left")
    feat = feat.merge(anom_counts, on="sensor_id", how="left")

    feat["avg_demand"] = feat["avg_demand"].fillna(0)
    feat["avg_inflow"] = feat["avg_inflow"].fillna(0)
    feat["anomaly_count"] = feat["anomaly_count"].fillna(0)

    # Risk score (simple, recruiter-friendly)
    def minmax(s):
        mn = float(s.min())
        mx = float(s.max())
        if mx == mn:
            return s * 0
        return (s - mn) / (mx - mn)

    feat["cong_norm"] = minmax(feat["avg_congestion"])
    feat["dem_norm"] = minmax(feat["avg_demand"])
    feat["inf_norm"] = minmax(feat["avg_inflow"])
    feat["anom_norm"] = minmax(feat["anomaly_count"])

    # weights
    feat["risk_score"] = (
        0.50 * feat["cong_norm"] +
        0.20 * feat["dem_norm"] +
        0.10 * feat["inf_norm"] +
        0.20 * feat["anom_norm"]
    )

    p80 = feat["risk_score"].quantile(0.80)
    p50 = feat["risk_score"].quantile(0.50)

    def band(x):
        if x >= p80:
            return "HIGH"
        if x >= p50:
            return "MEDIUM"
        return "LOW"

    feat["risk_level"] = feat["risk_score"].apply(band)

    # Zones via clustering
    k = st.sidebar.slider("Number of zones", 3, 6, 3)
    X = feat[["avg_congestion", "avg_demand", "avg_inflow", "anomaly_count"]].copy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=int(k), random_state=42, n_init=10)
    feat["zone_id"] = km.fit_predict(Xs)
    feat["zone_name"] = feat["zone_id"].apply(lambda z: f"Zone_{int(z)+1}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Priority Sensors (by risk score)")
        top = feat.sort_values("risk_score", ascending=False).head(10)
        show = top[["sensor_id","zone_name","risk_level","risk_score","avg_congestion","avg_demand","avg_inflow","anomaly_count"]].copy()
        st.dataframe(show, use_container_width=True)

    with col2:
        st.subheader("Zone Ranking")
        zone_rank = feat.groupby("zone_name", as_index=False).agg(
            sensors=("sensor_id","count"),
            avg_zone_risk=("risk_score","mean"),
            high_risk=("risk_level", lambda s: int((s == "HIGH").sum()))
        ).sort_values("avg_zone_risk", ascending=False)
        st.dataframe(zone_rank, use_container_width=True)

    st.subheader("Risk distribution")
    fig = px.histogram(feat, x="risk_score")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Page: Incidents
# -----------------------------
elif page == "Incidents":
    st.subheader("🚨 Incidents (Forecast-error style alerts)")

    st.caption(
        "This page creates a simple 'incident' flag: if the value drops sharply vs recent history. "
        "It’s a practical alerting idea used in monitoring systems."
    )

    lookback = st.sidebar.slider("Lookback window", 30, 300, 120)

    df = query_df(f"""
    SELECT time_index, value
    FROM fact_traffic_long
    WHERE metric='{metric}' AND sensor_id='{sensor_id}'
      AND time_index BETWEEN {max(0, t0 - lookback)} AND {t1}
    ORDER BY time_index;
    """)

    if len(df) < lookback + 10:
        st.warning("Not enough data in this window. Increase the time window.")
        st.stop()

    # Simple baseline forecast: rolling median
    df = df.copy()
    df["baseline"] = df["value"].rolling(window=int(lookback), min_periods=10).median()

    # Incident flag: value much lower than baseline
    # threshold based on rolling std
    df["roll_std"] = df["value"].rolling(window=int(lookback), min_periods=10).std()
    df["abs_error"] = (df["value"] - df["baseline"]).abs()

    df["incident"] = 0
    mask = (df["roll_std"].notna()) & (df["abs_error"] > (2.5 * df["roll_std"]))
    df.loc[mask, "incident"] = 1

    st.subheader("Baseline vs Actual")
    fig = px.line(df[(df["time_index"] >= t0) & (df["time_index"] <= t1)], x="time_index", y=["value","baseline"])
    st.plotly_chart(fig, use_container_width=True)

    inc_pts = df[(df["time_index"] >= t0) & (df["time_index"] <= t1) & (df["incident"] == 1)].copy()
    st.subheader("Incident points in window")
    st.write(f"Incidents detected: {int(inc_pts.shape[0])}")
    st.dataframe(inc_pts[["time_index","value","baseline","abs_error","roll_std"]].head(200), use_container_width=True)

    if len(inc_pts) > 0:
        fig2 = px.scatter(inc_pts, x="time_index", y="value", hover_data=["abs_error"], title="Incident markers")
        st.plotly_chart(fig2, use_container_width=True)

