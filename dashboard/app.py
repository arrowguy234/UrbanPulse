import os
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px

DB_PATH = "urbanpulse.db"
DEMO_CSV = "data/demo/urbanpulse_demo.csv"

st.set_page_config(page_title="UrbanPulse • Smart City Platform", layout="wide")

def ensure_db():
    if os.path.exists(DB_PATH):
        return

    if not os.path.exists(DEMO_CSV):
        st.error("Demo dataset not found. Please add data/demo/urbanpulse_demo.csv")
        st.stop()

    df = pd.read_csv(DEMO_CSV).dropna(subset=["value"])
    con = sqlite3.connect(DB_PATH)

    df.to_sql("fact_traffic_long", con, if_exists="replace", index=False)

    # Views
    cur = con.cursor()

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

def query_df(sql):
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, con)
    con.close()
    return df

ensure_db()

st.title("🚦 UrbanPulse: Smart City Traffic Intelligence Platform (Demo)")

# Sidebar
metric = st.sidebar.selectbox("Metric", ["speed", "demand", "inflow"], index=0)

sensors = query_df(f"""
SELECT DISTINCT sensor_id
FROM fact_traffic_long
WHERE metric='{metric}'
ORDER BY sensor_id;
""")["sensor_id"].tolist()

sensor_id = st.sidebar.selectbox("Sensor", sensors, index=0)

max_time = int(query_df(f"""
SELECT MAX(time_index) AS m
FROM fact_traffic_long
WHERE metric='{metric}';
""")["m"].iloc[0])

t0, t1 = st.sidebar.slider("Time window", 0, max_time, (0, min(2000, max_time)))

# KPIs
total_rows = int(query_df("SELECT COUNT(*) AS n FROM fact_traffic_long;")["n"].iloc[0])
total_anom = int(query_df("SELECT COUNT(*) AS n FROM v_anomalies WHERE is_anomaly=1;")["n"].iloc[0])
avg_cong = query_df("SELECT AVG(congestion_index) AS a FROM v_congestion_index;")["a"].iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Total Records", f"{total_rows:,}")
c2.metric("Total Anomalies (z>=3)", f"{total_anom:,}")
c3.metric("Avg Congestion Index", f"{avg_cong:.3f}" if avg_cong is not None else "N/A")

st.divider()

# Leaderboards
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

st.divider()

# Drilldown plot
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

anom_pts = query_df(f"""
SELECT time_index, value, z_score
FROM v_anomalies
WHERE metric='{metric}' AND sensor_id='{sensor_id}'
  AND is_anomaly=1
  AND time_index BETWEEN {t0} AND {t1}
ORDER BY time_index;
""")
st.caption("Anomaly points (z>=3)")
st.dataframe(anom_pts, use_container_width=True)

