# 🚦 UrbanPulse -- Smart City Traffic Intelligence Platform

UrbanPulse is a live, end-to-end traffic analytics platform that
transforms raw spatiotemporal traffic signals into operational insights
for city-level decision making.

🔗 **Live Demo:** https://urbanpulse-mtykpucaj4znh9vj8wg4zc.streamlit.app
💻 **GitHub Repo:** https://github.com/arrowguy234/UrbanPulse

------------------------------------------------------------------------

## 📌 Project Overview

UrbanPulse simulates a smart-city traffic monitoring system capable of:

-   Monitoring congestion patterns
-   Detecting anomalies in traffic flow
-   Ranking high-risk traffic sensors
-   Segmenting sensors into operational zones
-   Identifying peak congestion hours
-   Generating incident-style alerts

This project demonstrates both **Data Analyst** and **Data Engineer**
capabilities through a production-style analytics workflow.

------------------------------------------------------------------------

## 🏗 Architecture

Raw Traffic Data\
→ Data Cleaning & Standardization\
→ Fact Table (fact_traffic_long)\
→ SQL Views (KPIs)\
→ Congestion Index\
→ Anomaly Detection (z-score)\
→ Risk Scoring + Clustering\
→ Streamlit Dashboard

------------------------------------------------------------------------

## 📊 Key Features

### 1️⃣ Congestion Index

Normalized per sensor:

congestion_index = 1 - (speed / max_speed_per_sensor)

------------------------------------------------------------------------

### 2️⃣ Anomaly Detection

Z-score based per metric + sensor:

is_anomaly = 1 if \|z\| \>= 3

------------------------------------------------------------------------

### 3️⃣ Risk Scoring Model

Weighted risk score combining: - Congestion - Demand - Inflow - Anomaly
count

Sensors classified into: - HIGH risk - MEDIUM risk - LOW risk

------------------------------------------------------------------------

### 4️⃣ Zone Segmentation

KMeans clustering groups sensors into operational zones.

------------------------------------------------------------------------

### 5️⃣ Time-Based Insights

-   Hourly congestion trends
-   Weekday vs weekend behavior
-   Peak demand analysis

------------------------------------------------------------------------

### 6️⃣ Incident Detection

Rolling baseline deviation flags traffic "incidents".

------------------------------------------------------------------------

## 📈 Dashboard Pages

• Overview -- KPIs + leaderboards\
• Time Insights -- hourly and weekly analysis\
• Risk & Zones -- clustering + prioritization\
• Incidents -- alert-style anomaly detection

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python (pandas, numpy)
-   SQL (SQLite)
-   Plotly
-   Streamlit
-   scikit-learn

------------------------------------------------------------------------

## 🎯 Skills Demonstrated

-   Data modeling & warehousing
-   KPI engineering
-   Feature engineering
-   Clustering (KMeans)
-   Statistical anomaly detection
-   Risk modeling
-   Interactive dashboard deployment
-   Git-based deployment workflow

------------------------------------------------------------------------

## 🚀 Future Enhancements

-   Migration to PostgreSQL
-   Airflow-based ETL scheduling
-   Real geospatial mapping
-   Forecast-based incident prediction
-   Executive PDF export

------------------------------------------------------------------------

## 👤 Author

Your Name\
LinkedIn: Your LinkedIn URL\
GitHub: https://github.com/arrowguy234
