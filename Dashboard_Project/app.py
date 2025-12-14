import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from dash import Dash, dcc, html, Input, Output, State
from dash.dash_table import DataTable
import plotly.express as px
import base64
from fpdf import FPDF
import webbrowser

# ===============================
# 1. LOAD DATA
# ===============================
df = pd.read_csv("nyc_taxi_with_coords.csv")

# Drop missing crucial info
df = df.dropna(subset=["pickup_longitude", "pickup_latitude", "pickup_borough", "tpep_pickup_datetime"])
df["pickup_longitude"] = df["pickup_longitude"].astype(float)
df["pickup_latitude"] = df["pickup_latitude"].astype(float)
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
df["hour"] = df["tpep_pickup_datetime"].dt.hour

# Optional: limit rows for performance
MAX_ROWS = 150000
if len(df) > MAX_ROWS:
    df = df.sample(MAX_ROWS, random_state=42)

# ===============================
# 2. DBSCAN HOTSPOTS
# ===============================
coords = np.radians(df[["pickup_latitude","pickup_longitude"]])
dbscan = DBSCAN(eps=0.003, min_samples=100, metric="haversine", algorithm="ball_tree")
df["cluster"] = dbscan.fit_predict(coords).astype(str)

# ===============================
# 3. DASH APP
# ===============================
app = Dash(__name__)
app.title = "NYC Taxi Interactive Dashboard"

# ===============================
# KPI CARD HELPER
# ===============================
def kpi_card(title, value):
    return html.Div(
        style={
            "backgroundColor":"#1e293b",
            "padding":"20px",
            "borderRadius":"12px",
            "width":"220px",
            "textAlign":"center",
            "color":"white",
            "boxShadow":"2px 2px 8px #00000050"
        },
        children=[html.H4(title), html.H2(str(value))]
    )

# ===============================
# 4. LAYOUT
# ===============================
app.layout = html.Div(
    style={"backgroundColor":"#0f172a","color":"white","minHeight":"100vh","padding":"20px","fontFamily":"Arial"},
    children=[

        html.H1("🚖 NYC Taxi Dashboard", style={"textAlign":"center"}),
        html.P("Geospatial hotspots, KPIs, charts, and data export", style={"textAlign":"center","color":"#94a3b8"}),
        html.Hr(style={"borderColor":"#334155"}),

        # Filters
        html.Div(
            style={"display":"flex","justifyContent":"center","gap":"30px","marginBottom":"20px"},
            children=[
                html.Div([
                    html.Label("Select Hour"),
                    dcc.Dropdown(
                        id="hour-filter",
                        options=[{"label":str(h),"value":h} for h in sorted(df["hour"].unique())],
                        value=12,
                        clearable=False,
                        style={"color":"black"}
                    )
                ], style={"width":"200px"}),

                html.Div([
                    html.Label("Select Borough"),
                    dcc.Dropdown(
                        id="borough-filter",
                        options=[{"label":b,"value":b} for b in sorted(df["pickup_borough"].unique())],
                        placeholder="All Boroughs",
                        clearable=True,
                        style={"color":"black"}
                    )
                ], style={"width":"200px"})
            ]
        ),

        html.Hr(style={"borderColor":"#334155"}),

        # KPI Cards
        html.Div(id="kpi-cards", style={"display":"flex","gap":"20px","flexWrap":"wrap","justifyContent":"center"}),

        html.Hr(style={"borderColor":"#334155"}),

        # Tabs
        dcc.Tabs(id="tabs", value="map", children=[
            dcc.Tab(label="🗺 Map", value="map", style={"backgroundColor":"#0f172a","color":"white"}, selected_style={"backgroundColor":"#1e293b","color":"white"}),
            dcc.Tab(label="📊 Charts", value="charts", style={"backgroundColor":"#0f172a","color":"white"}, selected_style={"backgroundColor":"#1e293b","color":"white"}),
            dcc.Tab(label="📁 Data", value="data", style={"backgroundColor":"#0f172a","color":"white"}, selected_style={"backgroundColor":"#1e293b","color":"white"})
        ]),
        html.Div(id="tabs-content", style={"padding":"20px"})
    ]
)

# ===============================
# 5. CALLBACKS
# ===============================
@app.callback(
    Output("kpi-cards","children"),
    Input("hour-filter","value"),
    Input("borough-filter","value")
)
def update_kpis(hour, borough):
    temp = df[df["hour"]==hour]
    if borough:
        temp = temp[temp["pickup_borough"]==borough]
    top_zone = temp["pickup_zone"].mode()[0] if not temp.empty else "N/A"
    return [
        kpi_card("Total Trips", len(temp)),
        kpi_card("Clusters", temp["cluster"].nunique()),
        kpi_card("Avg Distance", f"{temp['trip_distance'].mean():.2f} mi" if not temp.empty else "0"),
        kpi_card("Avg Fare", f"${temp['total_amount'].mean():.2f}" if not temp.empty else "0"),
        kpi_card("Top Zone", top_zone)
    ]

@app.callback(
    Output("tabs-content","children"),
    Input("tabs","value"),
    Input("hour-filter","value"),
    Input("borough-filter","value")
)
def render_tab(tab, hour, borough):
    temp = df[df["hour"]==hour]
    if borough:
        temp = temp[temp["pickup_borough"]==borough]

    if temp.empty:
        return html.Div("No data for this selection", style={"color":"red","textAlign":"center"})

    # -------- MAP --------
    if tab=="map":
        fig_map = px.scatter_map(
            temp,
            lat="pickup_latitude",
            lon="pickup_longitude",
            color="cluster",
            zoom=10,
            height=600,
            hover_data=["pickup_zone","trip_distance","total_amount"],
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_map.add_density_mapbox(
            lat=temp["pickup_latitude"],
            lon=temp["pickup_longitude"],
            z=temp["trip_distance"],
            radius=15,
            colorscale="Viridis",
            opacity=0.4,
            name="Hotspots"
        )
        fig_map.update_layout(
            mapbox_style="open-street-map",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="white",
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        return dcc.Graph(figure=fig_map)

    # -------- CHARTS --------
    elif tab=="charts":
        bar_data = temp.groupby("cluster").size().reset_index(name="trip_count")
        line_data = temp.groupby("hour").size().reset_index(name="trips_per_hour")

        fig_bar = px.bar(bar_data, x="cluster", y="trip_count", color="trip_count", title="Trips per Cluster")
        fig_bar.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font_color="white")

        fig_line = px.line(line_data, x="hour", y="trips_per_hour", markers=True, title="Trips per Hour")
        fig_line.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font_color="white")

        return html.Div([dcc.Graph(figure=fig_bar), dcc.Graph(figure=fig_line)])

    # -------- DATA --------
    elif tab=="data":
        table = DataTable(
            columns=[{"name":i,"id":i} for i in temp.columns],
            data=temp.to_dict("records"),
            page_size=20,
            style_table={"overflowX":"auto"},
            style_cell={"color":"white","backgroundColor":"#0f172a"}
        )

        # CSV download
        csv_string = temp.to_csv(index=False)
        b64_csv = base64.b64encode(csv_string.encode()).decode()
        download_csv = html.A("⬇ Download CSV", href="data:text/csv;base64,"+b64_csv,
                              download="nyc_taxi_filtered.csv", style={"color":"#60a5fa"})

        # PDF download
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        for i, row in temp.head(100).iterrows():
            pdf.multi_cell(0, 6, txt=str(row.to_dict()))
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        b64_pdf = base64.b64encode(pdf_bytes).decode()
        download_pdf = html.A("⬇ Download PDF (first 100 rows)", href="data:application/pdf;base64,"+b64_pdf,
                              download="nyc_taxi_filtered.pdf", style={"color":"#60a5fa","marginLeft":"20px"})

        return html.Div([table, html.Br(), download_csv, download_pdf])

# ===============================
# RUN SERVER
# ===============================
if __name__=="__main__":
    app.run(debug=True)
