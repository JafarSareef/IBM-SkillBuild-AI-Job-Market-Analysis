# =============================================================================
# Jafar_AI_Job_Market_Dashboard.py
# AI Job Market Analytics – Interactive Dashboard
# IBM SkillBuild Project  |  Author: Jafar
#
# Run with:
#   python Jafar_AI_Job_Market_Dashboard.py
# Then open: http://127.0.0.1:8050
#
# IMPORTANT NOTES
#   • Original CSV is read-only; no rows are deleted.
#   • Salary charts keep GBP / USD / INR strictly separate.
#   • The approx-USD cross-country chart is clearly labelled as
#     an analytical normalisation using fixed mid-2024 FX rates.
#   • No causal claims about AI and employment are made anywhere.
#   • "Unknown" contract values are displayed, not removed.
# =============================================================================

import sys
import os

# UTF-8 output for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State

# ---------------------------------------------------------------------------
# 0.  CONSTANTS
# ---------------------------------------------------------------------------

RAW_DATA_FILE = "ai_job_market_dataset.csv"

SALARY_SUSPICIOUS_THRESHOLDS = {
    "GBP": 1_000,
    "USD": 1_000,
    "INR": 10_000,
}

# Fixed mid-2024 FX rates for cross-country normalisation ONLY
FX_TO_USD = {"GBP": 1.27, "USD": 1.00, "INR": 0.012}

COLOUR_AI     = "#2563eb"   # blue
COLOUR_NONAI  = "#d97706"   # amber
COLOUR_PRED   = "#7c3aed"   # purple
COLOUR_STATED = "#059669"   # green
COLOUR_BG     = "#f8fafc"
COLOUR_CARD   = "#ffffff"
COLOUR_BORDER = "#e2e8f0"
COLOUR_TEXT   = "#1e293b"
COLOUR_MUTED  = "#64748b"

# ---------------------------------------------------------------------------
# 1.  DATA LOADING & PREPROCESSING
#     (mirrors Jafar_AI_Job_Market_Analytics.py – self-contained for the
#      dashboard so it can run without importing the analytics module)
# ---------------------------------------------------------------------------

def load_and_prepare() -> pd.DataFrame:
    """Load the raw CSV and return the fully preprocessed DataFrame."""
    if not os.path.exists(RAW_DATA_FILE):
        sys.exit(f"[ERROR] Dataset not found: {RAW_DATA_FILE}")

    df = pd.read_csv(RAW_DATA_FILE)

    # Drop trailing empty column
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df.drop(columns=unnamed, inplace=True)

    # Fill missing contract fields
    df["contract_type"] = df["contract_type"].fillna("Unknown")
    df["contract_time"]  = df["contract_time"].fillna("Unknown")

    # Parse dates
    df["created"] = pd.to_datetime(df["created"], utc=True, errors="coerce")
    df["created_date"] = pd.to_datetime(
        df["created_date"], format="%d-%m-%Y", errors="coerce"
    ).dt.date

    # Standardise booleans
    df["is_ai_related"] = df["is_ai_related"].map(
        {"TRUE": True, "FALSE": False, True: True, False: False}
    )
    df["salary_is_predicted"] = df["salary_is_predicted"].astype(bool)

    # Salary as float
    for col in ["salary_min", "salary_max", "salary_avg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Flag suspicious salaries
    df["salary_suspicious"] = False
    for ccy, threshold in SALARY_SUSPICIOUS_THRESHOLDS.items():
        mask = (df["currency"] == ccy) & df["salary_avg"].notna() & (df["salary_avg"] < threshold)
        df.loc[mask, "salary_suspicious"] = True

    # Derived columns
    df["job_type"]      = df["is_ai_related"].map({True: "AI-Related", False: "Non-AI"})
    df["salary_status"] = df["salary_is_predicted"].map({True: "Predicted", False: "Stated"})

    created_dt = pd.to_datetime(df["created_date"], errors="coerce")
    df["year"]       = created_dt.dt.year.astype("Int64")
    df["month"]      = created_dt.dt.month.astype("Int64")
    df["year_month"] = created_dt.dt.to_period("M").astype(str)

    # Re-post flag
    df["is_repost"] = df.duplicated(subset=["title", "company"], keep=False)

    # Approximate USD normalisation (for cross-country chart only)
    df["salary_usd_approx"] = df.apply(
        lambda r: r["salary_avg"] * FX_TO_USD.get(r["currency"], 1.0)
        if pd.notna(r["salary_avg"]) and not r["salary_suspicious"] else np.nan,
        axis=1,
    )

    return df


print("Loading and preprocessing data …", flush=True)
DF = load_and_prepare()
print(f"  Loaded {len(DF):,} rows × {len(DF.columns)} columns", flush=True)

# ---------------------------------------------------------------------------
# 2.  FILTER OPTION LISTS
# ---------------------------------------------------------------------------

ALL_COUNTRIES  = sorted(DF["country_name"].dropna().unique().tolist())
ALL_KEYWORDS   = sorted(DF["search_keyword"].dropna().unique().tolist())
ALL_JOB_TYPES  = ["AI-Related", "Non-AI"]
ALL_YEARS      = sorted([int(y) for y in DF["year"].dropna().unique()])
ALL_CURRENCIES = sorted(DF["currency"].dropna().unique().tolist())


def make_options(lst):
    """Turn a list into Dash dropdown option dicts."""
    return [{"label": str(v), "value": str(v)} for v in lst]


# ---------------------------------------------------------------------------
# 3.  CHART HELPERS
# ---------------------------------------------------------------------------

def apply_filters(df, countries, keywords, job_types, years, currencies):
    """Return a filtered slice of DF based on sidebar selections."""
    mask = pd.Series(True, index=df.index)
    if countries:
        mask &= df["country_name"].isin(countries)
    if keywords:
        mask &= df["search_keyword"].isin(keywords)
    if job_types:
        mask &= df["job_type"].isin(job_types)
    if years:
        mask &= df["year"].isin([int(y) for y in years])
    if currencies:
        mask &= df["currency"].isin(currencies)
    return df[mask].copy()


CHART_FONT   = dict(family="Segoe UI, system-ui, sans-serif", size=13, color=COLOUR_TEXT)
CHART_LAYOUT = dict(
    font=CHART_FONT,
    plot_bgcolor=COLOUR_BG,
    paper_bgcolor=COLOUR_CARD,
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

def empty_fig(msg="No data for the selected filters."):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color=COLOUR_MUTED))
    fig.update_layout(**CHART_LAYOUT, xaxis_visible=False, yaxis_visible=False)
    return fig


# ---------------------------------------------------------------------------
# 4.  APP LAYOUT
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="AI Job Market Analytics",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server   # expose Flask server for potential deployment

# ---- Sidebar ---------------------------------------------------------------

sidebar = dbc.Col([
    html.Div([
        html.H5("Filters", className="mb-3",
                style={"color": COLOUR_TEXT, "fontWeight": "700", "fontSize": "1rem"}),

        html.Label("Country", style={"fontWeight": "600", "fontSize": "0.85rem"}),
        dcc.Dropdown(
            id="filter-country",
            options=make_options(ALL_COUNTRIES),
            value=[],
            multi=True,
            placeholder="All countries",
            style={"fontSize": "0.85rem", "marginBottom": "12px"},
        ),

        html.Label("Search Keyword", style={"fontWeight": "600", "fontSize": "0.85rem"}),
        dcc.Dropdown(
            id="filter-keyword",
            options=make_options(ALL_KEYWORDS),
            value=[],
            multi=True,
            placeholder="All keywords",
            style={"fontSize": "0.85rem", "marginBottom": "12px"},
        ),

        html.Label("Job Type", style={"fontWeight": "600", "fontSize": "0.85rem"}),
        dcc.Dropdown(
            id="filter-jobtype",
            options=make_options(ALL_JOB_TYPES),
            value=[],
            multi=True,
            placeholder="All types",
            style={"fontSize": "0.85rem", "marginBottom": "12px"},
        ),

        html.Label("Year", style={"fontWeight": "600", "fontSize": "0.85rem"}),
        dcc.Dropdown(
            id="filter-year",
            options=make_options(ALL_YEARS),
            value=[],
            multi=True,
            placeholder="All years",
            style={"fontSize": "0.85rem", "marginBottom": "12px"},
        ),

        html.Label("Currency (salary charts)", style={"fontWeight": "600", "fontSize": "0.85rem"}),
        dcc.Dropdown(
            id="filter-currency",
            options=make_options(ALL_CURRENCIES),
            value=[],
            multi=True,
            placeholder="All currencies",
            style={"fontSize": "0.85rem", "marginBottom": "12px"},
        ),

        dbc.Button("Reset Filters", id="btn-reset", color="secondary",
                   size="sm", className="mt-2 w-100"),

        html.Hr(style={"borderColor": COLOUR_BORDER, "marginTop": "20px"}),

        # Data limitations panel
        html.Div([
            html.H6("Data Limitations", style={"color": COLOUR_TEXT, "fontWeight": "700",
                                                "fontSize": "0.82rem", "marginBottom": "6px"}),
            html.Ul([
                html.Li("Salaries are in GBP, USD, INR — not directly comparable."),
                html.Li("The approx-USD chart uses fixed mid-2024 FX rates only."),
                html.Li("68.5 % of salary values are algorithmically predicted."),
                html.Li("74.5 % of contract_type values are Unknown."),
                html.Li("5 search keywords only — not a full market sample."),
                html.Li("Re-posts are preserved; toggle 'unique only' with the filter."),
                html.Li("No causal claims about AI and employment are made."),
            ], style={"paddingLeft": "14px", "fontSize": "0.75rem",
                      "color": COLOUR_MUTED, "lineHeight": "1.6"}),
        ]),
    ], style={
        "background": COLOUR_CARD,
        "border": f"1px solid {COLOUR_BORDER}",
        "borderRadius": "10px",
        "padding": "16px",
        "height": "100%",
    }),
], width=2, style={"paddingRight": "8px"})


# ---- KPI row ---------------------------------------------------------------

def kpi_card(card_id, title, icon="📊"):
    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.Div(icon, style={"fontSize": "1.4rem", "marginBottom": "2px"}),
                html.Div(title, style={"fontSize": "0.75rem", "color": COLOUR_MUTED,
                                       "fontWeight": "600", "textTransform": "uppercase",
                                       "letterSpacing": "0.04em", "marginBottom": "4px"}),
                html.Div("—", id=card_id,
                         style={"fontSize": "1.55rem", "fontWeight": "700",
                                "color": COLOUR_TEXT}),
            ], style={"padding": "14px 16px"}),
        ], style={"border": f"1px solid {COLOUR_BORDER}", "borderRadius": "10px",
                  "background": COLOUR_CARD}),
        xs=6, md=4, lg=2, className="mb-3",
    )


kpi_row = dbc.Row([
    kpi_card("kpi-total",   "Total Postings",    "📋"),
    kpi_card("kpi-ai",      "AI-Related",        "🤖"),
    kpi_card("kpi-nonai",   "Non-AI",            "💼"),
    kpi_card("kpi-pct",     "AI %",              "📈"),
    kpi_card("kpi-unique",  "Unique Roles",      "🔍"),
    kpi_card("kpi-reposts", "Re-posts",          "🔄"),
], className="g-2 mb-3")


# ---- Main content ----------------------------------------------------------

def chart_card(title, chart_id, footnote=None):
    children = [
        html.H6(title, style={"fontWeight": "700", "color": COLOUR_TEXT,
                               "fontSize": "0.9rem", "marginBottom": "4px"}),
        dcc.Graph(id=chart_id, config={"displayModeBar": False},
                  style={"height": "340px"}),
    ]
    if footnote:
        children.append(
            html.P(footnote, style={"fontSize": "0.7rem", "color": COLOUR_MUTED,
                                    "marginTop": "4px", "marginBottom": "0"})
        )
    return dbc.Card(
        dbc.CardBody(children, style={"padding": "14px"}),
        style={"border": f"1px solid {COLOUR_BORDER}", "borderRadius": "10px",
               "background": COLOUR_CARD, "marginBottom": "16px"},
    )


content = dbc.Col([
    kpi_row,

    # Row 1: overall + by country
    dbc.Row([
        dbc.Col(chart_card("AI-Related vs Non-AI — Overall",
                           "chart-overall"), md=5),
        dbc.Col(chart_card("AI-Related vs Non-AI by Country",
                           "chart-country"), md=7),
    ], className="g-3"),

    # Row 2: keyword + category
    dbc.Row([
        dbc.Col(chart_card("AI Proportion by Search Keyword",
                           "chart-keyword"), md=5),
        dbc.Col(chart_card("AI-Related Jobs by Category (Top 15)",
                           "chart-category"), md=7),
    ], className="g-3"),

    # Row 3: time trend + reposts
    dbc.Row([
        dbc.Col(chart_card("Monthly Posting Trend",
                           "chart-trend"), md=8),
        dbc.Col(chart_card("Total vs Unique (Non-Reposted) Postings",
                           "chart-reposts"), md=4),
    ], className="g-3"),

    # Row 4: salary per currency
    dbc.Row([
        dbc.Col(chart_card(
            "Salary Distribution — AI vs Non-AI (per currency, suspicious excluded)",
            "chart-salary-box",
            footnote="Each currency shown separately. Suspicious salary records (£13, £20.50, $32, £500) are excluded."
        ), md=7),
        dbc.Col(chart_card(
            "Predicted vs Stated Salary (per currency)",
            "chart-salary-ps",
            footnote="Predicted = algorithmically estimated. Stated = employer-published. ~68.5 % are Predicted."
        ), md=5),
    ], className="g-3"),

    # Row 5: contract + normalised salary
    dbc.Row([
        dbc.Col(chart_card(
            "Contract Type & Time (Descriptive Only — high Unknown rate)",
            "chart-contract"
        ), md=5),
        dbc.Col(chart_card(
            "Approx. Salary (USD-normalised) — Analytical Estimate Only",
            "chart-salary-norm",
            footnote=(
                "ANALYTICAL ESTIMATE: GBP × 1.27, INR × 0.012 (fixed mid-2024 rates). "
                "Do NOT treat as precise converted values. For illustrative comparison only."
            )
        ), md=7),
    ], className="g-3"),

], width=10, style={"paddingLeft": "8px"})


# ---- Full layout -----------------------------------------------------------

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H3("AI Job Market Analytics",
                    style={"fontWeight": "800", "color": COLOUR_TEXT,
                           "marginBottom": "2px", "fontSize": "1.5rem"}),
            html.P(
                "Exploring how AI-related job postings differ from traditional postings "
                "across countries, industries, keywords, and salary — IBM SkillBuild Project | Author: Jafar",
                style={"color": COLOUR_MUTED, "fontSize": "0.82rem",
                       "marginBottom": "0"},
            ),
        ])
    ], className="mb-3 mt-2"),

    html.Hr(style={"borderColor": COLOUR_BORDER, "marginBottom": "16px"}),

    dbc.Row([sidebar, content], className="g-0"),

    html.Hr(style={"borderColor": COLOUR_BORDER, "marginTop": "24px"}),
    html.P(
        "Dataset: AI Job Market (UK / US / India) | 1,696 postings | "
        "May 2023 – Aug 2026 | Collected via 5 search keywords. "
        "This dashboard is descriptive only. No causal claims are made.",
        style={"fontSize": "0.72rem", "color": COLOUR_MUTED,
               "textAlign": "center", "marginBottom": "8px"},
    ),
], fluid=True, style={"background": COLOUR_BG, "minHeight": "100vh",
                       "fontFamily": "Segoe UI, system-ui, sans-serif",
                       "padding": "0 24px"})


# ---------------------------------------------------------------------------
# 5.  CALLBACKS
# ---------------------------------------------------------------------------

FILTER_INPUTS = [
    Input("filter-country",  "value"),
    Input("filter-keyword",  "value"),
    Input("filter-jobtype",  "value"),
    Input("filter-year",     "value"),
    Input("filter-currency", "value"),
]


# ---- Reset filters ---------------------------------------------------------

@app.callback(
    Output("filter-country",  "value"),
    Output("filter-keyword",  "value"),
    Output("filter-jobtype",  "value"),
    Output("filter-year",     "value"),
    Output("filter-currency", "value"),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(_):
    return [], [], [], [], []


# ---- KPI cards -------------------------------------------------------------

@app.callback(
    Output("kpi-total",   "children"),
    Output("kpi-ai",      "children"),
    Output("kpi-nonai",   "children"),
    Output("kpi-pct",     "children"),
    Output("kpi-unique",  "children"),
    Output("kpi-reposts", "children"),
    *FILTER_INPUTS,
)
def update_kpis(countries, keywords, job_types, years, currencies):
    d = apply_filters(DF, countries, keywords, job_types, years, currencies)
    total    = len(d)
    ai_cnt   = d["is_ai_related"].sum()
    nonai    = total - ai_cnt
    pct      = f"{ai_cnt / total * 100:.1f}%" if total else "—"
    unique   = (~d["is_repost"]).sum()
    reposts  = d["is_repost"].sum()
    return (
        f"{total:,}",
        f"{ai_cnt:,}",
        f"{nonai:,}",
        pct,
        f"{unique:,}",
        f"{reposts:,}",
    )


# ---- Chart 1: Overall pie + bar -------------------------------------------

@app.callback(Output("chart-overall", "figure"), *FILTER_INPUTS)
def chart_overall(countries, keywords, job_types, years, currencies):
    d = apply_filters(DF, countries, keywords, job_types, years, currencies)
    if d.empty:
        return empty_fig()
    vc = d["job_type"].value_counts().reset_index()
    vc.columns = ["Job Type", "Count"]
    fig = px.pie(
        vc, names="Job Type", values="Count",
        color="Job Type",
        color_discrete_map={"AI-Related": COLOUR_AI, "Non-AI": COLOUR_NONAI},
        hole=0.42,
    )
    fig.update_traces(
        textposition="outside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Pct: %{percent}<extra></extra>",
    )
    fig.update_layout(**CHART_LAYOUT, showlegend=True)
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    return fig


# ---- Chart 2: By country grouped bar --------------------------------------

@app.callback(Output("chart-country", "figure"), *FILTER_INPUTS)
def chart_country(countries, keywords, job_types, years, currencies):
    d = apply_filters(DF, countries, keywords, job_types, years, currencies)
    if d.empty:
        return empty_fig()
    ct = d.groupby(["country_name", "job_type"]).size().reset_index(name="Count")
    fig = px.bar(
        ct, x="country_name", y="Count", color="job_type",
        barmode="group",
        color_discrete_map={"AI-Related": COLOUR_AI, "Non-AI": COLOUR_NONAI},
        labels={"country_name": "Country", "Count": "Postings", "job_type": "Job Type"},
        text="Count",
    )
    fig.update_traces(textposition="outside",
                      hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:,}<extra></extra>")
    fig.update_layout(**CHART_LAYOUT, xaxis_title="", yaxis_title="Postings")
    return fig


# ---- Chart 3: AI proportion by keyword ------------------------------------

@app.callback(Output("chart-keyword", "figure"), *FILTER_INPUTS)
def chart_keyword(countries, keywords, job_types, years, currencies):
    d = apply_filters(DF, countries, keywords, job_types, years, currencies)
    if d.empty:
        return empty_fig()
    kw = (d.groupby(["search_keyword", "job_type"])
          .size().reset_index(name="Count"))
    totals = kw.groupby("search_keyword")["Count"].sum().rename("Total")
    kw = kw.join(totals, on="search_keyword")
    kw["AI %"] = (kw["Count"] / kw["Total"] * 100).round(1)
    ai_kw = (kw[kw["job_type"] == "AI-Related"]
             .sort_values("AI %", ascending=True))
    if ai_kw.empty:
        return empty_fig("No AI-Related data for selected filters.")
    fig = px.bar(
        ai_kw, y="search_keyword", x="AI %",
        orientation="h",
        color="AI %",
        color_continuous_scale=[[0, COLOUR_NONAI], [1, COLOUR_AI]],
        text=ai_kw["AI %"].apply(lambda v: f"{v:.0f}%"),
        labels={"search_keyword": "Keyword", "AI %": "AI-Related %"},
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>AI %: %{x:.1f}%<extra></extra>",
    )
    fig.update_layout(**CHART_LAYOUT, xaxis_range=[0, 115],
                      coloraxis_showscale=False, yaxis_title="")
    return fig


# ---- Chart 4: AI-related jobs by category ---------------------------------

@app.callback(Output("chart-category", "figure"), *FILTER_INPUTS)
def chart_category(countries, keywords, job_types, years, currencies):
    d = apply_filters(DF, countries, keywords, job_types, years, currencies)
    if d.empty:
        return empty_fig()
    ct = (d.groupby(["category", "job_type"])
          .size().reset_index(name="Count"))
    ai_only = ct[ct["job_type"] == "AI-Related"].sort_values("Count", ascending=False).head(15)
    if ai_only.empty:
        return empty_fig("No AI-Related data for selected filters.")
    fig = px.bar(
        ai_only.sort_values("Count"), y="category", x="Count",
        orientation="h",
        color_discrete_sequence=[COLOUR_AI],
        text="Count",
        labels={"category": "Category", "Count": "AI-Related Postings"},
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>AI-Related: %{x:,}<extra></extra>",
    )
    fig.update_layout(**CHART_LAYOUT, yaxis_title="", xaxis_title="AI-Related Postings")
    return fig


# ---- Chart 5: Monthly trend -----------------------------------------------

@app.callback(Output("chart-trend", "figure"), *FILTER_INPUTS)
def chart_trend(countries, keywords, job_types, years, currencies):
    d = apply_filters(DF, countries, keywords, job_types, years, currencies)
    if d.empty:
        return empty_fig()
    mo = (d.groupby(["year_month", "job_type"])
          .size().reset_index(name="Count")
          .sort_values("year_month"))
    fig = px.line(
        mo, x="year_month", y="Count", color="job_type",
        color_discrete_map={"AI-Related": COLOUR_AI, "Non-AI": COLOUR_NONAI},
        markers=True,
        labels={"year_month": "Month", "Count": "Postings", "job_type": "Job Type"},
    )
    fig.update_traces(
        marker_size=5,
        hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:,}<extra></extra>",
    )
    fig.update_layout(**CHART_LAYOUT, xaxis_title="", yaxis_title="Postings",
                      xaxis=dict(tickangle=40, tickfont=dict(size=10)))
    return fig


# ---- Chart 6: Salary boxplot per currency ---------------------------------

@app.callback(Output("chart-salary-box", "figure"), *FILTER_INPUTS)
def chart_salary_box(countries, keywords, job_types, years, currencies):
    d = apply_filters(DF, countries, keywords, job_types, years, currencies)
    # Always exclude suspicious records from salary charts
    s = d[d["salary_avg"].notna() & (~d["salary_suspicious"])].copy()
    if s.empty:
        return empty_fig("No salary data for selected filters.")

    # Determine which currencies to show (respect filter, default all)
    ccys_in_data = s["currency"].unique().tolist()
    show_ccys = [c for c in ["GBP", "USD", "INR"] if c in ccys_in_data]
    if not show_ccys:
        return empty_fig("No salary data for selected filters.")

    # One box-trace per (currency, job_type) combination
    fig = go.Figure()
    ccy_symbols = {"GBP": "£", "USD": "$", "INR": "₹"}
    for ccy in show_ccys:
        sub = s[s["currency"] == ccy]
        sym = ccy_symbols.get(ccy, "")
        for jt, colour in [("AI-Related", COLOUR_AI), ("Non-AI", COLOUR_NONAI)]:
            vals = sub[sub["job_type"] == jt]["salary_avg"].dropna()
            if vals.empty:
                continue
            fig.add_trace(go.Box(
                y=vals,
                name=f"{jt} ({ccy})",
                marker_color=colour,
                opacity=0.75,
                boxmean=True,
                hovertemplate=(
                    f"<b>{{fullData.name}}</b><br>"
                    f"Median: {sym}%{{median:,.0f}}<br>"
                    f"Mean: {sym}%{{mean:,.0f}}<extra></extra>"
                ),
            ))

    fig.update_layout(
        **CHART_LAYOUT,
        yaxis_title="Annual salary (see legend for currency)",
        boxmode="group",
    )
    return fig


# ---- Chart 7: Predicted vs Stated per currency ----------------------------

@app.callback(Output("chart-salary-ps", "figure"), *FILTER_INPUTS)
def chart_salary_ps(countries, keywords, job_types, years, currencies):
    d = apply_filters(DF, countries, keywords, job_types, years, currencies)
    s = d[d["salary_avg"].notna() & (~d["salary_suspicious"])].copy()
    if s.empty:
        return empty_fig("No salary data for selected filters.")

    ccys_in_data = s["currency"].unique().tolist()
    show_ccys = [c for c in ["GBP", "USD", "INR"] if c in ccys_in_data]

    fig = go.Figure()
    ccy_symbols = {"GBP": "£", "USD": "$", "INR": "₹"}
    for ccy in show_ccys:
        sub = s[s["currency"] == ccy]
        sym = ccy_symbols.get(ccy, "")
        for status, colour in [("Stated", COLOUR_STATED), ("Predicted", COLOUR_PRED)]:
            vals = sub[sub["salary_status"] == status]["salary_avg"].dropna()
            if vals.empty:
                continue
            fig.add_trace(go.Box(
                y=vals,
                name=f"{status} ({ccy})",
                marker_color=colour,
                opacity=0.75,
                boxmean=True,
                hovertemplate=(
                    f"<b>{{fullData.name}}</b><br>"
                    f"Median: {sym}%{{median:,.0f}}<extra></extra>"
                ),
            ))

    fig.update_layout(
        **CHART_LAYOUT,
        yaxis_title="Annual salary (see legend for currency)",
        boxmode="group",
    )
    return fig


# ---- Chart 8: Total vs unique reposts -------------------------------------

@app.callback(Output("chart-reposts", "figure"), *FILTER_INPUTS)
def chart_reposts(countries, keywords, job_types, years, currencies):
    d = apply_filters(DF, countries, keywords, job_types, years, currencies)
    if d.empty:
        return empty_fig()
    all_jt   = d["job_type"].value_counts().rename("All Postings")
    uniq_jt  = d[~d["is_repost"]]["job_type"].value_counts().rename("Unique Roles")
    compare  = pd.concat([all_jt, uniq_jt], axis=1).fillna(0).astype(int).reset_index()
    compare.columns = ["Job Type", "All Postings", "Unique Roles"]
    melted = compare.melt(id_vars="Job Type", var_name="View", value_name="Count")
    fig = px.bar(
        melted, x="Job Type", y="Count", color="View",
        barmode="group",
        color_discrete_map={"All Postings": "#1d4ed8", "Unique Roles": "#93c5fd"},
        text="Count",
        labels={"Count": "Count", "View": ""},
    )
    fig.update_traces(textposition="outside",
                      hovertemplate="<b>%{x}</b> — %{fullData.name}: %{y:,}<extra></extra>")
    fig.update_layout(**CHART_LAYOUT, xaxis_title="", yaxis_title="Count")
    return fig


# ---- Chart 9: Contract type + time ----------------------------------------

@app.callback(Output("chart-contract", "figure"), *FILTER_INPUTS)
def chart_contract(countries, keywords, job_types, years, currencies):
    d = apply_filters(DF, countries, keywords, job_types, years, currencies)
    if d.empty:
        return empty_fig()

    ct_vc  = d["contract_type"].value_counts().reset_index()
    ct_vc.columns = ["Value", "Count"]
    ct_vc["Field"] = "contract_type"

    ctm_vc = d["contract_time"].value_counts().reset_index()
    ctm_vc.columns = ["Value", "Count"]
    ctm_vc["Field"] = "contract_time"

    combined = pd.concat([ct_vc, ctm_vc], ignore_index=True)
    fig = px.bar(
        combined, x="Value", y="Count", color="Field",
        facet_col="Field",
        color_discrete_map={"contract_type": "#3b82f6", "contract_time": "#f59e0b"},
        text="Count",
        labels={"Value": "", "Count": "Count"},
    )
    fig.update_traces(textposition="outside",
                      hovertemplate="<b>%{x}</b>: %{y:,}<extra></extra>")
    fig.for_each_annotation(lambda a: a.update(
        text=a.text.replace("Field=contract_type", "Contract Type")
                   .replace("Field=contract_time", "Contract Time")
    ))
    fig.update_layout(**CHART_LAYOUT, showlegend=False,
                      xaxis_title="", yaxis_title="Count")
    return fig


# ---- Chart 10: Normalised salary (approx USD) -----------------------------

@app.callback(Output("chart-salary-norm", "figure"), *FILTER_INPUTS)
def chart_salary_norm(countries, keywords, job_types, years, currencies):
    d = apply_filters(DF, countries, keywords, job_types, years, currencies)
    s = d[d["salary_usd_approx"].notna()].copy()
    if s.empty:
        return empty_fig("No salary data for selected filters.")

    medians = (
        s.groupby(["country_name", "job_type"])["salary_usd_approx"]
        .median()
        .reset_index(name="Median USD (approx)")
    )

    fig = px.bar(
        medians, x="country_name", y="Median USD (approx)", color="job_type",
        barmode="group",
        color_discrete_map={"AI-Related": COLOUR_AI, "Non-AI": COLOUR_NONAI},
        text=medians["Median USD (approx)"].apply(lambda v: f"${v:,.0f}"),
        labels={"country_name": "Country", "Median USD (approx)": "Median Salary (~USD)",
                "job_type": "Job Type"},
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b> — %{fullData.name}<br>"
            "Median ~USD: $%{y:,.0f}<extra></extra>"
        ),
    )
    fig.update_layout(
        **CHART_LAYOUT,
        xaxis_title="",
        yaxis_title="Median salary (~USD, analytical estimate)",
        annotations=[dict(
            text="GBP x1.27 / INR x0.012 / fixed mid-2024 rates / NOT precise conversions",
            xref="paper", yref="paper", x=0.5, y=-0.18, showarrow=False,
            font=dict(size=10, color=COLOUR_MUTED), xanchor="center",
        )],
    )
    fig.update_layout(margin=dict(l=40, r=20, t=50, b=70))
    return fig


# ---------------------------------------------------------------------------
# 6.  RUN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  AI Job Market Analytics Dashboard")
    print("  IBM SkillBuild Project  |  Author: Jafar")
    print("=" * 60)
    print("  Open in your browser: http://127.0.0.1:8050")
    print("  Press Ctrl+C to stop.\n")
    app.run(debug=False, host="127.0.0.1", port=8050)
