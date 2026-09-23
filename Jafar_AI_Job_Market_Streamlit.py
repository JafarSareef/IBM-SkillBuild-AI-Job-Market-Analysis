# =============================================================================
# Jafar_AI_Job_Market_Streamlit.py
# AI Job Market Analytics – Streamlit Dashboard
# IBM SkillBuild Project | Author: Jafar
#
# Run:
#   streamlit run Jafar_AI_Job_Market_Streamlit.py
# =============================================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Job Market Analytics",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. CONSTANTS
# -----------------------------------------------------------------------------

RAW_DATA_FILE = "ai_job_market_dataset.csv"

SALARY_SUSPICIOUS_THRESHOLDS = {
    "GBP": 1000,
    "USD": 1000,
    "INR": 10000,
}

# Fixed mid-2024 FX rates for analytical normalization only
FX_TO_USD = {
    "GBP": 1.27,
    "USD": 1.00,
    "INR": 0.012
}

COLOUR_AI = "#2563eb"
COLOUR_NONAI = "#d97706"
COLOUR_PRED = "#7c3aed"
COLOUR_STATED = "#059669"

# -----------------------------------------------------------------------------
# 3. LOAD AND PREPARE DATA
# -----------------------------------------------------------------------------

@st.cache_data
def load_and_prepare():

    if not os.path.exists(RAW_DATA_FILE):
        st.error(
            f"Dataset not found: {RAW_DATA_FILE}. "
            "Make sure the CSV file is in the same folder as this Streamlit file."
        )
        st.stop()

    df = pd.read_csv(RAW_DATA_FILE)

    # Remove trailing unnamed columns
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df.drop(columns=unnamed, inplace=True)

    # Missing contract fields
    df["contract_type"] = df["contract_type"].fillna("Unknown")
    df["contract_time"] = df["contract_time"].fillna("Unknown")

    # Parse dates
    df["created"] = pd.to_datetime(
        df["created"],
        utc=True,
        errors="coerce"
    )

    df["created_date"] = pd.to_datetime(
        df["created_date"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    # Standardize AI flag
    df["is_ai_related"] = df["is_ai_related"].map({
        "TRUE": True,
        "FALSE": False,
        True: True,
        False: False
    })

    # Salary prediction flag
    df["salary_is_predicted"] = (
        df["salary_is_predicted"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": True, "FALSE": False})
        .fillna(False)
    )

    # Salary numeric conversion
    for col in ["salary_min", "salary_max", "salary_avg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Flag suspicious salary values
    df["salary_suspicious"] = False

    for currency, threshold in SALARY_SUSPICIOUS_THRESHOLDS.items():

        mask = (
            (df["currency"] == currency)
            & df["salary_avg"].notna()
            & (df["salary_avg"] < threshold)
        )

        df.loc[mask, "salary_suspicious"] = True

    # Derived columns
    df["job_type"] = df["is_ai_related"].map({
        True: "AI-Related",
        False: "Non-AI"
    })

    df["salary_status"] = df["salary_is_predicted"].map({
        True: "Predicted",
        False: "Stated"
    })

    df["year"] = df["created_date"].dt.year
    df["month"] = df["created_date"].dt.month

    df["year_month"] = (
        df["created_date"]
        .dt.to_period("M")
        .astype(str)
    )

    # Re-post flag
    df["is_repost"] = df.duplicated(
        subset=["title", "company"],
        keep=False
    )

    # Approximate USD salary
    df["salary_usd_approx"] = df.apply(
        lambda row:
            row["salary_avg"]
            * FX_TO_USD.get(row["currency"], 1.0)
            if (
                pd.notna(row["salary_avg"])
                and not row["salary_suspicious"]
            )
            else np.nan,
        axis=1
    )

    return df


DF = load_and_prepare()

# -----------------------------------------------------------------------------
# 4. FILTER OPTIONS
# -----------------------------------------------------------------------------

ALL_COUNTRIES = sorted(
    DF["country_name"].dropna().unique().tolist()
)

ALL_KEYWORDS = sorted(
    DF["search_keyword"].dropna().unique().tolist()
)

ALL_JOB_TYPES = ["AI-Related", "Non-AI"]

ALL_YEARS = sorted(
    [int(y) for y in DF["year"].dropna().unique()]
)

ALL_CURRENCIES = sorted(
    DF["currency"].dropna().unique().tolist()
)

# -----------------------------------------------------------------------------
# 5. FILTER FUNCTION
# -----------------------------------------------------------------------------

def apply_filters(
    df,
    countries,
    keywords,
    job_types,
    years,
    currencies
):

    filtered = df.copy()

    if countries:
        filtered = filtered[
            filtered["country_name"].isin(countries)
        ]

    if keywords:
        filtered = filtered[
            filtered["search_keyword"].isin(keywords)
        ]

    if job_types:
        filtered = filtered[
            filtered["job_type"].isin(job_types)
        ]

    if years:
        filtered = filtered[
            filtered["year"].isin(years)
        ]

    if currencies:
        filtered = filtered[
            filtered["currency"].isin(currencies)
        ]

    return filtered


# -----------------------------------------------------------------------------
# 6. TITLE
# -----------------------------------------------------------------------------

st.title("🤖 AI Job Market Analytics")

st.caption(
    "Exploring how AI-related job postings differ from traditional postings "
    "across countries, industries, keywords, and salary."
)

st.divider()

# -----------------------------------------------------------------------------
# 7. SIDEBAR FILTERS
# -----------------------------------------------------------------------------

st.sidebar.header("🔎 Filters")

# Initialize filter state only once.
for _key in [
    "country_filter",
    "keyword_filter",
    "job_type_filter",
    "year_filter",
    "currency_filter",
]:
    if _key not in st.session_state:
        st.session_state[_key] = "All"


def reset_filters():
    """Reset every sidebar filter before Streamlit reruns the app."""
    st.session_state["country_filter"] = "All"
    st.session_state["keyword_filter"] = "All"
    st.session_state["job_type_filter"] = "All"
    st.session_state["year_filter"] = "All"
    st.session_state["currency_filter"] = "All"


st.sidebar.selectbox(
    "Country",
    ["All"] + ALL_COUNTRIES,
    key="country_filter"
)

st.sidebar.selectbox(
    "Search Keyword",
    ["All"] + ALL_KEYWORDS,
    key="keyword_filter"
)

st.sidebar.selectbox(
    "Job Type",
    ["All"] + ALL_JOB_TYPES,
    key="job_type_filter"
)

st.sidebar.selectbox(
    "Year",
    ["All"] + ALL_YEARS,
    key="year_filter"
)

st.sidebar.selectbox(
    "Currency",
    ["All"] + ALL_CURRENCIES,
    key="currency_filter"
)

st.sidebar.button(
    "Reset Filters",
    on_click=reset_filters,
    use_container_width=True
)

# Read the current filter selections.
country = st.session_state["country_filter"]
keyword = st.session_state["keyword_filter"]
job_type = st.session_state["job_type_filter"]
year = st.session_state["year_filter"]
currency = st.session_state["currency_filter"]

# Convert "All" into an empty list so that dimension is not filtered.
countries = [] if country == "All" else [country]
keywords = [] if keyword == "All" else [keyword]
job_types = [] if job_type == "All" else [job_type]
years = [] if year == "All" else [int(year)]
currencies = [] if currency == "All" else [currency]

# -----------------------------------------------------------------------------
# 8. APPLY FILTERS
# -----------------------------------------------------------------------------

D = apply_filters(
    DF,
    countries,
    keywords,
    job_types,
    years,
    currencies
)

# -----------------------------------------------------------------------------
# 9. KPI CALCULATIONS
# -----------------------------------------------------------------------------

total = len(D)

ai_count = int(D["is_ai_related"].sum())

non_ai_count = total - ai_count

ai_percentage = (
    ai_count / total * 100
    if total > 0
    else 0
)

unique_roles = int(
    (~D["is_repost"]).sum()
)

reposts = int(
    D["is_repost"].sum()
)

# -----------------------------------------------------------------------------
# 10. KPI CARDS
# -----------------------------------------------------------------------------

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric(
    "Total Postings",
    f"{total:,}"
)

col2.metric(
    "AI-Related",
    f"{ai_count:,}"
)

col3.metric(
    "Non-AI",
    f"{non_ai_count:,}"
)

col4.metric(
    "AI %",
    f"{ai_percentage:.1f}%"
)

col5.metric(
    "Unique Roles",
    f"{unique_roles:,}"
)

col6.metric(
    "Re-posts",
    f"{reposts:,}"
)

st.divider()

# -----------------------------------------------------------------------------
# 11. EMPTY DATA CHECK
# -----------------------------------------------------------------------------

if D.empty:

    st.warning(
        "No data available for the selected filters. "
        "Please change the filter selections."
    )

    st.stop()

# -----------------------------------------------------------------------------
# 12. AI VS NON-AI OVERALL
# -----------------------------------------------------------------------------

st.subheader("AI-Related vs Non-AI — Overall")

overall = (
    D["job_type"]
    .value_counts()
    .reset_index()
)

overall.columns = [
    "Job Type",
    "Count"
]

fig_overall = px.pie(
    overall,
    names="Job Type",
    values="Count",
    color="Job Type",
    color_discrete_map={
        "AI-Related": COLOUR_AI,
        "Non-AI": COLOUR_NONAI
    },
    hole=0.42
)

fig_overall.update_traces(
    textposition="outside",
    textinfo="percent+label"
)

st.plotly_chart(
    fig_overall,
    use_container_width=True
)

# -----------------------------------------------------------------------------
# 13. COUNTRY ANALYSIS
# -----------------------------------------------------------------------------

st.subheader("AI-Related vs Non-AI by Country")

country_data = (
    D.groupby(
        ["country_name", "job_type"]
    )
    .size()
    .reset_index(name="Count")
)

fig_country = px.bar(
    country_data,
    x="country_name",
    y="Count",
    color="job_type",
    barmode="group",
    text="Count",
    color_discrete_map={
        "AI-Related": COLOUR_AI,
        "Non-AI": COLOUR_NONAI
    },
    labels={
        "country_name": "Country",
        "Count": "Postings",
        "job_type": "Job Type"
    }
)

fig_country.update_traces(
    textposition="outside"
)

st.plotly_chart(
    fig_country,
    use_container_width=True
)

# -----------------------------------------------------------------------------
# 14. KEYWORD ANALYSIS
# -----------------------------------------------------------------------------

st.subheader("AI Proportion by Search Keyword")

keyword_data = (
    D.groupby(
        ["search_keyword", "job_type"]
    )
    .size()
    .reset_index(name="Count")
)

keyword_totals = (
    keyword_data
    .groupby("search_keyword")["Count"]
    .sum()
    .rename("Total")
)

keyword_data = keyword_data.join(
    keyword_totals,
    on="search_keyword"
)

keyword_data["AI %"] = (
    keyword_data["Count"]
    / keyword_data["Total"]
    * 100
)

ai_keywords = keyword_data[
    keyword_data["job_type"] == "AI-Related"
].copy()

ai_keywords = ai_keywords.sort_values(
    "AI %",
    ascending=True
)

if not ai_keywords.empty:

    fig_keyword = px.bar(
        ai_keywords,
        y="search_keyword",
        x="AI %",
        orientation="h",
        text=ai_keywords["AI %"].apply(
            lambda x: f"{x:.1f}%"
        ),
        labels={
            "search_keyword": "Keyword",
            "AI %": "AI-Related %"
        }
    )

    fig_keyword.update_traces(
        textposition="outside"
    )

    fig_keyword.update_layout(
        xaxis_range=[0, 115]
    )

    st.plotly_chart(
        fig_keyword,
        use_container_width=True
    )

# -----------------------------------------------------------------------------
# 15. CATEGORY ANALYSIS
# -----------------------------------------------------------------------------

st.subheader("AI-Related Jobs by Category — Top 15")

category_data = (
    D[D["job_type"] == "AI-Related"]
    .groupby("category")
    .size()
    .reset_index(name="Count")
    .sort_values(
        "Count",
        ascending=False
    )
    .head(15)
)

category_data = category_data.sort_values(
    "Count",
    ascending=True
)

fig_category = px.bar(
    category_data,
    y="category",
    x="Count",
    orientation="h",
    text="Count",
    labels={
        "category": "Category",
        "Count": "AI-Related Postings"
    }
)

fig_category.update_traces(
    textposition="outside"
)

st.plotly_chart(
    fig_category,
    use_container_width=True
)

# -----------------------------------------------------------------------------
# 16. MONTHLY TREND
# -----------------------------------------------------------------------------

st.subheader("Monthly Posting Trend")

trend_data = (
    D.groupby(
        ["year_month", "job_type"]
    )
    .size()
    .reset_index(name="Count")
    .sort_values("year_month")
)

fig_trend = px.line(
    trend_data,
    x="year_month",
    y="Count",
    color="job_type",
    markers=True,
    color_discrete_map={
        "AI-Related": COLOUR_AI,
        "Non-AI": COLOUR_NONAI
    },
    labels={
        "year_month": "Month",
        "Count": "Postings",
        "job_type": "Job Type"
    }
)

st.plotly_chart(
    fig_trend,
    use_container_width=True
)

# -----------------------------------------------------------------------------
# 17. TOTAL VS UNIQUE POSTINGS
# -----------------------------------------------------------------------------

st.subheader("Total vs Unique Non-Reposted Postings")

all_job_types = (
    D["job_type"]
    .value_counts()
    .rename("All Postings")
)

unique_job_types = (
    D[~D["is_repost"]]["job_type"]
    .value_counts()
    .rename("Unique Roles")
)

repost_compare = pd.concat(
    [
        all_job_types,
        unique_job_types
    ],
    axis=1
).fillna(0).astype(int)

repost_compare = (
    repost_compare
    .reset_index()
)

repost_compare.columns = [
    "Job Type",
    "All Postings",
    "Unique Roles"
]

repost_melted = repost_compare.melt(
    id_vars="Job Type",
    var_name="View",
    value_name="Count"
)

fig_reposts = px.bar(
    repost_melted,
    x="Job Type",
    y="Count",
    color="View",
    barmode="group",
    text="Count"
)

fig_reposts.update_traces(
    textposition="outside"
)

st.plotly_chart(
    fig_reposts,
    use_container_width=True
)

# -----------------------------------------------------------------------------
# 18. SALARY ANALYSIS
# -----------------------------------------------------------------------------

st.subheader(
    "Salary Distribution — AI vs Non-AI"
)

salary_data = D[
    D["salary_avg"].notna()
    & (~D["salary_suspicious"])
].copy()

if not salary_data.empty:

    salary_fig = go.Figure()

    currency_symbols = {
        "GBP": "£",
        "USD": "$",
        "INR": "₹"
    }

    for currency in ["GBP", "USD", "INR"]:

        currency_data = salary_data[
            salary_data["currency"] == currency
        ]

        for job_type, label_color in [
            ("AI-Related", COLOUR_AI),
            ("Non-AI", COLOUR_NONAI)
        ]:

            values = currency_data[
                currency_data["job_type"] == job_type
            ]["salary_avg"].dropna()

            if values.empty:
                continue

            symbol = currency_symbols.get(
                currency,
                ""
            )

            salary_fig.add_trace(
                go.Box(
                    y=values,
                    name=f"{job_type} ({currency})",
                    boxmean=True,
                    marker_color=label_color,
                    opacity=0.75
                )
            )

    salary_fig.update_layout(
        yaxis_title="Annual Salary",
        boxmode="group"
    )

    st.plotly_chart(
        salary_fig,
        use_container_width=True
    )

    st.caption(
        "Salary charts keep GBP, USD and INR separate. "
        "Four suspicious salary records are excluded from salary distributions."
    )

else:

    st.info("No salary data available for the selected filters.")

# -----------------------------------------------------------------------------
# 19. PREDICTED VS STATED SALARY
# -----------------------------------------------------------------------------

st.subheader("Predicted vs Stated Salary")

if not salary_data.empty:

    predicted_fig = go.Figure()

    for currency in ["GBP", "USD", "INR"]:

        currency_data = salary_data[
            salary_data["currency"] == currency
        ]

        for status, label_color in [
            ("Stated", COLOUR_STATED),
            ("Predicted", COLOUR_PRED)
        ]:

            values = currency_data[
                currency_data["salary_status"] == status
            ]["salary_avg"].dropna()

            if values.empty:
                continue

            predicted_fig.add_trace(
                go.Box(
                    y=values,
                    name=f"{status} ({currency})",
                    boxmean=True,
                    marker_color=label_color,
                    opacity=0.75
                )
            )

    predicted_fig.update_layout(
        yaxis_title="Annual Salary",
        boxmode="group"
    )

    st.plotly_chart(
        predicted_fig,
        use_container_width=True
    )

    predicted_count = len(
        salary_data[
            salary_data["salary_status"] == "Predicted"
        ]
    )

    salary_count = len(salary_data)

    predicted_percentage = (
        predicted_count / salary_count * 100
        if salary_count > 0
        else 0
    )

    st.caption(
        f"{predicted_percentage:.1f}% of salary-valid, "
        "non-suspicious records are algorithmically predicted."
    )

# -----------------------------------------------------------------------------
# 20. CONTRACT ANALYSIS
# -----------------------------------------------------------------------------

st.subheader(
    "Contract Type & Contract Time"
)

contract_col1, contract_col2 = st.columns(2)

with contract_col1:

    contract_type_data = (
        D["contract_type"]
        .value_counts()
        .reset_index()
    )

    contract_type_data.columns = [
        "Contract Type",
        "Count"
    ]

    fig_contract_type = px.bar(
        contract_type_data,
        x="Contract Type",
        y="Count",
        text="Count"
    )

    fig_contract_type.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_contract_type,
        use_container_width=True
    )

with contract_col2:

    contract_time_data = (
        D["contract_time"]
        .value_counts()
        .reset_index()
    )

    contract_time_data.columns = [
        "Contract Time",
        "Count"
    ]

    fig_contract_time = px.bar(
        contract_time_data,
        x="Contract Time",
        y="Count",
        text="Count"
    )

    fig_contract_time.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_contract_time,
        use_container_width=True
    )

st.caption(
    "Contract fields contain substantial Unknown values, "
    "so these charts are descriptive only."
)

# -----------------------------------------------------------------------------
# 21. APPROXIMATE USD NORMALIZED SALARY
# -----------------------------------------------------------------------------

st.subheader(
    "Approximate Salary — USD Normalized"
)

st.warning(
    "ANALYTICAL ESTIMATE ONLY: GBP × 1.27, INR × 0.012, "
    "USD × 1.00 using fixed mid-2024 rates. "
    "This is not a precise currency conversion."
)

normalized_data = D[
    D["salary_usd_approx"].notna()
].copy()

if not normalized_data.empty:

    median_salary = (
        normalized_data
        .groupby(
            ["country_name", "job_type"]
        )["salary_usd_approx"]
        .median()
        .reset_index()
    )

    median_salary.columns = [
        "Country",
        "Job Type",
        "Median USD Approx"
    ]

    fig_normalized = px.bar(
        median_salary,
        x="Country",
        y="Median USD Approx",
        color="Job Type",
        barmode="group",
        text=median_salary[
            "Median USD Approx"
        ].apply(
            lambda x: f"${x:,.0f}"
        ),
        color_discrete_map={
            "AI-Related": COLOUR_AI,
            "Non-AI": COLOUR_NONAI
        }
    )

    fig_normalized.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_normalized,
        use_container_width=True
    )

# -----------------------------------------------------------------------------
# 22. DATA LIMITATIONS
# -----------------------------------------------------------------------------

st.divider()

st.subheader("⚠️ Data Limitations")

st.markdown("""
- Salaries are reported in **GBP, USD and INR** and should not be directly compared.
- The approximate USD chart uses fixed mid-2024 exchange rates only.
- Many salary values are algorithmically predicted rather than employer-stated.
- Contract fields contain substantial missing/Unknown values.
- The dataset uses only five search keywords.
- Re-posted job listings are preserved in the dataset.
- Four suspicious salary records are excluded from salary distributions.
- The dataset should not be treated as a complete representation of the global job market.
- The analysis is descriptive and does **not establish that AI causes employment changes**.
""")

# -----------------------------------------------------------------------------
# 23. PROJECT INFORMATION
# -----------------------------------------------------------------------------

st.divider()

st.caption(
    "AI Job Market Analytics | IBM SkillBuild Project | Author: Jafar"
)

st.caption(
    "Dataset: AI Job Market 2026 — Automation vs Traditional Roles | "
    "UK / US / India"
)

st.caption(
    f"Current filtered dataset: {len(D):,} postings"
)