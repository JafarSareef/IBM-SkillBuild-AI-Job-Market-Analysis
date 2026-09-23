# =============================================================================
# Jafar_AI_Job_Market_Analytics.py
# AI Job Market Analytics – IBM SkillBuild Project
# Author : Jafar
# Purpose: Load, inspect, and preprocess the AI job market dataset;
#          perform exploratory data analysis; generate charts.
# =============================================================================
# IMPORTANT NOTES
#   • The original CSV file is NEVER modified.
#   • Salaries are in three different currencies (GBP, USD, INR).
#     They are NOT compared directly across currencies.
#   • Currency normalisation (to USD) is used ONLY for cross-country
#     visualisations and is clearly labelled as an analytical estimate.
#   • is_ai_related and salary_is_predicted are preserved as-is from the source.
#   • No causal claims are made. All analysis is descriptive.
# =============================================================================

import io
import os
import sys
import warnings
import textwrap

# Ensure UTF-8 output on Windows terminals that default to cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ---------------------------------------------------------------------------
# 0.  CONFIGURATION
# ---------------------------------------------------------------------------

RAW_DATA_FILE = "ai_job_market_dataset.csv"

# Output directory for charts
CHARTS_DIR = "charts"

# Salary thresholds used to flag suspicious values (per currency, annual).
# These are intentionally conservative to flag only clear anomalies.
SALARY_SUSPICIOUS_THRESHOLDS = {
    "GBP": 1_000,    # below £1,000 annual is almost certainly hourly/daily
    "USD": 1_000,    # below $1,000 annual
    "INR": 10_000,   # below ₹10,000 annual (very low even for India)
}

# Approximate exchange rates used ONLY for cross-country normalisation charts.
# These are fixed analytical estimates, NOT live/official rates.
# Source: approximate mid-market rates as of mid-2024 for consistency.
# Label all outputs derived from these as "Approx. USD equivalent".
FX_TO_USD = {
    "GBP": 1.27,   # 1 GBP ~ 1.27 USD
    "USD": 1.00,
    "INR": 0.012,  # 1 INR ~ 0.012 USD
}

# Minimum group size for category-level proportion analysis
MIN_CATEGORY_SIZE = 10

# Columns that are allowed to be NaN without being considered problematic
NULLABLE_COLUMNS = {"contract_type", "contract_time", "salary_min", "salary_max", "salary_avg"}

# ---------------------------------------------------------------------------
# 1.  HELPER UTILITIES
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    """Print a clearly visible section heading to stdout."""
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def subsection(title: str) -> None:
    """Print a subsection heading."""
    print(f"\n--- {title} ---")


def indent(text: str, spaces: int = 4) -> None:
    """Print text indented by a fixed number of spaces."""
    prefix = " " * spaces
    for line in str(text).splitlines():
        print(prefix + line)


# ---------------------------------------------------------------------------
# 2.  LOAD RAW DATA
# ---------------------------------------------------------------------------

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Read the original CSV without modification.
    The trailing comma in the header produces an unnamed last column;
    it is dropped immediately.
    """
    section("STEP 1 – LOAD RAW DATA")
    print(f"  File : {os.path.abspath(filepath)}")

    if not os.path.exists(filepath):
        sys.exit(f"[ERROR] Dataset not found: {filepath}")

    df = pd.read_csv(filepath)

    # Drop the spurious empty column caused by the trailing comma in the header
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)
        print(f"  Dropped {len(unnamed_cols)} unnamed trailing column(s): {unnamed_cols}")

    print(f"  Shape after load : {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# 3.  INITIAL INSPECTION
# ---------------------------------------------------------------------------

def inspect_raw(df: pd.DataFrame) -> None:
    """Print a concise but informative overview of the raw dataset."""
    section("STEP 2 – INITIAL INSPECTION")

    subsection("Column names and dtypes")
    for col in df.columns:
        indent(f"{col:<30} {str(df[col].dtype):<12} "
               f"nunique={df[col].nunique():>5}  "
               f"null={df[col].isna().sum():>5}")

    subsection("Missing values")
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    missing_df = pd.DataFrame({"missing_count": missing, "missing_%": missing_pct})
    missing_df = missing_df[missing_df["missing_count"] > 0]
    if missing_df.empty:
        indent("No missing values found.")
    else:
        indent(missing_df.to_string())

    subsection("Categorical column distributions")
    cat_cols = ["search_keyword", "search_country", "country_name",
                "category", "contract_type", "contract_time",
                "currency", "is_ai_related", "salary_is_predicted"]
    for col in cat_cols:
        if col in df.columns:
            print(f"\n  [{col}]")
            vc = df[col].value_counts(dropna=False)
            for val, cnt in vc.items():
                indent(f"{str(val):<40} {cnt:>6}  ({cnt/len(df)*100:.1f}%)", 6)

    subsection("Salary summary (raw, mixed currencies – informational only)")
    for col in ["salary_min", "salary_max", "salary_avg"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            indent(f"{col}: min={s.min():.2f}  max={s.max():.2f}  "
                   f"mean={s.mean():.2f}  non-null={s.notna().sum()}")

    subsection("Date range")
    if "created_date" in df.columns:
        dates = pd.to_datetime(df["created_date"], format="%d-%m-%Y", errors="coerce")
        indent(f"Earliest : {dates.min().date()}")
        indent(f"Latest   : {dates.max().date()}")


# ---------------------------------------------------------------------------
# 4.  DUPLICATE DETECTION
# ---------------------------------------------------------------------------

def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and report two kinds of duplication:
      (a) Exact row duplicates   – same values in every column
      (b) Same-role re-posts     – same id appearing more than once
    Re-posts are NOT removed; they represent genuine temporal re-listings.
    """
    section("STEP 3 – DUPLICATE DETECTION")

    # (a) Exact row duplicates
    exact_mask = df.duplicated(keep=False)
    n_exact = exact_mask.sum()
    subsection(f"(a) Exact row duplicates : {n_exact}")
    if n_exact > 0:
        print("  [INFO] Exact duplicates detected – first occurrence kept, "
              "rest marked in 'exact_duplicate' flag column.")
        df["exact_duplicate"] = df.duplicated(keep="first")
    else:
        df["exact_duplicate"] = False
        indent("No exact duplicates found – all rows are unique.")

    # (b) Same-role re-posts (same title + company, different id / date)
    repost_mask = df.duplicated(subset=["title", "company"], keep=False)
    n_repost_rows = repost_mask.sum()
    n_repost_groups = df[repost_mask].groupby(["title", "company"]).ngroups
    subsection(f"(b) Same title+company re-posts : "
               f"{n_repost_groups} role groups → {n_repost_rows} rows")
    indent("These are preserved as legitimate temporal re-listings.")
    df["is_repost"] = repost_mask

    return df


# ---------------------------------------------------------------------------
# 5.  CLEAN AND STANDARDISE
# ---------------------------------------------------------------------------

def clean_and_standardise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all data-quality fixes:
      • Fill missing contract_type / contract_time with 'Unknown'
      • Parse date columns to proper Python types
      • Standardise boolean columns (is_ai_related, salary_is_predicted)
      • Coerce salary columns to float
    """
    section("STEP 4 – CLEAN AND STANDARDISE")

    # --- 4a. Missing contract fields → 'Unknown' -------------------------
    subsection("4a. Missing contract_type / contract_time → 'Unknown'")
    for col in ["contract_type", "contract_time"]:
        n_filled = df[col].isna().sum()
        df[col] = df[col].fillna("Unknown")
        indent(f"{col}: {n_filled} blanks filled with 'Unknown'")

    # --- 4b. Parse date columns ------------------------------------------
    subsection("4b. Parse date columns")

    # 'created' is a full timezone-aware timestamp
    df["created"] = pd.to_datetime(df["created"], utc=True, errors="coerce")
    n_bad_created = df["created"].isna().sum()
    indent(f"'created' parsed as UTC datetime  (parse failures: {n_bad_created})")

    # 'created_date' is a date-only string in DD-MM-YYYY format
    df["created_date"] = pd.to_datetime(
        df["created_date"], format="%d-%m-%Y", errors="coerce"
    ).dt.date
    n_bad_date = pd.Series(df["created_date"]).isna().sum()
    indent(f"'created_date' parsed as date     (parse failures: {n_bad_date})")

    # --- 4c. Standardise boolean / binary columns -------------------------
    subsection("4c. Standardise boolean / binary columns")

    # is_ai_related: 'TRUE'/'FALSE' strings → Python bool
    df["is_ai_related"] = df["is_ai_related"].map(
        {"TRUE": True, "FALSE": False, True: True, False: False}
    )
    indent(f"'is_ai_related' → bool  "
           f"(True={df['is_ai_related'].sum()}, False={(~df['is_ai_related']).sum()})")

    # salary_is_predicted: int 0/1 → Python bool
    df["salary_is_predicted"] = df["salary_is_predicted"].astype(bool)
    indent(f"'salary_is_predicted' → bool  "
           f"(Predicted={df['salary_is_predicted'].sum()}, "
           f"Stated={(~df['salary_is_predicted']).sum()})")

    # --- 4d. Coerce salary columns to float --------------------------------
    subsection("4d. Coerce salary columns to float")
    for col in ["salary_min", "salary_max", "salary_avg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        indent(f"'{col}' coerced to float  (non-null: {df[col].notna().sum()})")

    return df


# ---------------------------------------------------------------------------
# 6.  FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new derived columns useful for analysis and visualisation:
      • job_type        : human-readable AI vs Non-AI label
      • salary_status   : whether salary was stated or predicted
      • year            : listing year (integer)
      • month           : listing month (integer)
      • year_month      : YYYY-MM string for time-series grouping
    """
    section("STEP 5 – FEATURE ENGINEERING")

    new_cols = []

    # job_type
    df["job_type"] = df["is_ai_related"].map(
        {True: "AI-Related", False: "Non-AI"}
    )
    new_cols.append("job_type")
    indent("'job_type'     : AI-Related / Non-AI  <- from is_ai_related")

    # salary_status
    df["salary_status"] = df["salary_is_predicted"].map(
        {True: "Predicted", False: "Stated"}
    )
    new_cols.append("salary_status")
    indent("'salary_status': Predicted / Stated   <- from salary_is_predicted")

    # year / month / year_month from created_date
    created_dt = pd.to_datetime(df["created_date"], errors="coerce")
    df["year"] = created_dt.dt.year.astype("Int64")  # nullable integer
    df["month"] = created_dt.dt.month.astype("Int64")
    df["year_month"] = created_dt.dt.to_period("M").astype(str)
    new_cols += ["year", "month", "year_month"]
    indent("'year'         : listing year          <- from created_date")
    indent("'month'        : listing month         <- from created_date")
    indent("'year_month'   : YYYY-MM period        <- from created_date")

    subsection("New columns added")
    indent(", ".join(new_cols))

    return df


# ---------------------------------------------------------------------------
# 7.  SUSPICIOUS SALARY DETECTION
# ---------------------------------------------------------------------------

def flag_suspicious_salaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag rows where salary_avg is below a per-currency threshold.
    These are NOT deleted – they are marked with a boolean flag column
    'salary_suspicious' and reported for manual review.

    Rationale: very low values (e.g. £13, £20.50, $32) are almost certainly
    hourly or daily rates stored incorrectly as annual salaries.
    No assumption is made about the cause; they are flagged for the analyst.
    """
    section("STEP 6 – SUSPICIOUS SALARY DETECTION")

    df["salary_suspicious"] = False

    suspicious_rows = []
    for currency, threshold in SALARY_SUSPICIOUS_THRESHOLDS.items():
        mask = (
            (df["currency"] == currency) &
            df["salary_avg"].notna() &
            (df["salary_avg"] < threshold)
        )
        df.loc[mask, "salary_suspicious"] = True
        flagged = df[mask][["id", "title", "company", "country_name",
                             "currency", "salary_avg"]]
        if not flagged.empty:
            suspicious_rows.append(flagged)

    n_suspicious = df["salary_suspicious"].sum()
    subsection(f"Total suspicious salary records flagged : {n_suspicious}")

    if suspicious_rows:
        combined = pd.concat(suspicious_rows, ignore_index=True)
        indent(combined.to_string(index=False))
        indent("\n[NOTE] These rows are preserved. Treat salary values with "
               "caution in downstream analysis – possible hourly/daily rates.")
    else:
        indent("No suspicious salary values detected.")

    return df


# ---------------------------------------------------------------------------
# 8.  FINAL DATASET SUMMARY
# ---------------------------------------------------------------------------

def final_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Print a concise before/after summary of the preprocessing pipeline."""
    section("STEP 7 – PREPROCESSING SUMMARY")

    subsection("Row counts")
    indent(f"Rows before preprocessing : {len(df_raw):,}")
    indent(f"Rows after  preprocessing : {len(df_clean):,}  "
           "(no rows deleted – original data preserved)")

    subsection("Column counts")
    indent(f"Columns before : {len(df_raw.columns)}")
    indent(f"Columns after  : {len(df_clean.columns)}")
    new_cols = [c for c in df_clean.columns if c not in df_raw.columns]
    indent(f"New columns added ({len(new_cols)}) : {new_cols}")

    subsection("Exact duplicates")
    indent(f"{df_clean['exact_duplicate'].sum()} exact duplicate rows "
           "(flagged, NOT deleted)")

    subsection("Re-posted roles")
    n_reposts = df_clean["is_repost"].sum()
    indent(f"{n_reposts} rows are temporal re-posts of the same role "
           "(flagged, NOT deleted – legitimate business data)")

    subsection("Suspicious salary records")
    n_susp = df_clean["salary_suspicious"].sum()
    indent(f"{n_susp} rows flagged as potentially suspicious salary values "
           "(retained with 'salary_suspicious' = True)")

    subsection("job_type distribution")
    vc = df_clean["job_type"].value_counts()
    for val, cnt in vc.items():
        indent(f"{val:<20} {cnt:>5}  ({cnt/len(df_clean)*100:.1f}%)", 6)

    subsection("salary_status distribution")
    vc2 = df_clean["salary_status"].value_counts()
    for val, cnt in vc2.items():
        indent(f"{val:<20} {cnt:>5}  ({cnt/len(df_clean)*100:.1f}%)", 6)

    subsection("Date range in cleaned data")
    created_dt = pd.to_datetime(df_clean["created_date"], errors="coerce")
    indent(f"Earliest listing : {created_dt.min().date()}")
    indent(f"Latest  listing  : {created_dt.max().date()}")

    subsection("Missing values remaining (columns with > 0 nulls)")
    missing = df_clean.isna().sum()
    remaining = missing[missing > 0]
    if remaining.empty:
        indent("None – all expected nulls have been handled.")
    else:
        for col, cnt in remaining.items():
            indent(f"{col:<30} {cnt:>5} null(s)  "
                   f"({cnt/len(df_clean)*100:.1f}%)")

    subsection("Issues to address before analysis stage")
    issues = [
        "1. Salaries are in GBP, USD, and INR – do NOT compare across "
           "currencies without conversion. Consider adding a converted "
           "'salary_usd' column using exchange rates for cross-country analysis.",
        "2. contract_type (74.5% 'Unknown') and contract_time (51.3% 'Unknown') "
           "are too sparse for reliable segmentation – treat with caution.",
        "3. salary_is_predicted = True for ~68.5% of rows – salary figures for "
           "those rows are model estimates, not employer-stated values.",
        f"4. {df_clean['salary_suspicious'].sum()} rows have suspiciously low "
           "salary_avg values (likely hourly/daily rates). Exclude or treat "
           "separately in salary-based analysis.",
        "5. 783 rows represent temporal re-posts of the same role. For counting "
           "distinct job openings, consider deduplicating on title+company.",
        "6. descriptions are truncated in the CSV (~500 chars). Full-text NLP "
           "analysis of job descriptions will be limited.",
    ]
    for issue in issues:
        print()
        for line in textwrap.wrap(issue, width=72, subsequent_indent="      "):
            indent(line, 4)

    print()


# ---------------------------------------------------------------------------
# 9.  EDA UTILITIES
# ---------------------------------------------------------------------------

def ensure_charts_dir() -> str:
    """Create the charts output directory if it does not exist."""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return CHARTS_DIR


def save_fig(fig, filename: str) -> str:
    """Save a matplotlib figure to CHARTS_DIR and close it."""
    import matplotlib.pyplot as plt
    path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [chart saved] {path}")
    return path


def salary_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a salary-analysis-ready slice:
      - salary_avg is not null
      - salary_suspicious is False
    Used for all per-currency salary charts.
    """
    return df[df["salary_avg"].notna() & (~df["salary_suspicious"])].copy()


def salary_normalised(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'salary_usd_approx' column using fixed FX_TO_USD rates.
    Returns only clean (non-suspicious, non-null) salary rows.
    IMPORTANT: This column is an analytical normalisation only.
    It uses fixed mid-2024 approximate rates and must NOT be presented
    as exact converted values.
    """
    s = salary_clean(df).copy()
    s["salary_usd_approx"] = s.apply(
        lambda r: r["salary_avg"] * FX_TO_USD.get(r["currency"], 1.0), axis=1
    )
    return s


# ---------------------------------------------------------------------------
# 10.  EDA ANALYSIS FUNCTIONS
# ---------------------------------------------------------------------------

# ---- 10.1  AI vs Non-AI overview -----------------------------------------

def eda_ai_vs_nonai(df: pd.DataFrame) -> dict:
    """
    Analysis 1: AI-Related vs Non-AI job postings.
    Reports overall counts, by country, and by category.
    Uses both all postings and unique (non-repost) postings.
    """
    section("EDA 1 – AI-RELATED vs NON-AI JOB POSTINGS")
    results = {}

    # ---- Overall (all postings) ----
    subsection("1a. Overall counts – ALL postings")
    total = len(df)
    vc = df["job_type"].value_counts()
    for label, cnt in vc.items():
        pct = cnt / total * 100
        indent(f"{label:<20} {cnt:>5}  ({pct:.1f}%)")
        results[f"overall_{label.replace('-','_').replace(' ','_')}"] = cnt
    results["overall_total"] = total
    indent(f"Total postings : {total:,}")

    # ---- Unique / non-repost postings ----
    subsection("1b. Unique role counts – non-repost postings only (is_repost=False)")
    df_unique = df[~df["is_repost"]]
    total_u = len(df_unique)
    vc_u = df_unique["job_type"].value_counts()
    for label, cnt in vc_u.items():
        pct = cnt / total_u * 100
        indent(f"{label:<20} {cnt:>5}  ({pct:.1f}%)")
        results[f"unique_{label.replace('-','_').replace(' ','_')}"] = cnt
    results["unique_total"] = total_u
    indent(f"  [filter: is_repost == False → {total_u:,} rows]")

    # ---- By country ----
    subsection("1c. AI-Related vs Non-AI by country (all postings)")
    ct = df.groupby(["country_name", "job_type"]).size().unstack(fill_value=0)
    ct["Total"] = ct.sum(axis=1)
    for col in ["AI-Related", "Non-AI"]:
        if col in ct.columns:
            ct[f"{col} %"] = (ct[col] / ct["Total"] * 100).round(1)
    indent(ct.to_string())
    results["by_country"] = ct.to_dict()

    # ---- By category ----
    subsection("1d. AI-Related vs Non-AI by category (all postings, sorted by AI count)")
    cat = df.groupby(["category", "job_type"]).size().unstack(fill_value=0)
    cat["Total"] = cat.sum(axis=1)
    if "AI-Related" in cat.columns:
        cat["AI %"] = (cat["AI-Related"] / cat["Total"] * 100).round(1)
        cat = cat.sort_values("AI-Related", ascending=False)
    indent(cat.to_string())
    results["by_category"] = cat.to_dict()

    return results


# ---- 10.2  Search keyword analysis ----------------------------------------

def eda_keywords(df: pd.DataFrame) -> dict:
    """
    Analysis 2: Compare search keywords by AI-related proportion.
    """
    section("EDA 2 – SEARCH KEYWORD ANALYSIS")
    results = {}

    kw = df.groupby(["search_keyword", "job_type"]).size().unstack(fill_value=0)
    kw["Total"] = kw.sum(axis=1)
    if "AI-Related" in kw.columns:
        kw["AI %"] = (kw["AI-Related"] / kw["Total"] * 100).round(1)
    kw = kw.sort_values("AI %", ascending=False) if "AI %" in kw.columns else kw
    indent(kw.to_string())

    subsection("Key observations")
    for kw_name, row in kw.iterrows():
        ai_cnt = row.get("AI-Related", 0)
        total = row["Total"]
        ai_pct = row.get("AI %", 0)
        indent(f"  {kw_name:<30} AI={ai_cnt:>4} / {total:>4} ({ai_pct:.1f}%)")

    results["keyword_summary"] = kw.to_dict()
    return results


# ---- 10.3  Country analysis -----------------------------------------------

def eda_country(df: pd.DataFrame) -> dict:
    """
    Analysis 3: Country breakdown.
    Salary is shown per-currency only; no cross-currency comparison.
    """
    section("EDA 3 – COUNTRY ANALYSIS")
    results = {}

    subsection("3a. Total postings and AI proportion by country")
    ct = df.groupby("country_name")["is_ai_related"].agg(["sum", "count"])
    ct.columns = ["AI_count", "Total"]
    ct["Non_AI"] = ct["Total"] - ct["AI_count"]
    ct["AI %"] = (ct["AI_count"] / ct["Total"] * 100).round(1)
    ct = ct.sort_values("AI %", ascending=False)
    indent(ct.to_string())
    results["country_ai"] = ct.to_dict()

    subsection("3b. Top categories per country")
    for country in df["country_name"].unique():
        print(f"\n  [{country}]")
        top = (df[df["country_name"] == country]
               .groupby("category").size()
               .sort_values(ascending=False)
               .head(5))
        for cat, cnt in top.items():
            indent(f"    {cat:<40} {cnt:>5}", 2)

    subsection("3c. Salary summary per country per currency (suspicious excluded)")
    s = salary_clean(df)
    for country in s["country_name"].unique():
        sub = s[s["country_name"] == country]
        ccy = sub["currency"].iloc[0]
        ai_sal = sub[sub["is_ai_related"]]["salary_avg"]
        nonai_sal = sub[~sub["is_ai_related"]]["salary_avg"]
        print(f"\n  [{country}]  currency={ccy}")
        indent(f"  AI-Related  : n={len(ai_sal):>4}  "
               f"median={ai_sal.median():>12,.0f}  mean={ai_sal.mean():>12,.0f}")
        indent(f"  Non-AI      : n={len(nonai_sal):>4}  "
               f"median={nonai_sal.median():>12,.0f}  mean={nonai_sal.mean():>12,.0f}")
        indent(f"  NOTE: These values are in {ccy} and cannot be directly "
               f"compared with other countries.")

    results["country_salary"] = {
        country: {
            "currency": s[s["country_name"] == country]["currency"].iloc[0],
            "ai_median": float(
                s[(s["country_name"] == country) & s["is_ai_related"]]["salary_avg"].median()
            ) if len(s[(s["country_name"] == country) & s["is_ai_related"]]) > 0 else None
        }
        for country in s["country_name"].unique()
    }
    return results


# ---- 10.4  Category analysis ----------------------------------------------

def eda_category(df: pd.DataFrame) -> dict:
    """
    Analysis 4: Categories with highest AI counts and strongest AI proportions.
    Proportion analysis restricted to categories with >= MIN_CATEGORY_SIZE rows.
    """
    section("EDA 4 – JOB CATEGORY ANALYSIS")
    results = {}

    cat = df.groupby(["category", "job_type"]).size().unstack(fill_value=0)
    cat["Total"] = cat.sum(axis=1)
    if "AI-Related" in cat.columns:
        cat["AI %"] = (cat["AI-Related"] / cat["Total"] * 100).round(1)

    subsection(f"4a. Top 10 categories by AI-Related job COUNT")
    top_count = cat.sort_values("AI-Related", ascending=False).head(10)
    indent(top_count[["AI-Related", "Non-AI", "Total", "AI %"]].to_string())
    results["top_by_count"] = top_count.to_dict()

    subsection(f"4b. Top categories by AI proportion (min sample={MIN_CATEGORY_SIZE})")
    eligible = cat[cat["Total"] >= MIN_CATEGORY_SIZE]
    top_pct = eligible.sort_values("AI %", ascending=False).head(10)
    indent(top_pct[["AI-Related", "Non-AI", "Total", "AI %"]].to_string())
    results["top_by_proportion"] = top_pct.to_dict()

    return results


# ---- 10.5  Time trend analysis --------------------------------------------

def eda_time(df: pd.DataFrame) -> dict:
    """
    Analysis 5: Monthly and yearly posting volume trends.
    Excludes rows with unparseable dates (none expected).
    """
    section("EDA 5 – TIME TREND ANALYSIS")
    results = {}

    df_t = df.copy()
    df_t["created_date_dt"] = pd.to_datetime(df_t["created_date"], errors="coerce")
    df_t = df_t.dropna(subset=["created_date_dt"])

    # ---- Yearly ----
    subsection("5a. Job postings per year (all)")
    yr = df_t.groupby(["year", "job_type"]).size().unstack(fill_value=0)
    yr["Total"] = yr.sum(axis=1)
    if "AI-Related" in yr.columns:
        yr["AI %"] = (yr["AI-Related"] / yr["Total"] * 100).round(1)
    indent(yr.to_string())
    results["by_year"] = yr.to_dict()

    # ---- Monthly (year_month) ----
    subsection("5b. Monthly volume (year_month) – all postings")
    mo = (df_t.groupby(["year_month", "job_type"])
          .size()
          .unstack(fill_value=0)
          .sort_index())
    mo["Total"] = mo.sum(axis=1)
    if "AI-Related" in mo.columns:
        mo["AI %"] = (mo["AI-Related"] / mo["Total"] * 100).round(1)
    indent(mo.to_string())
    results["by_month"] = mo.to_dict()

    # Note about dataset period
    earliest = df_t["created_date_dt"].min().date()
    latest   = df_t["created_date_dt"].max().date()
    indent(f"\n  Dataset spans: {earliest} to {latest}")

    return results


# ---- 10.6  Salary analysis ------------------------------------------------

def eda_salary(df: pd.DataFrame) -> dict:
    """
    Analysis 6: Per-currency salary analysis.
      - Suspicious records excluded (salary_suspicious=False)
      - Predicted vs Stated treated separately
      - No cross-currency comparisons
      - Normalised USD column created for cross-country chart only,
        clearly labelled as an analytical estimate
    """
    section("EDA 6 – SALARY ANALYSIS")
    results = {}

    s = salary_clean(df)
    indent(f"Rows used for salary analysis: {len(s):,}  "
           f"(excluded {df['salary_suspicious'].sum()} suspicious + "
           f"{df['salary_avg'].isna().sum()} nulls)")

    # ---- Per-currency summary ----
    subsection("6a. Salary summary per currency (suspicious excluded)")
    for ccy in ["GBP", "USD", "INR"]:
        sub = s[s["currency"] == ccy]
        if sub.empty:
            continue
        ai = sub[sub["is_ai_related"]]["salary_avg"]
        nonai = sub[~sub["is_ai_related"]]["salary_avg"]
        print(f"\n  [{ccy}]  n={len(sub)}")
        indent(f"  ALL      : median={sub['salary_avg'].median():>12,.0f}  "
               f"mean={sub['salary_avg'].mean():>12,.0f}  "
               f"min={sub['salary_avg'].min():>10,.0f}  "
               f"max={sub['salary_avg'].max():>12,.0f}")
        if len(ai):
            indent(f"  AI-Rel.  : n={len(ai):>4}  median={ai.median():>12,.0f}  "
                   f"mean={ai.mean():>12,.0f}")
        if len(nonai):
            indent(f"  Non-AI   : n={len(nonai):>4}  median={nonai.median():>12,.0f}  "
                   f"mean={nonai.mean():>12,.0f}")
        results[f"salary_{ccy}"] = {
            "n": len(sub),
            "ai_median": float(ai.median()) if len(ai) else None,
            "nonai_median": float(nonai.median()) if len(nonai) else None,
        }

    # ---- Predicted vs Stated split ----
    subsection("6b. Predicted vs Stated salary split (per currency)")
    for ccy in ["GBP", "USD", "INR"]:
        sub = s[s["currency"] == ccy]
        if sub.empty:
            continue
        stated = sub[~sub["salary_is_predicted"]]["salary_avg"]
        predicted = sub[sub["salary_is_predicted"]]["salary_avg"]
        print(f"\n  [{ccy}]")
        if len(stated):
            indent(f"  Stated    : n={len(stated):>4}  median={stated.median():>12,.0f}  "
                   f"mean={stated.mean():>12,.0f}")
        if len(predicted):
            indent(f"  Predicted : n={len(predicted):>4}  median={predicted.median():>12,.0f}  "
                   f"mean={predicted.mean():>12,.0f}")

    # ---- Normalised cross-country comparison ----
    subsection("6c. Cross-country salary (approx. USD equivalent – ANALYTICAL ESTIMATE ONLY)")
    indent("NOTE: Conversion uses fixed approximate mid-2024 FX rates.")
    indent("      GBP x 1.27, INR x 0.012. Do NOT treat as precise exchange values.")
    sn = salary_normalised(df)
    for country in sn["country_name"].unique():
        sub = sn[sn["country_name"] == country]
        ccy = sub["currency"].iloc[0]
        rate = FX_TO_USD.get(ccy, 1.0)
        ai  = sub[sub["is_ai_related"]]["salary_usd_approx"]
        nai = sub[~sub["is_ai_related"]]["salary_usd_approx"]
        print(f"\n  [{country}]  ({ccy} × {rate} = ~USD)")
        if len(ai):
            indent(f"  AI-Rel.  : n={len(ai):>4}  "
                   f"median~${ai.median():>9,.0f}  mean~${ai.mean():>9,.0f}")
        if len(nai):
            indent(f"  Non-AI   : n={len(nai):>4}  "
                   f"median~${nai.median():>9,.0f}  mean~${nai.mean():>9,.0f}")

    results["normalised"] = {
        country: {
            "currency": sn[sn["country_name"] == country]["currency"].iloc[0],
            "ai_median_usd": float(
                sn[(sn["country_name"] == country) & sn["is_ai_related"]]
                ["salary_usd_approx"].median()
            ) if len(sn[(sn["country_name"] == country) & sn["is_ai_related"]]) else None
        }
        for country in sn["country_name"].unique()
    }
    return results


# ---- 10.7  Contract analysis ----------------------------------------------

def eda_contract(df: pd.DataFrame) -> dict:
    """
    Analysis 7: Contract type and time breakdown.
    Results are DESCRIPTIVE ONLY due to high Unknown rate.
    """
    section("EDA 7 – CONTRACT ANALYSIS  [DESCRIPTIVE ONLY – high Unknown rate]")
    results = {}

    for field in ["contract_type", "contract_time"]:
        subsection(f"{field} distribution")
        vc = df[field].value_counts()
        total = len(df)
        for val, cnt in vc.items():
            indent(f"  {val:<20} {cnt:>5}  ({cnt/total*100:.1f}%)")
        unknown_pct = (vc.get("Unknown", 0) / total * 100)
        indent(f"  [CAUTION: {unknown_pct:.1f}% Unknown – treat distributions with caution]")
        results[field] = vc.to_dict()

    subsection("Contract type x job_type (known values only)")
    known = df[df["contract_type"] != "Unknown"]
    ct = known.groupby(["contract_type", "job_type"]).size().unstack(fill_value=0)
    ct["Total"] = ct.sum(axis=1)
    indent(ct.to_string())
    indent("[filter: contract_type != 'Unknown']")
    results["contract_x_jobtype"] = ct.to_dict()

    return results


# ---- 10.8  Repost analysis ------------------------------------------------

def eda_reposts(df: pd.DataFrame) -> dict:
    """
    Analysis 8: Show total vs unique (non-repost) postings side by side.
    """
    section("EDA 8 – REPOST ANALYSIS")
    results = {}

    total = len(df)
    reposts = df["is_repost"].sum()
    unique = total - reposts

    subsection("Total postings vs unique (non-reposted) postings")
    indent(f"Total postings                        : {total:,}")
    indent(f"Re-posted rows (is_repost=True)        : {reposts:,}")
    indent(f"Unique postings (is_repost=False)       : {unique:,}")
    indent(f"Re-post rate                           : {reposts/total*100:.1f}%")
    indent("")
    indent("EXPLANATION: A 're-post' occurs when the same job title appears")
    indent("  from the same company on multiple dates. These are legitimate")
    indent("  temporal re-listings (still open / re-advertised). They are")
    indent("  preserved in the dataset but flagged so analysts can choose")
    indent("  whether to count all listings or only distinct roles.")

    results["total"] = total
    results["reposts"] = int(reposts)
    results["unique"] = int(unique)

    subsection("job_type split – ALL postings vs UNIQUE postings")
    all_jt = df["job_type"].value_counts()
    uniq_jt = df[~df["is_repost"]]["job_type"].value_counts()
    compare = pd.DataFrame({"All": all_jt, "Unique": uniq_jt}).fillna(0).astype(int)
    indent(compare.to_string())

    return results


# ---------------------------------------------------------------------------
# 11.  CHART GENERATION
# ---------------------------------------------------------------------------

def generate_charts(df: pd.DataFrame) -> list:
    """
    Generate and save all EDA charts to CHARTS_DIR.
    Returns list of saved file paths.

    Charts produced:
      01_ai_vs_nonai_overall.png
      02_ai_vs_nonai_by_country.png
      03_ai_vs_nonai_by_category.png
      04_keyword_comparison.png
      05_time_trend_monthly.png
      06_salary_gbp_boxplot.png
      07_salary_usd_boxplot.png
      08_salary_inr_boxplot.png
      09_salary_predicted_vs_stated.png
      10_salary_normalised_cross_country.png
      11_contract_type.png
      12_contract_time.png
      13_reposts_vs_unique.png
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend; works headless
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    section("EDA 9 – CHART GENERATION")
    ensure_charts_dir()
    saved = []

    PALETTE = {
        "AI-Related": "#2563eb",   # blue
        "Non-AI":     "#d97706",   # amber
        "Predicted":  "#7c3aed",   # purple
        "Stated":     "#059669",   # green
    }

    def bar_colors(labels, palette=PALETTE):
        return [palette.get(lbl, "#6b7280") for lbl in labels]

    # ---- Chart 01: AI vs Non-AI overall (pie + bar) ----------------------
    subsection("Chart 01 – AI vs Non-AI overall")
    vc = df["job_type"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("AI-Related vs Non-AI Job Postings", fontsize=14, fontweight="bold")

    axes[0].pie(
        vc.values, labels=vc.index,
        colors=bar_colors(vc.index),
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 12}
    )
    axes[0].set_title("All postings")

    axes[1].bar(vc.index, vc.values, color=bar_colors(vc.index), edgecolor="white")
    for i, (lbl, val) in enumerate(zip(vc.index, vc.values)):
        axes[1].text(i, val + 8, str(val), ha="center", fontsize=11)
    axes[1].set_ylabel("Number of postings")
    axes[1].set_title("All postings – count")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    saved.append(save_fig(fig, "01_ai_vs_nonai_overall.png"))

    # ---- Chart 02: AI vs Non-AI by country (grouped bar) -----------------
    subsection("Chart 02 – AI vs Non-AI by country")
    ct = df.groupby(["country_name", "job_type"]).size().unstack(fill_value=0)
    x = range(len(ct))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    cols = [c for c in ["AI-Related", "Non-AI"] if c in ct.columns]
    for i, col in enumerate(cols):
        bars = ax.bar(
            [xi + i * w for xi in x], ct[col],
            width=w, label=col, color=PALETTE.get(col, "#6b7280"), edgecolor="white"
        )
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(bar.get_height()), ha="center", va="bottom", fontsize=9)
    ax.set_xticks([xi + w / 2 for xi in x])
    ax.set_xticklabels(ct.index)
    ax.set_ylabel("Number of postings")
    ax.set_title("AI-Related vs Non-AI by Country", fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    saved.append(save_fig(fig, "02_ai_vs_nonai_by_country.png"))

    # ---- Chart 03: Top categories by AI count (horizontal bar) ----------
    subsection("Chart 03 – AI jobs by category (top 15)")
    cat = df.groupby(["category", "job_type"]).size().unstack(fill_value=0)
    cat["Total"] = cat.sum(axis=1)
    if "AI-Related" in cat.columns:
        top15 = cat.sort_values("AI-Related", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(11, 7))
        labels = top15.index.tolist()
        ai_vals   = top15.get("AI-Related", pd.Series(0, index=top15.index)).values
        nonai_vals = top15.get("Non-AI", pd.Series(0, index=top15.index)).values
        y = range(len(labels))
        ax.barh([yi + 0.2 for yi in y], ai_vals,   height=0.38,
                label="AI-Related", color=PALETTE["AI-Related"], edgecolor="white")
        ax.barh([yi - 0.2 for yi in y], nonai_vals, height=0.38,
                label="Non-AI",     color=PALETTE["Non-AI"],     edgecolor="white")
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Number of postings")
        ax.set_title("Top 15 Categories by AI-Related Job Count", fontweight="bold")
        ax.legend()
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        fig.tight_layout()
        saved.append(save_fig(fig, "03_ai_vs_nonai_by_category.png"))

    # ---- Chart 04: Keyword comparison ------------------------------------
    subsection("Chart 04 – Keyword comparison")
    kw = df.groupby(["search_keyword", "job_type"]).size().unstack(fill_value=0)
    kw["Total"] = kw.sum(axis=1)
    if "AI-Related" in kw.columns:
        kw["AI %"] = (kw["AI-Related"] / kw["Total"] * 100).round(1)
        kw = kw.sort_values("AI %", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Search Keyword Analysis", fontsize=14, fontweight="bold")

    # Stacked bar – counts
    cols2 = [c for c in ["AI-Related", "Non-AI"] if c in kw.columns]
    bottom = np.zeros(len(kw))
    for col in cols2:
        axes[0].bar(kw.index, kw[col], bottom=bottom,
                    label=col, color=PALETTE.get(col, "#6b7280"), edgecolor="white")
        bottom += kw[col].values
    axes[0].set_ylabel("Number of postings")
    axes[0].set_title("Count by keyword (stacked)")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)

    # AI % bar
    axes[1].bar(kw.index, kw["AI %"],
                color=[PALETTE["AI-Related"] if p >= 50 else PALETTE["Non-AI"]
                       for p in kw["AI %"]],
                edgecolor="white")
    for i, (idx, row) in enumerate(kw.iterrows()):
        axes[1].text(i, row["AI %"] + 0.5, f"{row['AI %']:.0f}%",
                     ha="center", fontsize=10)
    axes[1].set_ylabel("AI-Related %")
    axes[1].set_title("AI proportion by keyword")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylim(0, 110)
    fig.tight_layout()
    saved.append(save_fig(fig, "04_keyword_comparison.png"))

    # ---- Chart 05: Monthly time trend ------------------------------------
    subsection("Chart 05 – Monthly time trend")
    df_t = df.copy()
    df_t["created_date_dt"] = pd.to_datetime(df_t["created_date"], errors="coerce")
    mo = (df_t.groupby(["year_month", "job_type"])
          .size()
          .unstack(fill_value=0)
          .sort_index())

    fig, ax = plt.subplots(figsize=(14, 5))
    for col in [c for c in ["AI-Related", "Non-AI"] if c in mo.columns]:
        ax.plot(mo.index, mo[col], label=col,
                color=PALETTE.get(col, "#6b7280"),
                linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Number of postings")
    ax.set_title("Monthly Job Posting Volume: AI-Related vs Non-AI", fontweight="bold")
    ax.legend()
    # Show every 3rd label to avoid crowding
    ticks = list(mo.index)
    ax.set_xticks(ticks[::3])
    ax.set_xticklabels(ticks[::3], rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    saved.append(save_fig(fig, "05_time_trend_monthly.png"))

    # ---- Charts 06-08: Salary boxplots per currency ----------------------
    s = salary_clean(df)
    ccy_info = {"GBP": ("06", "£"), "USD": ("07", "$"), "INR": ("08", "₹")}
    for ccy, (num, sym) in ccy_info.items():
        sub = s[s["currency"] == ccy]
        if sub.empty:
            continue
        subsection(f"Chart {num} – Salary distribution [{ccy}]")
        data_ai   = sub[sub["is_ai_related"]]["salary_avg"].dropna()
        data_nai  = sub[~sub["is_ai_related"]]["salary_avg"].dropna()
        plot_data = [data_ai.values, data_nai.values]
        plot_labels = [f"AI-Related\n(n={len(data_ai)})", f"Non-AI\n(n={len(data_nai)})"]

        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(
            plot_data, tick_labels=plot_labels,
            patch_artist=True, notch=False,
            medianprops={"color": "white", "linewidth": 2}
        )
        colors = [PALETTE["AI-Related"], PALETTE["Non-AI"]]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel(f"Annual salary ({ccy})")
        ax.set_title(
            f"Salary Distribution by Job Type – {ccy}\n"
            f"(suspicious records excluded; salary_suspicious=False)",
            fontweight="bold"
        )
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{sym}{int(x):,}")
        )
        fig.tight_layout()
        saved.append(save_fig(fig, f"{num}_salary_{ccy.lower()}_boxplot.png"))

    # ---- Chart 09: Predicted vs Stated salary per currency ---------------
    subsection("Chart 09 – Predicted vs Stated salary")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Salary Distribution: Predicted vs Stated (by currency)",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02,
             "NOTE: Predicted = algorithmic estimate; Stated = employer-published figure",
             ha="center", fontsize=9, style="italic", color="#555")

    ccy_syms = {"GBP": "£", "USD": "$", "INR": "₹"}
    for ax, ccy in zip(axes, ["GBP", "USD", "INR"]):
        sub = s[s["currency"] == ccy]
        stated    = sub[~sub["salary_is_predicted"]]["salary_avg"].dropna()
        predicted = sub[sub["salary_is_predicted"]]["salary_avg"].dropna()
        if stated.empty and predicted.empty:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(
            [stated.values if len(stated) else [0],
             predicted.values if len(predicted) else [0]],
            tick_labels=[f"Stated\n(n={len(stated)})", f"Predicted\n(n={len(predicted)})"],
            patch_artist=True,
            medianprops={"color": "white", "linewidth": 2}
        )
        for patch, color in zip(bp["boxes"], [PALETTE["Stated"], PALETTE["Predicted"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(ccy)
        sym = ccy_syms.get(ccy, "")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _, s=sym: f"{s}{int(x):,}")
        )
    fig.tight_layout()
    saved.append(save_fig(fig, "09_salary_predicted_vs_stated.png"))

    # ---- Chart 10: Normalised cross-country salary (approx. USD) ---------
    subsection("Chart 10 – Salary normalised to approx. USD (cross-country)")
    sn = salary_normalised(df)
    countries = sn["country_name"].unique().tolist()
    ai_medians  = []
    nai_medians = []
    for country in countries:
        sub = sn[sn["country_name"] == country]
        ai_medians.append(
            sub[sub["is_ai_related"]]["salary_usd_approx"].median()
            if len(sub[sub["is_ai_related"]]) else 0
        )
        nai_medians.append(
            sub[~sub["is_ai_related"]]["salary_usd_approx"].median()
            if len(sub[~sub["is_ai_related"]]) else 0
        )

    x = range(len(countries))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([xi for xi in x],       ai_medians,  width=w, label="AI-Related",
           color=PALETTE["AI-Related"], edgecolor="white")
    ax.bar([xi + w for xi in x],   nai_medians, width=w, label="Non-AI",
           color=PALETTE["Non-AI"],     edgecolor="white")
    ax.set_xticks([xi + w / 2 for xi in x])
    ax.set_xticklabels(countries)
    ax.set_ylabel("Median salary (approx. USD)")
    ax.set_title(
        "Median Salary by Country and Job Type\n"
        "(APPROX. USD equivalent – analytical normalisation only;\n"
        "GBP×1.27, INR×0.012; mid-2024 fixed rates)",
        fontweight="bold", fontsize=10
    )
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
    fig.tight_layout()
    saved.append(save_fig(fig, "10_salary_normalised_cross_country.png"))

    # ---- Chart 11: Contract type -----------------------------------------
    subsection("Chart 11 – Contract type")
    vc_ct = df["contract_type"].value_counts()
    total = len(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(vc_ct.index, vc_ct.values,
                  color=["#94a3b8" if v == "Unknown" else "#3b82f6"
                         for v in vc_ct.index],
                  edgecolor="white")
    for bar in bars:
        pct = bar.get_height() / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{bar.get_height()}\n({pct:.1f}%)", ha="center", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title(
        "Contract Type Distribution\n(DESCRIPTIVE ONLY – 74.5% Unknown)",
        fontweight="bold"
    )
    fig.tight_layout()
    saved.append(save_fig(fig, "11_contract_type.png"))

    # ---- Chart 12: Contract time -----------------------------------------
    subsection("Chart 12 – Contract time")
    vc_ctm = df["contract_time"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(vc_ctm.index, vc_ctm.values,
                  color=["#94a3b8" if v == "Unknown" else "#f59e0b"
                         for v in vc_ctm.index],
                  edgecolor="white")
    for bar in bars:
        pct = bar.get_height() / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{bar.get_height()}\n({pct:.1f}%)", ha="center", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title(
        "Contract Time Distribution\n(DESCRIPTIVE ONLY – 51.3% Unknown)",
        fontweight="bold"
    )
    fig.tight_layout()
    saved.append(save_fig(fig, "12_contract_time.png"))

    # ---- Chart 13: Reposts vs unique -------------------------------------
    subsection("Chart 13 – Total vs Unique postings")
    all_jt  = df["job_type"].value_counts()
    uniq_jt = df[~df["is_repost"]]["job_type"].value_counts()
    cats = [c for c in ["AI-Related", "Non-AI"] if c in all_jt.index]
    x = range(len(cats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([xi for xi in x],
           [all_jt.get(c, 0) for c in cats],
           width=w, label="All postings",
           color=["#1d4ed8", "#b45309"], edgecolor="white")
    ax.bar([xi + w for xi in x],
           [uniq_jt.get(c, 0) for c in cats],
           width=w, label="Unique roles (non-repost)",
           color=["#93c5fd", "#fcd34d"], edgecolor="white")
    ax.set_xticks([xi + w / 2 for xi in x])
    ax.set_xticklabels(cats)
    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title(
        "Total Postings vs Unique Roles\n"
        "(Unique = is_repost=False; reposts are legitimate re-listings)",
        fontweight="bold"
    )
    ax.legend()
    fig.tight_layout()
    saved.append(save_fig(fig, "13_reposts_vs_unique.png"))

    section(f"CHART GENERATION COMPLETE – {len(saved)} charts saved to '{CHARTS_DIR}/'")
    for p in saved:
        indent(p)

    return saved


# ---------------------------------------------------------------------------
# 12.  EDA KEY FINDINGS SUMMARY
# ---------------------------------------------------------------------------

def print_eda_findings(df: pd.DataFrame) -> None:
    """
    Print the consolidated EDA findings, limitations, and next steps.
    This section is intended to feed directly into the project report.
    """
    section("EDA 10 – KEY FINDINGS & LIMITATIONS")

    s = salary_clean(df)
    total = len(df)
    ai_total   = df["is_ai_related"].sum()
    nonai_total = total - ai_total

    unique_df  = df[~df["is_repost"]]
    ai_unique  = unique_df["is_ai_related"].sum()
    nonai_unique = len(unique_df) - ai_unique

    subsection("KEY NUMERICAL FINDINGS")

    findings = [
        # --- Overall
        f"[1]  Total job postings in dataset          : {total:,}",
        f"[2]  AI-Related postings (all)              : {ai_total:,} ({ai_total/total*100:.1f}%)",
        f"[3]  Non-AI postings (all)                  : {nonai_total:,} ({nonai_total/total*100:.1f}%)",
        f"[4]  Unique (non-repost) postings           : {len(unique_df):,}",
        f"[5]  AI-Related unique roles                : {ai_unique:,} ({ai_unique/len(unique_df)*100:.1f}%)",
        f"[6]  Non-AI unique roles                    : {nonai_unique:,} ({nonai_unique/len(unique_df)*100:.1f}%)",
        # --- Keywords
        f"[7]  'machine learning' keyword: 100% AI-related (298/298)",
        f"[8]  'artificial intelligence' keyword: 85.3% AI-related (384/450)",
        f"[9]  'automation' keyword: 94.5% AI-related (329/348)",
        f"[10] 'data entry' keyword: 3% AI-related (9/300)",
        f"[11] 'customer service' keyword: 0% AI-related (0/300)",
        # --- Country
        f"[12] India (in): highest AI proportion – "
             f"{df[df['country_name']=='India']['is_ai_related'].mean()*100:.1f}%",
        f"[13] UK (gb): "
             f"{df[df['country_name']=='United Kingdom']['is_ai_related'].mean()*100:.1f}% AI-related",
        f"[14] US (us): "
             f"{df[df['country_name']=='United States']['is_ai_related'].mean()*100:.1f}% AI-related",
        # --- Category
        "[15] IT Jobs is the largest category (695) with the most AI postings",
        "[16] 'Engineering Jobs' and 'Scientific & QA Jobs' have high AI proportions",
        # --- Salary (per-currency, no cross comparison)
    ]

    for ccy, sym in [("GBP", "£"), ("USD", "$"), ("INR", "₹")]:
        sub = s[s["currency"] == ccy]
        ai_s  = sub[sub["is_ai_related"]]["salary_avg"]
        nai_s = sub[~sub["is_ai_related"]]["salary_avg"]
        if len(ai_s) and len(nai_s):
            findings.append(
                f"[SAL-{ccy}] AI median={sym}{ai_s.median():,.0f} vs "
                f"Non-AI median={sym}{nai_s.median():,.0f}  "
                f"(n_ai={len(ai_s)}, n_nonai={len(nai_s)}, currency={ccy} only)"
            )

    findings += [
        f"[17] Salary records used in analysis: {len(s):,} "
             f"(excluded {df['salary_suspicious'].sum()} suspicious + "
             f"{df['salary_avg'].isna().sum()} nulls)",
        f"[18] {df['salary_is_predicted'].sum():,} of {len(s):,} salary rows "
              "are PREDICTED (algorithmic), not employer-stated",
        f"[19] Re-post rate: {df['is_repost'].sum():,} / {total:,} "
              f"= {df['is_repost'].sum()/total*100:.1f}%",
    ]

    for f_line in findings:
        indent(f_line)

    subsection("IMPORTANT LIMITATIONS")
    limitations = [
        "L1. Salaries in GBP, USD, INR are not directly comparable. "
           "The 'salary_usd_approx' normalisation uses fixed mid-2024 FX rates "
           "and is labelled as an analytical estimate only.",
        "L2. 74.5% of contract_type and 51.3% of contract_time values are "
           "'Unknown'. Contract analysis is descriptive only.",
        "L3. 68.5% of salary_avg values are algorithmically predicted "
           "(salary_is_predicted=True), not employer-stated figures.",
        "L4. The dataset was collected using 5 specific search keywords. "
           "It does not represent all jobs in the market.",
        "L5. No causal claims can be made. Observed correlations between "
           "AI-related postings and higher salaries do not imply causation.",
        "L6. Job descriptions are truncated (~500 chars). Skills analysis "
           "from descriptions is not feasible with this dataset.",
        "L7. The dataset includes listings from May 2023 to August 2026, "
           "but coverage is uneven across the period.",
        "L8. 4 suspicious salary records (£13, £20.50, $32, £500) "
           "are excluded from salary analysis but retained in the dataset.",
    ]
    for lim in limitations:
        print()
        for line in textwrap.wrap(lim, width=72, subsequent_indent="      "):
            indent(line, 4)

    subsection("CHARTS CREATED (saved to charts/)")
    chart_list = [
        "01_ai_vs_nonai_overall.png          – Pie + bar: overall AI vs Non-AI",
        "02_ai_vs_nonai_by_country.png       – Grouped bar by country",
        "03_ai_vs_nonai_by_category.png      – Horizontal bar top 15 categories",
        "04_keyword_comparison.png           – Stacked bar + AI% by keyword",
        "05_time_trend_monthly.png           – Monthly line chart",
        "06_salary_gbp_boxplot.png           – GBP salary boxplot AI vs Non-AI",
        "07_salary_usd_boxplot.png           – USD salary boxplot AI vs Non-AI",
        "08_salary_inr_boxplot.png           – INR salary boxplot AI vs Non-AI",
        "09_salary_predicted_vs_stated.png   – Predicted vs Stated per currency",
        "10_salary_normalised_cross_country.png – Approx. USD median by country",
        "11_contract_type.png               – Contract type distribution",
        "12_contract_time.png               – Contract time distribution",
        "13_reposts_vs_unique.png           – Total vs unique postings",
    ]
    for c in chart_list:
        indent(c)

    subsection("ADDITIONAL COLUMNS CREATED DURING EDA")
    new_eda_cols = [
        "salary_usd_approx   – Salary normalised to approx. USD "
                              "(analytical estimate; not in main df)",
    ]
    for c in new_eda_cols:
        indent(c)

    subsection("RECOMMENDED DASHBOARD COMPONENTS")
    dashboard_items = [
        "1. KPI row: total postings / AI count / Non-AI count / AI%",
        "2. Pie/donut: AI vs Non-AI proportion (Chart 01)",
        "3. Grouped bar: AI vs Non-AI by country (Chart 02)",
        "4. Horizontal bar: Top categories by AI count (Chart 03)",
        "5. Stacked bar: Keyword comparison (Chart 04)",
        "6. Line chart: Monthly posting trend (Chart 05)",
        "7. Side-by-side boxplots: Salary by currency, AI vs Non-AI (Charts 06-08)",
        "8. Boxplot: Predicted vs Stated salary (Chart 09)",
        "9. Bar: Approx. normalised salary cross-country (Chart 10)",
        "10. Bar: Contract type / time (Charts 11-12) – labelled 'descriptive only'",
        "11. Bar: Total vs Unique postings (Chart 13)",
        "Optional filters: country, keyword, currency, job_type, year",
    ]
    for item in dashboard_items:
        indent(item)

    subsection("WHAT STILL NEEDS TO BE COMPLETED BEFORE FINAL DOCX REPORT")
    remaining = [
        "[ ] Build interactive dashboard (Plotly/Dash) using charts above",
        "[ ] Capture dashboard screenshots for report",
        "[ ] Write report narrative based on EDA findings",
        "[ ] Create Jafar_AI_Job_Market_ProjectReport.docx",
        "[ ] Final review of all findings for accuracy and limitations",
    ]
    for r in remaining:
        indent(r)

    print()


# ---------------------------------------------------------------------------
# 13.  MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    """
    Execute the full pipeline:
      1-7: Preprocessing (load, inspect, deduplicate, clean, engineer, flag)
      8-10: EDA (analysis functions + chart generation + findings summary)
    Returns the cleaned, analysis-ready DataFrame.
    The original CSV is never modified.
    """
    print("\n" + "=" * 70)
    print("  AI JOB MARKET ANALYTICS – FULL PIPELINE")
    print("  IBM SkillBuild Project  |  Author: Jafar")
    print("=" * 70)

    # ── PREPROCESSING ──────────────────────────────────────────────────────
    df_raw = load_raw_data(RAW_DATA_FILE)
    inspect_raw(df_raw)
    df = df_raw.copy()
    df = detect_duplicates(df)
    df = clean_and_standardise(df)
    df = engineer_features(df)
    df = flag_suspicious_salaries(df)
    final_summary(df_raw, df)

    # ── EDA ────────────────────────────────────────────────────────────────
    eda_ai_vs_nonai(df)
    eda_keywords(df)
    eda_country(df)
    eda_category(df)
    eda_time(df)
    eda_salary(df)
    eda_contract(df)
    eda_reposts(df)
    generate_charts(df)
    print_eda_findings(df)

    return df


if __name__ == "__main__":
    df_clean = main()
    # df_clean is the fully preprocessed and analysed DataFrame.
    # The next step is the interactive dashboard (Plotly/Dash).
