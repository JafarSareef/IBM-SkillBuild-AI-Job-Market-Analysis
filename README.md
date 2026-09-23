# AI Job Market Analytics

**IBM SkillBuild Project**  
**Author:** Jafar  
**Status:** 🔄 In Progress – Preprocessing complete; EDA and dashboard pending

---

## Project Overview

This project analyses a real-world job listings dataset to explore the
landscape of AI-related versus traditional job postings across the United
Kingdom, United States, and India. The goal is to surface trends in job
demand, salary distributions, and role categories through descriptive
analysis and an interactive dashboard.

> **Note:** This project is descriptive and exploratory. No causal claims
> are made about AI and employment outcomes.

---

## Dataset

| Property       | Detail                              |
|----------------|-------------------------------------|
| File           | `ai_job_market_dataset.csv`         |
| Rows           | 1,696 job postings                  |
| Columns        | 19 (raw) → 27 (after preprocessing) |
| Countries      | United Kingdom, United States, India|
| Date range     | May 2023 – July 2026                |
| Currencies     | GBP, USD, INR (not directly comparable) |

### Search Keywords Used in Data Collection

| Keyword              | Category        |
|----------------------|-----------------|
| artificial intelligence | AI-type      |
| machine learning     | AI-type         |
| automation           | AI-type         |
| data entry           | Traditional     |
| customer service     | Traditional     |

---

## Project Structure

```
archive/
├── ai_job_market_dataset.csv              # Original dataset (read-only)
├── Jafar_AI_Job_Market_Analytics.py       # Preprocessing + EDA script
├── Jafar_AI_Job_Market_Dashboard.py       # Interactive Plotly/Dash dashboard
├── charts/                                # 13 EDA charts (PNG)
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
└── Jafar_AI_Job_Market_ProjectReport.docx # ⏳ Pending – not yet created
```

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the preprocessing + EDA pipeline

```bash
python Jafar_AI_Job_Market_Analytics.py
```

The script prints a full preprocessing and EDA report to the terminal.
Saves 13 charts to `charts/`. The original CSV is **never modified**.

### 4. Run the interactive dashboard

```bash
python Jafar_AI_Job_Market_Dashboard.py
```

Then open **http://127.0.0.1:8050** in your browser.
Press `Ctrl+C` to stop the server.

---

## Pipeline Stages

| Stage | Description | Status |
|-------|-------------|--------|
| 1 – Load | Read CSV, drop trailing empty column | ✅ Done |
| 2 – Inspect | Column types, missing values, distributions | ✅ Done |
| 3 – Duplicate detection | Exact dupes + re-post flagging | ✅ Done |
| 4 – Clean & standardise | Dates, booleans, floats, Unknown fills | ✅ Done |
| 5 – Feature engineering | `job_type`, `salary_status`, `year`, `month` | ✅ Done |
| 6 – Suspicious salaries | Flag anomalously low salary values | ✅ Done |
| 7 – EDA | Charts and statistical summaries | ✅ Done |
| 8 – Dashboard | Interactive Plotly/Dash dashboard | ✅ Done |
| 9 – Report | Final DOCX report | ⏳ Pending |

---

## Engineered Columns

| New Column         | Source                  | Values                    |
|--------------------|-------------------------|---------------------------|
| `job_type`         | `is_ai_related`         | AI-Related / Non-AI       |
| `salary_status`    | `salary_is_predicted`   | Predicted / Stated        |
| `year`             | `created_date`          | Integer year              |
| `month`            | `created_date`          | Integer month (1–12)      |
| `year_month`       | `created_date`          | String "YYYY-MM"          |
| `exact_duplicate`  | All columns             | Boolean flag              |
| `is_repost`        | `title` + `company`     | Boolean flag              |
| `salary_suspicious`| `salary_avg` + `currency` | Boolean flag            |

---

## Known Data Quality Issues

1. **Multi-currency salaries** – GBP, USD, and INR values must NOT be
   compared directly. Currency conversion is required for cross-country
   salary analysis.

2. **Sparse contract fields** – `contract_type` is 74.5% unknown;
   `contract_time` is 51.3% unknown. Use with caution.

3. **Predicted salaries** – ~68.5% of salary figures are algorithmic
   estimates (`salary_is_predicted = True`), not employer-stated values.

4. **Suspicious salary records** – 3 rows have salary_avg values below
   the per-currency threshold (likely hourly/daily rates stored as annual).
   These are flagged with `salary_suspicious = True`.

5. **Temporal re-posts** – 783 rows are re-posts of the same role across
   different dates. They are preserved but flagged with `is_repost = True`.

6. **Truncated descriptions** – Free-text descriptions are cut off at
   ~500 characters in the CSV; full NLP analysis is limited.

---

## Analysis Plan *(pending)*

> Results will be added once EDA and dashboard are complete.

- [ ] Salary comparison: AI-Related vs Non-AI (within each currency)
- [ ] Job category distribution by job_type
- [ ] Geographic breakdown (UK / US / India)
- [ ] Time-series trend: job posting volume by month
- [ ] Contract type and work-hours patterns
- [ ] Top hiring companies in AI roles
- [ ] Dashboard with interactive filters

---

## Report

📄 `Jafar_AI_Job_Market_ProjectReport.docx` — **not yet created**  
Will be authored once EDA and dashboard are finalised.

---

## Important Constraints

- The original `ai_job_market_dataset.csv` is **read-only** throughout.
- No causal claims are made about AI and job displacement.
- Salary values across currencies are **never compared directly**.
- `is_ai_related` and `salary_is_predicted` are preserved from the source.
