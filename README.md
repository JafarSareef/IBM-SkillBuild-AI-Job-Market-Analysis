# AI Job Market Analytics

**IBM SkillBuild Project**<br>
**Author:** Jafar<br>
**Status:** ✅ Completed

---
## 🚀 Live Dashboard

👉 [Open Streamlit Dashboard](https://ibm-skillbuild-ai-job-market-analysis.streamlit.app/)
## Project Overview

This project analyses a real-world job listings dataset to explore the landscape of AI-related versus traditional job postings across the United Kingdom, United States, and India. The goal is to surface trends in job demand, salary distributions, and role categories through descriptive analysis and interactive dashboards.

> **Note:** This project is descriptive and exploratory. No causal claims are made about AI and employment outcomes.

---

## Dataset

| Property   | Detail                                  |
| ---------- | --------------------------------------- |
| File       | `ai_job_market_dataset.csv`             |
| Rows       | 1,696 job postings                      |
| Columns    | 19 (raw)                                |
| Countries  | United Kingdom, United States, India    |
| Date range | May 2023 – July 2026                    |
| Currencies | GBP, USD, INR (not directly comparable) |

### Dataset Source

[Kaggle – AI Job Market 2026: Automation vs Traditional Roles](https://www.kaggle.com/datasets/mariaaqdas/ai-job-market-2026-automation-vs-traditional-role)

### Search Keywords Used in Data Collection

| Keyword                 | Category    |
| ----------------------- | ----------- |
| artificial intelligence | AI-type     |
| machine learning        | AI-type     |
| automation              | AI-type     |
| data entry              | Traditional |
| customer service        | Traditional |

---

## Project Structure

```text
IBM-SkillBuild-AI-Job-Market-Analysis/
├── ai_job_market_dataset.csv
├── Jafar_AI_Job_Market_Analytics.py
├── Jafar_AI_Job_Market_Dashboard.py
├── Jafar_AI_Job_Market_Streamlit.py
├── charts/
│   ├── 01_ai_vs_nonai_overall.png
│   ├── 02_ai_vs_nonai_by_country.png
│   ├── 03_ai_vs_nonai_by_category.png
│   ├── 04_keyword_comparison.png
│   ├── 05_time_trend_monthly.png
│   ├── 06_salary_gbp_boxplot.png
│   ├── 07_salary_usd_boxplot.png
│   ├── 08_salary_inr_boxplot.png
│   ├── 09_salary_predicted_vs_stated.png
│   ├── 10_salary_normalised_cross_country.png
│   ├── 11_contract_type.png
│   ├── 12_contract_time.png
│   └── 13_reposts_vs_unique.png
├── requirements.txt
├── README.md
└── Jafar_AI_Job_Market_ProjectReport.docx
```

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
```

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
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

The script performs data preprocessing and exploratory data analysis and saves 13 charts to the `charts/` folder. The original CSV is **never modified**.

### 4. Run the interactive Dash dashboard

```bash
python Jafar_AI_Job_Market_Dashboard.py
```

Then open:

**[http://127.0.0.1:8050](http://127.0.0.1:8050)**

Press `Ctrl+C` to stop the server.

### 5. Run the Streamlit dashboard

```bash
streamlit run Jafar_AI_Job_Market_Streamlit.py
```

Then open:

**[http://localhost:8501](http://localhost:8501)**

The Streamlit dashboard provides interactive filters for country, search keyword, job type, year, and currency.

---

## Pipeline Stages

| Stage                   | Description                                    | Status |
| ----------------------- | ---------------------------------------------- | ------ |
| 1 – Load                | Read CSV, drop trailing empty column           | ✅ Done |
| 2 – Inspect             | Column types, missing values, distributions    | ✅ Done |
| 3 – Duplicate detection | Exact duplicates + re-post flagging            | ✅ Done |
| 4 – Clean & standardise | Dates, booleans, floats, Unknown fills         | ✅ Done |
| 5 – Feature engineering | `job_type`, `salary_status`, `year`, `month`   | ✅ Done |
| 6 – Suspicious salaries | Flag anomalously low salary values             | ✅ Done |
| 7 – EDA                 | Charts and statistical summaries               | ✅ Done |
| 8 – Dashboard           | Interactive Plotly/Dash + Streamlit dashboards | ✅ Done |
| 9 – Report              | Final DOCX report                              | ✅ Done |

---

## Engineered Columns

| New Column          | Source                    | Values               |
| ------------------- | ------------------------- | -------------------- |
| `job_type`          | `is_ai_related`           | AI-Related / Non-AI  |
| `salary_status`     | `salary_is_predicted`     | Predicted / Stated   |
| `year`              | `created_date`            | Integer year         |
| `month`             | `created_date`            | Integer month (1–12) |
| `year_month`        | `created_date`            | String "YYYY-MM"     |
| `exact_duplicate`   | All columns               | Boolean flag         |
| `is_repost`         | `title` + `company`       | Boolean flag         |
| `salary_suspicious` | `salary_avg` + `currency` | Boolean flag         |

---

## Key Findings

* **1,696** total job postings were analysed.
* **1,021 (60.2%)** postings were classified as AI-related.
* **675 (39.8%)** postings were classified as non-AI-related.
* **918** postings were identified as unique roles after repost analysis.
* **778** postings were identified as reposts.
* Among unique roles, **633 (69.0%)** were AI-related.
* India had an AI-related posting proportion of approximately **83.5%**.
* The United States had an AI-related posting proportion of approximately **59.9%**.
* The United Kingdom had an AI-related posting proportion of approximately **54.3%**.
* Machine learning postings had a **100% AI-related classification rate** within the dataset.
* Artificial intelligence postings had an **85.3% AI-related classification rate**.
* Automation postings had a **94.5% AI-related classification rate**.
* Data entry postings had a **3.0% AI-related classification rate**.
* Customer service postings had a **0.3% AI-related classification rate**.

---

## Salary Analysis

Salary analysis was performed separately for **GBP, USD, and INR** because the currencies are not directly comparable.

Suspicious salary records were flagged and excluded from salary distribution analysis.

Among salary-valid, non-suspicious records, approximately **76.9%** of salary values were predicted rather than employer-stated.

The project also includes an approximate USD-normalised salary analysis for illustrative comparison. These values are analytical estimates and should not be treated as precise currency conversions.

---

## Known Data Quality Issues

1. **Multi-currency salaries** – GBP, USD, and INR values must NOT be compared directly. Salary analysis is performed separately by currency.

2. **Sparse contract fields** – `contract_type` is 74.5% unknown; `contract_time` is 51.3% unknown. These fields should be interpreted cautiously.

3. **Predicted salaries** – Approximately **76.9%** of salary-valid, non-suspicious records contain algorithmically predicted salary values rather than employer-stated values.

4. **Suspicious salary records** – 4 rows were flagged because their salary values were considered anomalously low under the project thresholds. These records are flagged with `salary_suspicious = True` and excluded from salary distribution analysis.

5. **Temporal re-posts** – **778 rows** were identified as re-posts based on the project repost logic. They are preserved but flagged with `is_repost = True`.

6. **Truncated descriptions** – Free-text descriptions are cut off at approximately 500 characters in the CSV; therefore, full NLP-based skill analysis is limited.

---

## Analysis Completed

* [x] Salary comparison: AI-Related vs Non-AI (within each currency)
* [x] Job category distribution by job_type
* [x] Geographic breakdown (UK / US / India)
* [x] Time-series trend: job posting volume by month
* [x] Contract type and work-hours patterns
* [x] Reposted vs unique job-posting analysis
* [x] Search keyword comparison
* [x] Interactive Plotly/Dash dashboard
* [x] Interactive Streamlit dashboard
* [x] Final DOCX project report

---

## Report

📄 `Jafar_AI_Job_Market_ProjectReport.docx` — **Completed**

The report contains the project methodology, dataset description, preprocessing, exploratory data analysis, salary analysis, dashboard discussion, findings, limitations, conclusion, future scope, references, and appendix.

---

## Important Constraints

* The original `ai_job_market_dataset.csv` is **read-only** throughout.
* No causal claims are made about AI and job displacement.
* Salary values across currencies are **never compared directly** without clearly stated analytical normalization.
* `is_ai_related` and `salary_is_predicted` are preserved from the source.
* Job postings represent observable employer-demand signals and do not necessarily represent confirmed hiring outcomes.
* The dataset does not represent the complete global job market.

---

## Project Status

**✅ Completed — IBM SkillBuild AI Job Market Analytics Project**

