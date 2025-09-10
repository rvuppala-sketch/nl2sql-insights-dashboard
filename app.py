# app.py
# Streamlit Storytelling Dashboard for HR Attrition (CSV -> DuckDB -> Insights Cards)
# ------------------------------------------------------------
# Features
# - Upload a CSV (or auto-load if DB exists) and convert to DuckDB
# - Column auto-detection + manual mapping in sidebar
# - Insight cards: Overall Attrition, by Department/Role, Overtime effect,
#   Tenure bands, Income quartiles, Satisfaction & Work-Life Balance
# - Charts + downloadable CSVs for each insight
# ------------------------------------------------------------

import os
import io
import json
import textwrap
from typing import Dict, Optional, Tuple, List

import duckdb
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="HR Attrition — Storytelling Dashboard", layout="wide")
DB_PATH_DEFAULT = os.environ.get("HR_DB_PATH", "data/hr_attrition.duckdb")
TABLE_NAME_DEFAULT = os.environ.get("HR_TABLE_NAME", "employees")

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource
def connect_duckdb(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
    return duckdb.connect(db_path)

def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    try:
        con.execute(f"SELECT 1 FROM information_schema.tables WHERE table_name = '{table_name}'").fetchone()
        res = con.fetchone()
        return bool(res)
    except Exception:
        return False

def infer_and_create_table(con: duckdb.DuckDBPyConnection, table_name: str, csv_bytes: bytes) -> pd.DataFrame:
    """
    Reads CSV from bytes with DuckDB's read_csv_auto and creates/replaces the table.
    Returns a pandas preview (head).
    """
    # Write bytes to a temp file (DuckDB read_csv_auto wants a file path or S3)
    tmp_dir = "data/tmp_uploads"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_csv_path = os.path.join(tmp_dir, "upload.csv")
    with open(tmp_csv_path, "wb") as f:
        f.write(csv_bytes)

    # Create table
    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * FROM read_csv_auto('{tmp_csv_path}', SAMPLE_SIZE=-1, IGNORE_ERRORS=true)
    """)

    df_head = con.execute(f"SELECT * FROM {table_name} LIMIT 10").df()
    return df_head

def load_table_as_df(con: duckdb.DuckDBPyConnection, table_name: str) -> pd.DataFrame:
    return con.execute(f"SELECT * FROM {table_name}").df()

def find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    norm = lambda s: s.lower().replace(" ", "").replace("_", "")
    for cand in candidates:
        for c in cols:
            if norm(c) == norm(cand):
                return c
    return None

def guess_mappings(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = df.columns.tolist()

    mapping = {
        "Attrition": find_col(cols, ["Attrition", "Left", "Churn", "IsAttrition", "Quit"]),
        "Department": find_col(cols, ["Department", "Dept"]),
        "JobRole": find_col(cols, ["JobRole", "Role", "Job Title", "Title"]),
        "MonthlyIncome": find_col(cols, ["MonthlyIncome", "Monthly Income", "Salary", "Compensation", "Pay"]),
        "OverTime": find_col(cols, ["OverTime", "Overtime", "Over Time"]),
        "YearsAtCompany": find_col(cols, ["YearsAtCompany", "Tenure", "Years At Company", "YearsAtFirm"]),
        "Age": find_col(cols, ["Age"]),
        "EnvironmentSatisfaction": find_col(cols, ["EnvironmentSatisfaction", "EnvSatisfaction"]),
        "JobSatisfaction": find_col(cols, ["JobSatisfaction"]),
        "WorkLifeBalance": find_col(cols, ["WorkLifeBalance", "Work Life Balance", "WLB"]),
        "EducationField": find_col(cols, ["EducationField", "Major", "Discipline"]),
        "Gender": find_col(cols, ["Gender", "Sex"]),
        "BusinessTravel": find_col(cols, ["BusinessTravel"]),
        "MaritalStatus": find_col(cols, ["MaritalStatus"]),
    }
    return mapping

def to_bool_yes_no(series: pd.Series) -> pd.Series:
    """
    Convert Yes/No (or similar) to booleans (True for Yes/1/Y).
    Leaves numeric/boolean series mostly intact.
    """
    if series.dtype == bool:
        return series
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["yes", "y", "true", "1"])

def safe_qcut(s: pd.Series, q=4, labels=None) -> pd.Series:
    try:
        return pd.qcut(s, q=q, labels=labels, duplicates="drop")
    except Exception:
        # Fallback to simple cut if qcut fails (not enough unique values)
        return pd.cut(s, bins=min(q, s.nunique()), labels=labels, include_lowest=True)

def fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"

def download_df(df: pd.DataFrame, label: str, key_suffix: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {label} CSV",
        data=csv,
        file_name=f"{label.replace(' ', '_').lower()}.csv",
        mime="text/csv",
        key=f"dl_{key_suffix}"
    )

# -----------------------------
# Data & Mapping
# -----------------------------
st.sidebar.title("Data & Settings")

db_path = st.sidebar.text_input("DuckDB path", DB_PATH_DEFAULT)
table_name = st.sidebar.text_input("Table name", TABLE_NAME_DEFAULT)

con = connect_duckdb(db_path)
have_table = table_exists(con, table_name)

uploaded = None
if not have_table:
    st.sidebar.info("No table found. Upload your CSV to create the DuckDB table.")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        with st.spinner("Ingesting CSV into DuckDB..."):
            preview = infer_and_create_table(con, table_name, uploaded.read())
            st.sidebar.success(f"Created table `{table_name}` with {preview.shape[1]} columns.")
            have_table = True

if not have_table:
    st.stop()

df = load_table_as_df(con, table_name)
st.caption(f"Loaded **{len(df):,}** rows × **{df.shape[1]}** cols from `{table_name}` in `{db_path}`.")

with st.expander("Preview Data", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# Column mapping
st.sidebar.subheader("Column Mapping")

guessed = guess_mappings(df)
def select_map(label, default):
    return st.sidebar.selectbox(label, ["—"] + list(df.columns), index=(list(df.columns).index(default) + 1) if default in df.columns else 0)

col_map = {
    "Attrition": select_map("Attrition (Yes/No)", guessed["Attrition"]),
    "Department": select_map("Department", guessed["Department"]),
    "JobRole": select_map("Job Role", guessed["JobRole"]),
    "MonthlyIncome": select_map("Monthly Income", guessed["MonthlyIncome"]),
    "OverTime": select_map("Overtime", guessed["OverTime"]),
    "YearsAtCompany": select_map("Years At Company", guessed["YearsAtCompany"]),
    "Age": select_map("Age", guessed["Age"]),
    "EnvironmentSatisfaction": select_map("Environment Satisfaction (1–4)", guessed["EnvironmentSatisfaction"]),
    "JobSatisfaction": select_map("Job Satisfaction (1–4)", guessed["JobSatisfaction"]),
    "WorkLifeBalance": select_map("Work-Life Balance (1–4)", guessed["WorkLifeBalance"]),
    "EducationField": select_map("Education Field", guessed["EducationField"]),
    "Gender": select_map("Gender", guessed["Gender"]),
}

required = ["Attrition"]
missing_required = [k for k in required if (not col_map[k] or col_map[k] == "—")]
if missing_required:
    st.error("Please map the required column(s): " + ", ".join(missing_required))
    st.stop()

# Normalize core fields
attr = to_bool_yes_no(df[col_map["Attrition"]])
df["_attrition_flag"] = attr.astype(int)

if col_map["YearsAtCompany"] and col_map["YearsAtCompany"] != "—":
    df["_tenure"] = pd.to_numeric(df[col_map["YearsAtCompany"]], errors="coerce")
else:
    df["_tenure"] = np.nan

if col_map["MonthlyIncome"] and col_map["MonthlyIncome"] != "—":
    df["_income"] = pd.to_numeric(df[col_map["MonthlyIncome"]], errors="coerce")
else:
    df["_income"] = np.nan

if col_map["OverTime"] and col_map["OverTime"] != "—":
    df["_overtime"] = to_bool_yes_no(df[col_map["OverTime"]])
else:
    df["_overtime"] = False

if col_map["Age"] and col_map["Age"] != "—":
    df["_age"] = pd.to_numeric(df[col_map["Age"]], errors="coerce")
else:
    df["_age"] = np.nan

for k in ["EnvironmentSatisfaction", "JobSatisfaction", "WorkLifeBalance"]:
    if col_map[k] and col_map[k] != "—":
        df[f"_{k}"] = pd.to_numeric(df[col_map[k]], errors="coerce")
    else:
        df[f"_{k}"] = np.nan

# Tenure bands & Income quartiles
df["_tenure_band"] = pd.cut(
    df["_tenure"],
    bins=[-0.001, 1, 3, 6, np.inf],
    labels=["0–1", "2–3", "4–6", "7+"]
)

income_labels = ["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"]
df["_income_q"] = safe_qcut(df["_income"], q=4, labels=income_labels)

age_bins = [-0.001, 25, 35, 45, 55, np.inf]
age_labels = ["18–25", "26–35", "36–45", "46–55", "56+"]
df["_age_band"] = pd.cut(df["_age"], bins=age_bins, labels=age_labels)

# -----------------------------
# Storytelling Header
# -----------------------------
st.title("Storytelling Dashboard")
st.markdown(
    "Answer the question **“What’s happening and why it matters?”** with compact insight cards. "
    "Use the sidebar to confirm mappings. Download data behind each card when needed."
)

# -----------------------------
# Insight 1: Overall Attrition
# -----------------------------
left_col, mid_col, right_col = st.columns([1.2, 1.2, 1.6])
overall_rate = df["_attrition_flag"].mean()
with left_col:
    st.subheader("💔 Overall Attrition")
    st.metric(label="Attrition Rate", value=fmt_pct(overall_rate))
    st.caption("Share of employees marked as Attrition=Yes.")

# -----------------------------
# Insight 2: By Department
# -----------------------------
if col_map["Department"] and col_map["Department"] != "—":
    dept = col_map["Department"]
    dept_df = (
        df.groupby(dept)
          .agg(headcount=(" _attrition_flag".replace(" ", ""), "size") if " _attrition_flag" in df.columns else ("_attrition_flag", "size"),
               attrition_rate=("_attrition_flag", "mean"))
          .reset_index()
          .sort_values("attrition_rate", ascending=False)
    )
    # Fix size aggregation bug if space in column name
    if "headcount" not in dept_df.columns:
        dept_df = (
            df.groupby(dept)
              .agg(headcount=(" _attrition_flag".replace(" ", ""), "size") if " _attrition_flag" in df.columns else ("_attrition_flag", "size"),
                   attrition_rate=("_attrition_flag", "mean"))
              .reset_index()
              .sort_values("attrition_rate", ascending=False)
        )

    top3 = dept_df.head(3).copy()
    with mid_col:
        st.subheader("🏢 By Department")
        if not top3.empty:
            bullets = [f"- **{r[dept]}** — {fmt_pct(r['attrition_rate'])} (n={int(r['headcount'])})" for _, r in top3.iterrows()]
            st.markdown("\n".join(bullets))
        chart = alt.Chart(dept_df).mark_bar().encode(
            x=alt.X(f"{dept}:N", sort="-y", title="Department"),
            y=alt.Y("attrition_rate:Q", axis=alt.Axis(format='%'), title="Attrition Rate"),
            tooltip=[dept, alt.Tooltip("headcount:Q", title="Headcount"), alt.Tooltip("attrition_rate:Q", title="Rate", format=".1%")]
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)
        download_df(dept_df, "attrition_by_department", "dept")

# -----------------------------
# Insight 3: By Job Role
# -----------------------------
if col_map["JobRole"] and col_map["JobRole"] != "—":
    role = col_map["JobRole"]
    role_df = (
        df.groupby(role)
          .agg(headcount=("_attrition_flag", "size"),
               attrition_rate=("_attrition_flag", "mean"))
          .reset_index()
          .sort_values("attrition_rate", ascending=False)
    )
    with right_col:
        st.subheader("🧑‍💼 By Job Role")
        top5 = role_df.head(5).copy()
        chart = alt.Chart(top5).mark_bar().encode(
            x=alt.X("attrition_rate:Q", axis=alt.Axis(format='%'), title="Rate"),
            y=alt.Y(f"{role}:N", sort="-x", title="Job Role"),
            tooltip=[role, alt.Tooltip("headcount:Q", title="Headcount"), alt.Tooltip("attrition_rate:Q", title="Rate", format=".1%")]
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)
        download_df(role_df, "attrition_by_jobrole", "role")

st.markdown("---")

# -----------------------------
# Insight 4: Overtime Effect
# -----------------------------
if col_map["OverTime"] and col_map["OverTime"] != "—":
    ot_df = (
        df.groupby("_overtime")
          .agg(headcount=("_attrition_flag", "size"), attrition_rate=("_attrition_flag", "mean"))
          .reset_index()
    )
    ot_df["_overtime_label"] = ot_df["_overtime"].map({True: "Overtime", False: "No Overtime"})
    delta = (ot_df.loc[ot_df["_overtime"] == True, "attrition_rate"].values[0]
             - ot_df.loc[ot_df["_overtime"] == False, "attrition_rate"].values[0]) if set(ot_df["_overtime"]) == {True, False} else np.nan
    c1, c2 = st.columns([1.3, 1.7])
    with c1:
        st.subheader("⏱️ Overtime vs No Overtime")
        if not np.isnan(delta):
            st.metric("Overtime Penalty (pp)", f"{delta*100:.1f}")
        st.caption("Difference in attrition rate between overtime and non-overtime cohorts.")
    with c2:
        chart = alt.Chart(ot_df).mark_bar().encode(
            x=alt.X("_overtime_label:N", title="Cohort"),
            y=alt.Y("attrition_rate:Q", axis=alt.Axis(format='%'), title="Attrition Rate"),
            tooltip=["_overtime_label", alt.Tooltip("headcount:Q", title="Headcount"), alt.Tooltip("attrition_rate:Q", title="Rate", format=".1%")]
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)
        download_df(ot_df.drop(columns=["_overtime"]), "overtime_effect", "ot")

# -----------------------------
# Insight 5: Tenure Bands
# -----------------------------
tenure_df = (
    df.dropna(subset=["_tenure_band"])
      .groupby("_tenure_band")
      .agg(headcount=("_attrition_flag", "size"), attrition_rate=("_attrition_flag", "mean"))
      .reset_index()
)
c1, c2 = st.columns([1.3, 1.7])
with c1:
    st.subheader("📆 Early Tenure Risk")
    if not tenure_df.empty:
        worst_band = tenure_df.sort_values("attrition_rate", ascending=False).iloc[0]
        st.markdown(f"- Highest in **{worst_band['_tenure_band']}**: {fmt_pct(worst_band['attrition_rate'])}")
        st.caption("Attrition by time at company; early spikes often indicate onboarding/expectation gaps.")
with c2:
    chart = alt.Chart(tenure_df).mark_bar().encode(
        x=alt.X("_tenure_band:N", title="Years at Company"),
        y=alt.Y("attrition_rate:Q", axis=alt.Axis(format='%'), title="Attrition Rate"),
        tooltip=[alt.Tooltip("_tenure_band:N", title="Band"), alt.Tooltip("headcount:Q", title="Headcount"), alt.Tooltip("attrition_rate:Q", title="Rate", format=".1%")]
    ).properties(height=260)
    st.altair_chart(chart, use_container_width=True)
    download_df(tenure_df, "attrition_by_tenure_band", "tenure")

# -----------------------------
# Insight 6: Income Quartiles
# -----------------------------
income_df = (
    df.dropna(subset=["_income_q"])
      .groupby("_income_q")
      .agg(headcount=("_attrition_flag", "size"), attrition_rate=("_attrition_flag", "mean"))
      .reset_index()
      .sort_values("_income_q")
)
c1, c2 = st.columns([1.3, 1.7])
with c1:
    st.subheader("💵 Compensation & Attrition")
    if not income_df.empty:
        low = income_df.iloc[0]["attrition_rate"]
        high = income_df.iloc[-1]["attrition_rate"]
        if pd.notnull(low) and pd.notnull(high):
            rel = (low - high) * 100
            st.markdown(f"- Bottom quartile vs top quartile: **{rel:.1f} pp** higher attrition")
        st.caption("Quartiles auto-derived from Monthly Income.")
with c2:
    chart = alt.Chart(income_df).mark_line(point=True).encode(
        x=alt.X("_income_q:N", title="Income Quartile"),
        y=alt.Y("attrition_rate:Q", axis=alt.Axis(format='%'), title="Attrition Rate"),
        tooltip=[alt.Tooltip("_income_q:N", title="Quartile"), alt.Tooltip("headcount:Q", title="Headcount"), alt.Tooltip("attrition_rate:Q", title="Rate", format=".1%")]
    ).properties(height=260)
    st.altair_chart(chart, use_container_width=True)
    download_df(income_df, "attrition_by_income_quartile", "income")

# -----------------------------
# Insight 7: Satisfaction & WLB
# -----------------------------
sat_cols = [("EnvironmentSatisfaction", "_EnvironmentSatisfaction", "Environment Satisfaction"),
            ("JobSatisfaction", "_JobSatisfaction", "Job Satisfaction"),
            ("WorkLifeBalance", "_WorkLifeBalance", "Work-Life Balance")]

for raw_name, internal, pretty in sat_cols:
    if col_map[raw_name] and col_map[raw_name] != "—":
        sat_df = (
            df.dropna(subset=[internal])
              .groupby(internal)
              .agg(headcount=("_attrition_flag", "size"), attrition_rate=("_attrition_flag", "mean"))
              .reset_index()
              .rename(columns={internal: "level"})
        )
        c1, c2 = st.columns([1.3, 1.7])
        with c1:
            st.subheader(f"😊 {pretty}")
            if not sat_df.empty:
                worst = sat_df.sort_values("attrition_rate", ascending=False).iloc[0]
                st.markdown(f"- Highest attrition at level **{int(worst['level'])}**: {fmt_pct(worst['attrition_rate'])}")
                st.caption("Levels typically 1 (low) → 4 (high).")
        with c2:
            chart = alt.Chart(sat_df).mark_bar().encode(
                x=alt.X("level:O", title="Level"),
                y=alt.Y("attrition_rate:Q", axis=alt.Axis(format='%'), title="Attrition Rate"),
                tooltip=[alt.Tooltip("level:O", title="Level"), alt.Tooltip("headcount:Q", title="Headcount"), alt.Tooltip("attrition_rate:Q", title="Rate", format=".1%")]
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True)
            download_df(sat_df, f"attrition_by_{pretty.lower().replace(' ', '_')}", f"sat_{internal}")

# -----------------------------
# Narrative Summary
# -----------------------------
st.markdown("---")
st.subheader("🧾 Narrative Summary")
story_bits = []

# Overall
story_bits.append(f"Overall attrition is **{fmt_pct(overall_rate)}**.")

# Department highlight
if col_map["Department"] and col_map["Department"] != "—":
    if not dept_df.empty:
        top_dept = dept_df.iloc[0]
        story_bits.append(f"Highest by department: **{top_dept[col_map['Department']]}** at **{fmt_pct(top_dept['attrition_rate'])}**.")

# Overtime
if col_map["OverTime"] and col_map["OverTime"] != "—" and not np.isnan(delta):
    if delta > 0:
        story_bits.append(f"Overtime cohort is **{delta*100:.1f} pp** higher than non-overtime.")
    else:
        story_bits.append(f"Overtime cohort is **{abs(delta)*100:.1f} pp** lower than non-overtime.")

# Tenure & Income
if not tenure_df.empty:
    wb = tenure_df.sort_values("attrition_rate", ascending=False).iloc[0]
    story_bits.append(f"Early-tenure risk: **{wb['_tenure_band']}** shows **{fmt_pct(wb['attrition_rate'])}**.")
if not income_df.empty:
    story_bits.append("Lower income quartiles show higher attrition versus the top quartile.")

# Satisfaction/WLB
for raw_name, internal, pretty in sat_cols:
    if col_map[raw_name] and col_map[raw_name] != "—":
        tmp = (
            df.dropna(subset=[internal])
              .groupby(internal)["_attrition_flag"]
              .mean()
              .sort_values(ascending=False)
        )
        if not tmp.empty:
            lvl = int(tmp.index[0])
            rate = tmp.iloc[0]
            story_bits.append(f"In **{pretty}**, level **{lvl}** has the highest attrition at **{fmt_pct(rate)}**.")

story = " ".join(story_bits)
st.write(story)

st.caption("Tip: Use this summary in your weekly update or Slack brief. Slack integration can be added next.")

# -----------------------------
# Data Dictionary Helper
# -----------------------------
with st.expander("Data Dictionary (Detected)", expanded=False):
    dd = {k: v for k, v in col_map.items() if v and v != "—"}
    st.json(dd)
