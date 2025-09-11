# app.py
# Storytelling Dashboard + NL→SQL Agent (DuckDB + Streamlit + Ollama)
# -------------------------------------------------------------------
# What you get
# - CSV -> DuckDB table (upload in sidebar)
# - Insight cards (attrition, dept/role, overtime, tenure, income, satisfaction)
# - NEW: NL→SQL agent tab to ask questions in natural language
#   * Schema-aware prompt built from DESCRIBE <table>
#   * Safe, read-only SQL sanitizer (+ auto LIMIT for big queries)
#   * Shows generated SQL, results table, and a short agent summary
# -------------------------------------------------------------------

import os
import json
import re
from typing import Dict, Optional, List

import duckdb
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import requests

# -----------------------------
# Streamlit & global config
# -----------------------------
st.set_page_config(page_title="NL→SQL Insights Dashboard", layout="wide")
DB_PATH_DEFAULT = os.environ.get("HR_DB_PATH", "data/hr_attrition.duckdb")
TABLE_NAME_DEFAULT = os.environ.get("HR_TABLE_NAME", "employees")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")  # change if remote
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")            # any chat-capable model you pulled

# -----------------------------
# Utils
# -----------------------------
@st.cache_resource
def connect_duckdb(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
    return duckdb.connect(db_path)

def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    try:
        r = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
            [table_name.lower()]
        ).fetchone()
        return bool(r)
    except Exception:
        return False

def infer_and_create_table(con: duckdb.DuckDBPyConnection, table_name: str, csv_bytes: bytes) -> pd.DataFrame:
    tmp_dir = "data/tmp_uploads"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_csv_path = os.path.join(tmp_dir, "upload.csv")
    with open(tmp_csv_path, "wb") as f:
        f.write(csv_bytes)

    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * FROM read_csv_auto('{tmp_csv_path}', SAMPLE_SIZE=-1, IGNORE_ERRORS=true)
    """)
    return con.execute(f"SELECT * FROM {table_name} LIMIT 10").df()

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
    return {
        "Attrition": find_col(cols, ["Attrition","Left","Churn","IsAttrition","Quit"]),
        "Department": find_col(cols, ["Department","Dept"]),
        "JobRole": find_col(cols, ["JobRole","Role","Job Title","Title"]),
        "MonthlyIncome": find_col(cols, ["MonthlyIncome","Monthly Income","Salary","Compensation","Pay"]),
        "OverTime": find_col(cols, ["OverTime","Overtime","Over Time"]),
        "YearsAtCompany": find_col(cols, ["YearsAtCompany","Tenure","Years At Company","YearsAtFirm"]),
        "Age": find_col(cols, ["Age"]),
        "EnvironmentSatisfaction": find_col(cols, ["EnvironmentSatisfaction","EnvSatisfaction"]),
        "JobSatisfaction": find_col(cols, ["JobSatisfaction"]),
        "WorkLifeBalance": find_col(cols, ["WorkLifeBalance","Work Life Balance","WLB"]),
        "EducationField": find_col(cols, ["EducationField","Major","Discipline"]),
        "Gender": find_col(cols, ["Gender","Sex"]),
    }

def to_bool_yes_no(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["yes","y","true","1"])

def safe_qcut(s: pd.Series, q=4, labels=None) -> pd.Series:
    try:
        return pd.qcut(s, q=q, labels=labels, duplicates="drop")
    except Exception:
        return pd.cut(s, bins=min(q, max(1, s.nunique())), labels=labels, include_lowest=True)

def fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"

def download_df(df: pd.DataFrame, label: str, key_suffix: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {label} CSV",
        data=csv,
        file_name=f"{label.replace(' ','_').lower()}.csv",
        mime="text/csv",
        key=f"dl_{key_suffix}"
    )

# -----------------------------
# Sidebar: data bootstrap
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

# -----------------------------
# Column mapping (used by cards)
# -----------------------------
st.sidebar.subheader("Column Mapping")
guessed = guess_mappings(df)
def select_map(label, default):
    return st.sidebar.selectbox(label, ["—"] + list(df.columns), index=(list(df.columns).index(default)+1) if default in df.columns else 0)

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

if not col_map["Attrition"] or col_map["Attrition"] == "—":
    st.error("Please map the required column: Attrition")
    st.stop()

# Normalize fields for cards
df["_attrition_flag"] = to_bool_yes_no(df[col_map["Attrition"]]).astype(int)
df["_tenure"] = pd.to_numeric(df[col_map["YearsAtCompany"]], errors="coerce") if col_map["YearsAtCompany"] and col_map["YearsAtCompany"]!="—" else np.nan
df["_income"] = pd.to_numeric(df[col_map["MonthlyIncome"]], errors="coerce") if col_map["MonthlyIncome"] and col_map["MonthlyIncome"]!="—" else np.nan
df["_overtime"] = to_bool_yes_no(df[col_map["OverTime"]]) if col_map["OverTime"] and col_map["OverTime"]!="—" else False
df["_age"] = pd.to_numeric(df[col_map["Age"]], errors="coerce") if col_map["Age"] and col_map["Age"]!="—" else np.nan
for k in ["EnvironmentSatisfaction","JobSatisfaction","WorkLifeBalance"]:
    df[f"_{k}"] = pd.to_numeric(df[col_map[k]], errors="coerce") if col_map[k] and col_map[k]!="—" else np.nan

df["_tenure_band"] = pd.cut(df["_tenure"], bins=[-0.001,1,3,6,np.inf], labels=["0–1","2–3","4–6","7+"])
df["_income_q"] = safe_qcut(df["_income"], q=4, labels=["Q1 (Lowest)","Q2","Q3","Q4 (Highest)"])
df["_age_band"] = pd.cut(df["_age"], bins=[-0.001,25,35,45,55,np.inf], labels=["18–25","26–35","36–45","46–55","56+"])

# ---------------------------------------
# Tabs: Story cards | NL→SQL Agent
# ---------------------------------------
tab_cards, tab_agent = st.tabs(["📖 Storytelling Cards", "🤖 NL→SQL Agent"])

# =======================================
# 📖 STORY CARDS
# =======================================
with tab_cards:
    st.title("Storytelling Dashboard")
    st.markdown("Compact insight cards that answer **what’s happening** and **why it matters**.")

    left_col, mid_col, right_col = st.columns([1.2, 1.2, 1.6])
    overall_rate = df["_attrition_flag"].mean()
    with left_col:
        st.subheader("💔 Overall Attrition")
        st.metric(label="Attrition Rate", value=fmt_pct(overall_rate))
        st.caption("Share of employees marked as Attrition=Yes.")

    # Department
    if col_map["Department"] and col_map["Department"]!="—":
        dept = col_map["Department"]
        dept_df = df.groupby(dept).agg(headcount=("_attrition_flag","size"),
                                       attrition_rate=("_attrition_flag","mean")).reset_index().sort_values("attrition_rate", ascending=False)
        with mid_col:
            st.subheader("🏢 By Department")
            top3 = dept_df.head(3)
            if not top3.empty:
                bullets = [f"- **{r[dept]}** — {fmt_pct(r['attrition_rate'])} (n={int(r['headcount'])})" for _, r in top3.iterrows()]
                st.markdown("\n".join(bullets))
            chart = alt.Chart(dept_df).mark_bar().encode(
                x=alt.X(f"{dept}:N", sort="-y", title="Department"),
                y=alt.Y("attrition_rate:Q", axis=alt.Axis(format='%'), title="Attrition Rate"),
                tooltip=[dept, alt.Tooltip("headcount:Q","Headcount"), alt.Tooltip("attrition_rate:Q","Rate",format=".1%")]
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True)
            download_df(dept_df, "attrition_by_department", "dept")

    # Role
    if col_map["JobRole"] and col_map["JobRole"]!="—":
        role = col_map["JobRole"]
        role_df = df.groupby(role).agg(headcount=("_attrition_flag","size"),
                                       attrition_rate=("_attrition_flag","mean")).reset_index().sort_values("attrition_rate", ascending=False)
        with right_col:
            st.subheader("🧑‍💼 By Job Role")
            chart = alt.Chart(role_df.head(5)).mark_bar().encode(
                x=alt.X("attrition_rate:Q", axis=alt.Axis(format='%'), title="Rate"),
                y=alt.Y(f"{role}:N", sort="-x", title="Job Role"),
                tooltip=[role, alt.Tooltip("headcount:Q","Headcount"), alt.Tooltip("attrition_rate:Q","Rate",format=".1%")]
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True)
            download_df(role_df, "attrition_by_jobrole", "role")

    st.markdown("---")

    # Overtime
    if col_map["OverTime"] and col_map["OverTime"]!="—":
        ot_df = df.groupby("_overtime").agg(headcount=("_attrition_flag","size"),
                                            attrition_rate=("_attrition_flag","mean")).reset_index()
        ot_df["_overtime_label"] = ot_df["_overtime"].map({True:"Overtime", False:"No Overtime"})
        c1, c2 = st.columns([1.3,1.7])
        with c1:
            st.subheader("⏱️ Overtime vs No Overtime")
            if set(ot_df["_overtime"])=={True,False}:
                delta = float(ot_df.loc[ot_df["_overtime"]==True,"attrition_rate"].values[0] -
                              ot_df.loc[ot_df["_overtime"]==False,"attrition_rate"].values[0])
                st.metric("Overtime Penalty (pp)", f"{delta*100:.1f}")
            st.caption("Difference in attrition rate between cohorts.")
        with c2:
            chart = alt.Chart(ot_df).mark_bar().encode(
                x=alt.X("_overtime_label:N", title="Cohort"),
                y=alt.Y("attrition_rate:Q", axis=alt.Axis(format='%'), title="Attrition Rate"),
                tooltip=["_overtime_label", alt.Tooltip("headcount:Q","Headcount"), alt.Tooltip("attrition_rate:Q","Rate",format=".1%")]
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True)
            download_df(ot_df.drop(columns=["_overtime"]), "overtime_effect", "ot")

    # Tenure
    tenure_df = (
        df.dropna(subset=["_tenure_band"]).groupby("_tenure_band")
          .agg(headcount=("_attrition_flag","size"), attrition_rate=("_attrition_flag","mean")).reset_index()
    )
    c1, c2 = st.columns([1.3,1.7])
    with c1:
        st.subheader("📆 Early Tenure Risk")
        if not tenure_df.empty:
            worst = tenure_df.sort_values("attrition_rate", ascending=False).iloc[0]
            st.markdown(f"- Highest in **{worst['_tenure_band']}**: {fmt_pct(worst['attrition_rate'])}")
    with c2:
        chart = alt.Chart(tenure_df).mark_bar().encode(
            x=alt.X("_tenure_band:N", title="Years at Company"),
            y=alt.Y("attrition_rate:Q", axis=alt.Axis(format='%'), title="Attrition Rate"),
            tooltip=[alt.Tooltip("_tenure_band:N","Band"), alt.Tooltip("headcount:Q","Headcount"), alt.Tooltip("attrition_rate:Q","Rate",format=".1%")]
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)
        download_df(tenure_df, "attrition_by_tenure_band", "tenure")

    # Income
    income_df = (
        df.dropna(subset=["_income_q"]).groupby("_income_q")
          .agg(headcount=("_attrition_flag","size"), attrition_rate=("_attrition_flag","mean")).reset_index()
          .sort_values("_income_q")
    )
    c1, c2 = st.columns([1.3,1.7])
    with c1:
        st.subheader("💵 Compensation & Attrition")
        if not income_df.empty:
            low = float(income_df.iloc[0]["attrition_rate"])
            high = float(income_df.iloc[-1]["attrition_rate"])
            st.markdown(f"- Bottom vs top quartile: **{(low-high)*100:.1f} pp** higher attrition")
    with c2:
        chart = alt.Chart(income_df).mark_line(point=True).encode(
            x=alt.X("_income_q:N", title="Income Quartile"),
            y=alt.Y("attrition_rate:Q", axis=alt.Axis(format='%'), title="Attrition Rate"),
            tooltip=[alt.Tooltip("_income_q:N","Quartile"), alt.Tooltip("headcount:Q","Headcount"), alt.Tooltip("attrition_rate:Q","Rate",format=".1%")]
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)
        download_df(income_df, "attrition_by_income_quartile", "income")

    # Satisfaction & WLB
    sat_specs = [("EnvironmentSatisfaction","Environment Satisfaction"),
                 ("JobSatisfaction","Job Satisfaction"),
                 ("WorkLifeBalance","Work-Life Balance")]
    for raw, pretty in sat_specs:
        if col_map[raw] and col_map[raw]!="—":
            internal = f"_{raw}"
            sat_df = (
                df.dropna(subset=[internal]).groupby(internal)
                  .agg(headcount=("_attrition_flag","size"), attrition_rate=("_attrition_flag","mean"))
                  .reset_index().rename(columns={internal:"level"})
            )
            c1, c2 = st.columns([1.3,1.7])
            with c1:
                st.subheader(f"😊 {pretty}")
                if not sat_df.empty:
                    worst = sat_df.sort_values("attrition_rate", ascending=False).iloc[0]
                    st.markdown(f"- Highest attrition at level **{int(worst['level'])}**: {fmt_pct(worst['attrition_rate'])}")
            with c2:
                chart = alt.Chart(sat_df).mark_bar().encode(
                    x=alt.X("level:O", title="Level"),
                    y=alt.Y("attrition_rate:Q", axis=alt.Axis(format='%'), title="Attrition Rate"),
                    tooltip=[alt.Tooltip("level:O","Level"), alt.Tooltip("headcount:Q","Headcount"), alt.Tooltip("attrition_rate:Q","Rate",format=".1%")]
                ).properties(height=260)
                st.altair_chart(chart, use_container_width=True)
                download_df(sat_df, f"attrition_by_{pretty.lower().replace(' ','_')}", f"sat_{internal}")

    # Narrative
    st.markdown("---")
    st.subheader("🧾 Narrative Summary")
    story_bits = [f"Overall attrition is **{fmt_pct(overall_rate)}**."]
    if col_map["Department"] and col_map["Department"]!="—":
        if not dept_df.empty:
            top_dept = dept_df.iloc[0]
            story_bits.append(f"Highest by department: **{top_dept[col_map['Department']]}** at **{fmt_pct(top_dept['attrition_rate'])}**.")
    if col_map["OverTime"] and col_map["OverTime"]!="—" and set(ot_df["_overtime"])=={True,False}:
        story_bits.append(f"Overtime cohort is **{delta*100:.1f} pp** higher than non-overtime.")
    if not tenure_df.empty:
        wb = tenure_df.sort_values("attrition_rate", ascending=False).iloc[0]
        story_bits.append(f"Early-tenure spike in **{wb['_tenure_band']}** at **{fmt_pct(wb['attrition_rate'])}**.")
    if not income_df.empty:
        story_bits.append("Lower income quartiles show higher attrition vs top quartile.")
    st.write(" ".join(story_bits))

# =======================================
# 🤖 NL→SQL AGENT
# =======================================
with tab_agent:
    st.title("NL→SQL Agent (DuckDB + Ollama)")
    st.caption("Ask questions in plain English. The agent generates **read-only SQL**, runs it, and returns results.")

    # 1) Schema introspection
    schema_df = con.execute(f"DESCRIBE {table_name}").df()
    columns_info = "\n".join([f"- {r['column_name']} ({r['column_type']})" for _, r in schema_df.iterrows()])
    st.markdown("**Schema**")
    st.code("\n".join([f"{r['column_name']:>22}  {r['column_type']}" for _, r in schema_df.iterrows()]))

    # 2) Model discovery
    def ollama_ok() -> bool:
        try:
            requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            return True
        except requests.exceptions.RequestException:
            return False

    models = []
    if ollama_ok():
        try:
            data = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
            models = [m["name"] for m in data.get("models", []) if "name" in m]
        except Exception:
            pass

    model = st.selectbox("Model", models or [DEFAULT_MODEL])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    user_q = st.text_area("Your question", placeholder="e.g., Which 5 job roles have the highest attrition?")

    # 3) System prompt + safety
    SYSTEM_PROMPT = f"""
You are a SQL agent for DuckDB. You can query a single table named "{table_name}".
Only generate **read-only** SQL (SELECT / WITH). Never write, update, delete, drop, or alter.
Return **only** a JSON object/tool-call like:
{{
  "name": "execute_sql_query",
  "arguments": {{
    "query": "SELECT ... LIMIT 200"
  }}
}}
Use ONLY these columns (exactly as named):
{columns_info}
    """.strip()

    def sanitize_sql(q: str) -> str:
        """Reject non-SELECT/CTE and append a LIMIT if missing."""
        q_clean = re.sub(r";+\s*$", "", q.strip(), flags=re.IGNORECASE)
        if not re.match(r"^(with|select)\b", q_clean.strip(), flags=re.IGNORECASE):
            raise ValueError("Only SELECT/WITH queries are allowed.")
        forbidden = r"\b(insert|update|delete|merge|drop|alter|create|grant|revoke|attach|detach|copy)\b"
        if re.search(forbidden, q_clean, flags=re.IGNORECASE):
            raise ValueError("Write/DDL statements are not allowed.")
        # add LIMIT if none present (simple heuristic)
        if not re.search(r"\blimit\s+\d+\b", q_clean, flags=re.IGNORECASE):
            q_clean += " LIMIT 500"
        return q_clean

    def extract_tool_json(text: str) -> Dict:
        # strip code fences
        text = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL)
        text = text.strip()
        # direct parse
        try:
            return json.loads(text)
        except Exception:
            # try to find first JSON object in text
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return {}
        return {}

    def chat_ollama(messages: List[Dict], model_name: str, temp: float = 0.2) -> str:
        payload = {"model": model_name, "messages": messages, "stream": False, "options": {"temperature": temp}}
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    # 4) Ask button
    if st.button("Ask the Agent", type="primary", disabled=not user_q):
        if not ollama_ok():
            st.error("Ollama server is not reachable at "
                     f"{OLLAMA_URL}. Install from https://ollama.com, run `ollama pull {model}` then retry.")
        else:
            with st.spinner("Thinking…"):
                # First call: get tool-call JSON
                content = chat_ollama(
                    messages=[{"role":"system","content": SYSTEM_PROMPT},
                              {"role":"user","content": user_q}],
                    model_name=model,
                    temp=temperature,
                )
                tool = extract_tool_json(content)
                sql = None
                if tool.get("name") == "execute_sql_query":
                    sql = tool.get("arguments", {}).get("query")

                if not sql:
                    st.warning("The model did not return a valid tool call. Showing raw response:")
                    st.code(content)
                else:
                    try:
                        sql_safe = sanitize_sql(sql)
                        st.markdown("**Generated SQL**")
                        st.code(sql_safe, language="sql")
                        result_df = con.execute(sql_safe).df()
                        st.markdown("**Results**")
                        st.dataframe(result_df, use_container_width=True, hide_index=True)
                        download_df(result_df, "nl2sql_result", "nl2sql")

                        # Second call: concise analysis
                        preview_json = result_df.head(20).to_json(orient="records")
                        ANALYZE_PROMPT = f"""
You just generated this SQL:

{sql_safe}

Here are the first rows of the result (JSON):
{preview_json}

In 2–3 short bullet points, explain what stands out. Be concrete, use percentages/counts, and avoid hedging.
                        """.strip()
                        content2 = chat_ollama(
                            messages=[{"role":"system","content": "You are a concise data analyst."},
                                      {"role":"user","content": ANALYZE_PROMPT}],
                            model_name=model,
                            temp=0.2,
                        )
                        st.markdown("**Agent’s Notes**")
                        st.write(content2)
                    except Exception as e:
                        st.error(f"Query blocked or failed: {e}")

    st.info("Tip: If this is your first run, install Ollama and pull a small chat model, e.g.: "
            "`brew install ollama && ollama pull llama3.1` (macOS) or see ollama.com for installers.")

# -----------------------------
# Data dictionary dump
# -----------------------------
with st.expander("Data Dictionary (Mapped Columns)", expanded=False):
    dd = {k: v for k, v in col_map.items() if v and v != "—"}
    st.json(dd)

