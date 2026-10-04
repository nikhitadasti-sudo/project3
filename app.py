import streamlit as st
import json
import os
import re
import sqlite3
import pandas as pd
import numpy as np
import sqlparse
from langchain_openai import ChatOpenAI

# page config
st.set_page_config(page_title="Credit Risk Query Engine", layout="wide")
st.title("Credit Risk Query Engine")
st.markdown("Ask plain-English questions about the commercial lending portfolio and get verified, auditable answers.")

# load credentials from streamlit secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
OPENAI_API_BASE = st.secrets.get("OPENAI_API_BASE", "")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE

# initialize the LLMs
llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    base_url="https://api.groq.com/openai/v1",
    api_key=OPENAI_API_KEY
)
evaluator_llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    base_url="https://api.groq.com/openai/v1",
    api_key=OPENAI_API_KEY
)

# connect to the database in read-only mode
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect("file:credit_risk_portfolio.db?mode=ro", uri=True)
    return conn

conn = get_db_connection()

# database schema
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier
  sector_name (TEXT): human-readable sector name
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  days_past_due (INTEGER): number of days the account is overdue
  asset_classification (TEXT): Standard, Special Mention, Substandard, Doubtful, Loss
  is_impaired (INTEGER): 1 if the loan is credit-impaired, 0 otherwise
  is_secured (INTEGER): 1 if secured, 0 if unsecured

borrower_rating:
  borrower_id (TEXT, PK component): borrower identifier
  rating_date (DATE, PK component): date of rating assessment
  internal_rating (TEXT): bank-assigned credit rating
  external_rating (TEXT): external agency credit rating
  external_rating_agency (TEXT): name of the external rating agency
  previous_rating (TEXT): rating from the prior cycle
  rating_direction (TEXT): Upgraded, Downgraded, Unchanged
  pd_estimate (REAL): probability of default as a decimal

provisioning:
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): 1, 2, or 3
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd (REAL): loss given default as a decimal
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / ecl_amount as percentage
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modelled
"""

# verified SQL queries
sql_1 = """
SELECT sm.sector_name,
       ROUND(SUM(lm.total_outstanding) / 1e6, 2) AS total_outstanding_mn,
       ROUND(SUM(CASE WHEN lm.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
                      THEN lm.total_outstanding ELSE 0 END) / 1e6, 2) AS npa_exposure_mn
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
GROUP BY sm.sector_name
ORDER BY total_outstanding_mn DESC
"""

sql_2 = """
SELECT loan_category,
       ROUND(SUM(total_outstanding) / 1e6, 2) AS total_outstanding_mn,
       COUNT(*) AS loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_mn DESC
"""

sql_3 = """
SELECT ifrs9_stage,
       COUNT(*) AS loan_count,
       ROUND(SUM(ead_amount) / 1e6, 2) AS total_ead_mn,
       ROUND(SUM(ecl_amount) / 1e6, 2) AS total_ecl_mn
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
"""

sql_4 = """
SELECT sm.sector_name,
       ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_coverage_ratio
FROM provisioning p
JOIN loan_master lm ON p.loan_account_number = lm.loan_account_number
JOIN sector_master sm ON lm.sector_code = sm.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY sm.sector_name
ORDER BY avg_coverage_ratio DESC
"""

sql_5 = """
SELECT lm.borrower_name, sm.sector_name,
       ROUND(lm.total_outstanding / 1e6, 2) AS outstanding_mn,
       lm.asset_classification
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
ORDER BY lm.total_outstanding DESC
LIMIT 10
"""

sql_6 = """
SELECT group_name,
       COUNT(*) AS loan_count,
       ROUND(SUM(total_outstanding) / 1e6, 2) AS total_outstanding_mn
FROM loan_master
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY total_outstanding_mn DESC
LIMIT 5
"""

sql_7 = """
SELECT loan_account_number, borrower_name, sector_code,
       ROUND(total_outstanding / 1e6, 2) AS outstanding_mn,
       days_past_due, asset_classification
FROM loan_master
WHERE days_past_due > 0
ORDER BY days_past_due DESC
"""

sql_8 = """
SELECT CASE
         WHEN days_past_due = 0 THEN '0 (Current)'
         WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
         WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
         WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
         ELSE '90+'
       END AS dpd_bucket,
       COUNT(*) AS loan_count,
       ROUND(SUM(total_outstanding) / 1e6, 2) AS total_outstanding_mn
FROM loan_master
GROUP BY dpd_bucket
ORDER BY MIN(days_past_due)
"""

sql_9 = """
SELECT borrower_id, previous_rating, internal_rating, pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC
"""

sql_10 = """
SELECT reporting_date,
       ROUND(SUM(ecl_amount) / 1e6, 2) AS total_ecl_mn
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date
"""

# verified query library
verified_query_library = {
    'VQ1': {'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors', 'sql': sql_1},
    'VQ2': {'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)', 'sql': sql_2},
    'VQ3': {'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter', 'sql': sql_3},
    'VQ4': {'description': 'Average provision coverage ratio by sector for the latest reporting quarter', 'sql': sql_4},
    'VQ5': {'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level', 'sql': sql_5},
    'VQ6': {'description': 'Top 5 largest exposures aggregated at the business group level', 'sql': sql_6},
    'VQ7': {'description': 'All overdue loan accounts with their days past due and asset classification', 'sql': sql_7},
    'VQ8': {'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio', 'sql': sql_8},
    'VQ9': {'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle', 'sql': sql_9},
    'VQ10': {'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time', 'sql': sql_10},
}


# tool functions

def classify_intent(user_question, query_library, llm_model):
    library_descriptions = "\n".join([f"{qid}: {entry['description']}" for qid, entry in query_library.items()])
    classification_prompt = f"""
You are a credit risk analytics routing engine. Your job is to determine whether a user's question can be answered by one of the pre-approved verified query templates, or whether it requires a new SQL query to be generated.

Here is the user's question:
{user_question}

Here are the available verified query templates:
{library_descriptions}

### INSTRUCTIONS
- If the user's question closely matches one of the verified templates, set route to "verified" and provide the matching query_id.
- If the question does not match any template well enough, set route to "generated" and set query_id to null.
- Consider semantic meaning, not just keyword overlap.

### OUTPUT
Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""
    response = llm_model.invoke(classification_prompt)
    raw = response.content.strip()
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```", "", raw)
    result = json.loads(raw)
    return result


def generate_query(user_question, schema_context, llm_model):
    generation_prompt = f"""
You are a SQL query generator for a credit risk analytics database. Generate a read-only SQLite-compatible SQL query that answers the user's question.

User Question:
{user_question}

Database Schema:
{schema_context}

Rules:
- Use only SELECT statements. No INSERT, UPDATE, DELETE, DROP, or ALTER.
- Use only tables and columns that exist in the schema above.
- Use proper JOINs when combining tables.
- Return only the SQL query with no explanation or markdown formatting.
"""
    response = llm_model.invoke(generation_prompt)
    sql = response.content.strip()
    sql = re.sub(r"```sql\s*", "", sql)
    sql = re.sub(r"```", "", sql)
    return sql


def validate_query(candidate_sql, user_question, db_connection, schema_context, evaluator_llm_model, track="generated"):
    checks = {}
    # check 1: read-only
    upper = candidate_sql.upper()
    for keyword in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]:
        if keyword in upper:
            checks['read_only'] = False
            return False, checks, "Query contains forbidden write operation"
    checks['read_only'] = True

    # check 2: parse/plan dry run
    try:
        db_connection.execute(f"EXPLAIN QUERY PLAN {candidate_sql}")
        checks['parse_plan_dry_run'] = True
    except Exception as e:
        checks['parse_plan_dry_run'] = False
        return False, checks, f"SQL failed to parse or plan: {str(e)}"

    # check 3: relevance check via LLM
    track_context = f"Track: {track}"
    relevance_prompt = f"""
You are a SQL validation expert for a credit risk database. Determine whether the given SQL query correctly answers the user's question.

{track_context}

User Question:
{user_question}

Candidate SQL:
{candidate_sql}

Check whether:
- The correct tables are used
- The right columns and metrics are selected
- Aggregations, filters, and joins are appropriate for the question
- The query logic matches the intent of the question

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
"""
    response = evaluator_llm_model.invoke(relevance_prompt)
    raw = response.content.strip()
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```", "", raw)
    relevance = json.loads(raw)
    checks['relevance'] = relevance

    if relevance.get("verdict", "").lower() != "yes":
        return False, checks, f"Query deemed irrelevant: {relevance.get('reason', '')}"

    return True, checks, relevance.get("confidence", 0.9)


def retry_generation(user_question, failed_sql, error_message, schema_context, llm_model):
    retry_prompt = f"""
You are a SQL query generator for a credit risk database. A previous SQL query failed validation. Generate a corrected read-only SQLite-compatible SQL query that fixes the issue while still answering the user's question.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

Rules:
- Fix the specific error described above.
- Use only SELECT statements. No INSERT, UPDATE, DELETE, DROP, or ALTER.
- Use only tables and columns that exist in the schema.
- Return only the corrected SQL query with no explanation or markdown formatting.
"""
    response = llm_model.invoke(retry_prompt)
    sql = response.content.strip()
    sql = re.sub(r"```sql\s*", "", sql)
    sql = re.sub(r"```", "", sql)
    return sql


def execute_query(validated_sql, db_connection):
    try:
        df = pd.read_sql(validated_sql, db_connection)
        return df, None
    except Exception as e:
        return None, str(e)


def generate_response(user_question, dataframe, llm_model):
    response_prompt = f"""
You are a credit risk analyst writing a response for a business user. Based on the query results below, provide a concise, business-focused answer to the user's question. Use exact figures from the data. Do not include SQL or technical details.

User Question:
{user_question}

Query Results:
{dataframe.to_string()}

Instructions:
- Answer the specific question asked.
- Highlight key figures and trends.
- Keep the response concise and professional.
- Use plain language suitable for non-technical stakeholders.
"""
    response = llm_model.invoke(response_prompt)
    return response.content.strip()


# pipeline
def run_pipeline(user_question, db_connection, query_library, schema_context):
    log = {"user_question": user_question}

    # step 1: classify
    classification = classify_intent(user_question, query_library, llm)
    route = classification.get("route", "generated")
    query_id = classification.get("query_id")
    log["route"] = route
    log["query_id"] = query_id
    log["match_reason"] = classification.get("match_reason", "")

    # step 2: get SQL
    if route == "verified" and query_id in query_library:
        executed_sql = query_library[query_id]["sql"]
        track = "verified"
    else:
        executed_sql = generate_query(user_question, schema_context, llm)
        track = "generated"

    # step 3: validate
    passed, checks, confidence_or_error = validate_query(
        executed_sql, user_question, db_connection, schema_context, evaluator_llm, track
    )

    if not passed:
        # retry once
        executed_sql = retry_generation(user_question, executed_sql, str(confidence_or_error), schema_context, llm)
        passed, checks, confidence_or_error = validate_query(
            executed_sql, user_question, db_connection, schema_context, evaluator_llm, track
        )
        if not passed:
            return {
                "route": route, "query_id": query_id, "executed_sql": executed_sql,
                "dataframe": None, "narrative": f"Validation failed: {confidence_or_error}",
                "confidence": 0.0, "log": log
            }

    # step 4: execute
    df, error = execute_query(executed_sql, db_connection)
    if error:
        return {
            "route": route, "query_id": query_id, "executed_sql": executed_sql,
            "dataframe": None, "narrative": f"Execution error: {error}",
            "confidence": 0.0, "log": log
        }

    # step 5: generate response
    confidence = confidence_or_error if isinstance(confidence_or_error, (int, float)) else 0.9
    narrative = generate_response(user_question, df, llm)

    return {
        "route": route, "query_id": query_id, "executed_sql": executed_sql,
        "dataframe": df, "narrative": narrative,
        "confidence": confidence, "log": log
    }


# streamlit UI
user_question = st.text_input("Enter your question:", placeholder="e.g., What is the total NPA exposure by sector?")

if st.button("Run Query") and user_question:
    with st.spinner("Processing your question..."):
        result = run_pipeline(user_question, conn, verified_query_library, database_schema)

    # display trace
    st.subheader("Pipeline Trace")
    st.write(f"Route: {result['route']}")
    st.write(f"Query ID: {result['query_id']}")
    st.write(f"Confidence: {result['confidence']}")

    # display narrative
    st.subheader("Answer")
    st.markdown(result["narrative"])

    # display SQL
    st.subheader("Executed SQL")
    st.code(result["executed_sql"], language="sql")

    # display data
    if result["dataframe"] is not None:
        st.subheader("Result Data")
        st.dataframe(result["dataframe"])
