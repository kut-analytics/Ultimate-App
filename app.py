"""
Mwangaza Intelligence Chat -- friends & family test build.

A minimal chat interface over the Mwangaza EPV register: ask a question
in plain language, get a synthesized answer plus the supporting rows
in a table. No tiers, no billing -- just the core query loop, meant to
be shared with a handful of people for early feedback.
"""

import json
import re
from datetime import date, timedelta

import streamlit as st
import psycopg2
from anthropic import Anthropic

# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Mwangaza Intelligence Chat", page_icon="🔎", layout="centered")

PG_CONN_STRING = st.secrets["PG_CONN_STRING"]
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-5"

client = Anthropic(api_key=ANTHROPIC_API_KEY)


@st.cache_resource
def get_connection():
    conn = psycopg2.connect(PG_CONN_STRING, connect_timeout=10)
    # autocommit so one failed query never poisons the cached connection for
    # every question after it -- without this, a single bad query leaves the
    # transaction stuck in a failed state until the app is restarted.
    conn.autocommit = True
    return conn


# Known vocabulary -- the filter-parsing step is only allowed to pick from
# these lists, so a model mistake can't turn into an arbitrary SQL value.
COUNTIES = [
    "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Kiambu", "Murang'a", "Nyeri",
    "Kirinyaga", "Laikipia", "Nyandarua", "Homa Bay", "Migori", "Kisii",
    "Nyamira", "Siaya", "Bomet", "Narok", "Kajiado", "Bungoma", "Busia",
    "Kakamega", "Taita Taveta", "Kwale", "Garissa", "Marsabit", "Turkana",
    "Uasin Gishu", "Elgeyo-Marakwet", "Nandi", "Baringo", "Trans Nzoia",
    "West Pokot", "Samburu", "Meru", "Tharaka-Nithi", "Embu", "Kitui",
    "Machakos", "Makueni", "Kericho", "Vihiga", "Wajir", "Mandera",
    "Isiolo", "Tana River", "Lamu", "Kilifi",
]
INTELLIGENCE_AREA = ["Elections and Political Violence", "Internal Instability, Conflict and Social Unrest"]
INSTABILITY_TYPE = ["Intercommunal conflict", "Land or resource conflict", "Cattle rustling",
                     "Banditry", "Protest or demonstration", "Riot or civil disorder",
                     "Labour unrest", "Other internal insecurity"]

FILTER_PROMPT = f"""You turn a question about Kenya political violence and instability into
a structured filter. Today's date context: the register covers 2024 onward.

Return ONLY a JSON object with these fields, all optional except none are required:
- "county": one exact value from {COUNTIES}, or null if not location-specific
- "intelligence_area": one of {INTELLIGENCE_AREA}, or null for both
- "instability_type": one of {INSTABILITY_TYPE}, or null
- "date_start": "YYYY-MM-DD" or null
- "date_end": "YYYY-MM-DD" or null
- "keyword": a single plain-text word or short phrase to search in the description, or null

Only fill a field if the question clearly implies it. Never invent a county or date
that was not mentioned or implied.
"""


def parse_filters(question: str) -> dict:
    resp = client.messages.create(
        model=MODEL, max_tokens=300,
        system=[{"type": "text", "text": FILTER_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": question}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text")
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return {}
    # validate against known vocab -- never trust the model's output directly in SQL
    if parsed.get("county") not in COUNTIES:
        parsed["county"] = None
    if parsed.get("intelligence_area") not in INTELLIGENCE_AREA:
        parsed["intelligence_area"] = None
    if parsed.get("instability_type") not in INSTABILITY_TYPE:
        parsed["instability_type"] = None
    return parsed


def run_query(filters: dict, limit: int = 200):
    conn = get_connection()
    clauses = ["fe.date_occurred IS NOT NULL"]
    params = []

    if filters.get("county"):
        clauses.append("dl.county = %s")
        params.append(filters["county"])
    if filters.get("intelligence_area"):
        clauses.append("fe.intelligence_area = %s")
        params.append(filters["intelligence_area"])
    if filters.get("instability_type"):
        clauses.append("fe.instability_type = %s")
        params.append(filters["instability_type"])
    if filters.get("date_start"):
        clauses.append("fe.date_occurred >= %s")
        params.append(filters["date_start"])
    if filters.get("date_end"):
        clauses.append("fe.date_occurred <= %s")
        params.append(filters["date_end"])
    if filters.get("keyword"):
        clauses.append("(o1.description ILIKE %s OR o2.description ILIKE %s)")
        params.append(f"%{filters['keyword']}%")
        params.append(f"%{filters['keyword']}%")

    query = f"""
        SELECT fe.event_id, fe.date_occurred, dl.county, dl.constituency, fe.temporal_status,
               fe.assertion_status, fe.is_confirmed_incident, fe.figure_confidence,
               fe.confirmed_fatalities, fe.confirmed_injuries, fe.confirmed_arrests,
               COALESCE(o1.description, o2.description) AS description,
               COALESCE(o1.evidence_sentence, o2.evidence_sentence) AS evidence_sentence,
               COALESCE(o1.source_outlet, o2.source_outlet) AS source_outlet,
               COALESCE(o1.source_url, o2.source_url) AS source_url
        FROM fact_events fe
        LEFT JOIN dim_location dl ON dl.location_id = fe.location_id
        LEFT JOIN observations o1 ON o1.observation_id = fe.event_id
        LEFT JOIN event_resolutions er ON er.resolution_id = fe.event_id
        LEFT JOIN observations o2 ON o2.observation_id = er.basis_observation_id
        WHERE {' AND '.join(clauses)}
        ORDER BY fe.date_occurred DESC
        LIMIT {limit}
    """
    cur = conn.cursor()
    cur.execute(query, params)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    cur.close()
    return rows


def run_monthly_counts(filters: dict):
    """Real counts by month, using fact_events -- one row per real event,
    already deduplicated, with temporal_status kept separate from raw
    assertion_status so a trend question doesn't conflate a retrospective
    reference or a future warning with something that actually happened
    in that month."""
    conn = get_connection()
    clauses = ["fe.date_occurred IS NOT NULL"]
    params = []
    if filters.get("county"):
        clauses.append("dl.county = %s")
        params.append(filters["county"])
    if filters.get("intelligence_area"):
        clauses.append("fe.intelligence_area = %s")
        params.append(filters["intelligence_area"])
    if filters.get("instability_type"):
        clauses.append("fe.instability_type = %s")
        params.append(filters["instability_type"])
    if filters.get("date_start"):
        clauses.append("fe.date_occurred >= %s")
        params.append(filters["date_start"])
    if filters.get("date_end"):
        clauses.append("fe.date_occurred <= %s")
        params.append(filters["date_end"])
    if filters.get("keyword"):
        clauses.append("(o1.description ILIKE %s OR o2.description ILIKE %s)")
        params.append(f"%{filters['keyword']}%")
        params.append(f"%{filters['keyword']}%")

    query = f"""
        SELECT to_char(fe.date_occurred, 'YYYY-MM') AS month,
               fe.temporal_status,
               fe.is_confirmed_incident,
               count(*) AS n
        FROM fact_events fe
        LEFT JOIN dim_location dl ON dl.location_id = fe.location_id
        LEFT JOIN observations o1 ON o1.observation_id = fe.event_id
        LEFT JOIN event_resolutions er ON er.resolution_id = fe.event_id
        LEFT JOIN observations o2 ON o2.observation_id = er.basis_observation_id
        WHERE {' AND '.join(clauses)}
        GROUP BY month, fe.temporal_status, fe.is_confirmed_incident
        ORDER BY month
    """
    cur = conn.cursor()
    cur.execute(query, params)
    counts = cur.fetchall()
    cur.close()
    return counts


SYNTHESIS_PROMPT = """You answer questions about Kenya political violence and instability using
ONLY the data provided. Never state anything not supported by it.

You get two things:
1. MONTHLY COUNTS -- the real, complete, DEDUPLICATED count of distinct events per month
   from the reporting mart, broken down by temporal_status and whether each is a confirmed
   incident. This is the authoritative source for any question about trend, frequency, or
   change over time -- always ground a trend answer in these counts, not in the sample rows.
   IMPORTANT: only "current period occurrence" rows where is_confirmed_incident is true are
   actual events that happened in that month. "retrospective reference" rows are mentions of
   earlier events (e.g. 2024, 2007-2008), not new incidents -- never count them as part of a
   trend for the period being asked about. "anticipatory statement" rows are about the future
   or hypothetical, not something that has happened.
2. SAMPLE ROWS -- a representative sample of individual events (not necessarily all of them)
   to cite specific examples and evidence.

Rules:
- If a row or count's assertion_status is "alleged", "warned against", "denied",
  "advisory", or "hypothetical", say so plainly -- never present it as a
  confirmed occurrence. When summarizing a trend, note if a large share of
  the count is non-occurrence (warnings, advisories) rather than confirmed events.
- If nothing is returned, say plainly that the register has no matching
  observations -- do not guess or fill in general knowledge.
- Keep the answer to 2-5 sentences. Be specific about counts, locations and
  dates when the data supports it.
"""


def synthesize_answer(question: str, rows: list, monthly_counts: list) -> str:
    if not rows and not monthly_counts:
        return "The register has no observations matching that question."

    counts_text = "\n".join(f"- {month} | {status}: {n}" for month, status, n in monthly_counts) or "none"

    # Sample spread evenly across the full result set, not just the newest
    # rows, so a trend question sees the whole date range, not just its tail.
    sample = rows if len(rows) <= 40 else rows[::max(1, len(rows) // 40)][:40]
    rows_text = "\n".join(
        f"- {r['date_occurred']} | {r['county']} | temporal_status={r['temporal_status']} | "
        f"assertion={r['assertion_status']} | confirmed_incident={r['is_confirmed_incident']} | {r['description']}"
        for r in sample
    ) or "none"

    resp = client.messages.create(
        model=MODEL, max_tokens=500,
        system=[{"type": "text", "text": SYNTHESIS_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content":
                   f"QUESTION: {question}\n\nMONTHLY COUNTS:\n{counts_text}\n\nSAMPLE ROWS:\n{rows_text}"}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.title("🔎 Mwangaza Intelligence Chat")
st.caption("Test build — ask a question about political violence or instability in Kenya. "
           "Answers are drawn only from Mwangaza's structured register.")

if "history" not in st.session_state:
    st.session_state.history = []
if "preset_question" not in st.session_state:
    st.session_state.preset_question = None

# Preset questions -- common election-related questions a tester can click
# instead of typing, not tied to any specific county. Short label shown on
# the button, full question sent to the model.
PRESET_QUESTIONS = [
    ("Election violence overview", "What election-related violence has been reported in Kenya recently?"),
    ("Most affected counties", "Which counties recorded the most political violence?"),
    ("Most common violence types", "What types of election-related violence are most frequently reported?"),
    ("Actors involved", "Which actors are most commonly associated with reported incidents?"),
    ("Trend over time", "How has election-related violence changed over the past few months?"),
    ("Intimidation & threats", "What forms of intimidation or threats against political actors have been reported?"),
    ("Key recent developments", "What are the key developments in Kenya's electoral environment recently?"),
]

st.caption("Or try one of these:")
row1 = st.columns(4)
row2 = st.columns(3)
button_cols = list(row1) + list(row2)
for col, (label, full_question) in zip(button_cols, PRESET_QUESTIONS):
    if col.button(label, use_container_width=True):
        st.session_state.preset_question = full_question

question = st.chat_input("Ask a question, e.g. 'What happened in Homa Bay in August 2026?'")
if st.session_state.preset_question:
    question = st.session_state.preset_question
    st.session_state.preset_question = None

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if entry["rows"]:
            st.dataframe(
                [
                    {
                        "Date": r["date_occurred"],
                        "County": r["county"],
                        "Status": r["assertion_status"],
                        "Evidence": r["evidence_sentence"],
                        "Source": r["source_outlet"],
                    }
                    for r in entry["rows"]
                ],
                use_container_width=True,
                hide_index=True,
            )

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the register..."):
            filters = parse_filters(question)
            rows = run_query(filters)
            monthly_counts = run_monthly_counts(filters)
            answer = synthesize_answer(question, rows, monthly_counts)
        st.write(answer)
        if rows:
            st.dataframe(
                [
                    {
                        "Date": r["date_occurred"],
                        "County": r["county"],
                        "Status": r["assertion_status"],
                        "Evidence": r["evidence_sentence"],
                        "Source": r["source_outlet"],
                    }
                    for r in rows
                ],
                use_container_width=True,
                hide_index=True,
            )
    st.session_state.history.append({"question": question, "answer": answer, "rows": rows})
