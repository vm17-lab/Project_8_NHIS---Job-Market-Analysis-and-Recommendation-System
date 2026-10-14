import html
import os
import re

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Job Recommender", page_icon="💼", layout="wide")

CSV_PATH = "all_upwork_jobs_2024-02-07-2024-03-24.csv"

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1000px;}

.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
    border-radius: 18px; padding: 2rem 2.2rem; margin-bottom: 1.4rem; color: #fff;
}
.hero h1 {margin: 0; font-size: 2rem; color: #fff; padding: 0;}
.hero p  {margin: .4rem 0 0 0; opacity: .9; font-size: 1rem;}

.result-count {font-weight: 600; margin: .6rem 0 .8rem 0; opacity: .85;}

.job-card {
    display: flex; gap: 1rem; align-items: center;
    border: 1px solid rgba(128,128,128,.28); border-radius: 14px;
    padding: 1rem 1.2rem; margin-bottom: .8rem;
    background: rgba(128,128,128,.06);
    transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
.job-card:hover {
    transform: translateY(-2px); border-color: #7c3aed;
    box-shadow: 0 6px 18px rgba(124,58,237,.18);
}
.rank {
    min-width: 38px; height: 38px; border-radius: 50%;
    background: linear-gradient(135deg, #4f46e5, #7c3aed); color: #fff;
    display: flex; align-items: center; justify-content: center; font-weight: 700;
}
.job-body {flex: 1; min-width: 0;}
.job-title {font-size: 1.05rem; font-weight: 600; margin-bottom: .45rem; word-break: break-word;}
.badge {
    display: inline-block; padding: .18rem .65rem; margin: 0 .4rem .3rem 0;
    border-radius: 999px; font-size: .78rem; font-weight: 600;
    background: rgba(128,128,128,.16);
}
.badge.pay    {background: rgba(16,185,129,.18); color: #059669;}
.badge.hourly {background: rgba(59,130,246,.18); color: #2563eb;}
.badge.fixed  {background: rgba(245,158,11,.20); color: #d97706;}
.open-btn {
    text-decoration: none !important; white-space: nowrap;
    background: #7c3aed; color: #fff !important;
    padding: .5rem 1rem; border-radius: 10px; font-weight: 600; font-size: .88rem;
}
.open-btn:hover {background: #6d28d9;}

.empty {
    text-align: center; padding: 2.5rem 1rem; border-radius: 14px;
    border: 1px dashed rgba(128,128,128,.4); opacity: .85;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- NLTK
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
stop_words.update({"looking", "need", "needed", "wanted", "expert",
                   "developer", "designer", "project", "help", "work"})


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    return " ".join(lemmatizer.lemmatize(t) for t in tokens
                    if t not in stop_words and len(t) > 2)


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Preparing data — first run takes 1-2 minutes...")
def build_model(source):
    df = pd.read_csv(source)
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")

    df["country"] = df["country"].fillna("Unknown")
    df["title"] = df["title"].fillna(df["title"].mode()[0])
    df["link"] = df["link"].fillna(df["link"].mode()[0])

    fixed = df["is_hourly"] == False
    hourly = df["is_hourly"] == True
    df.loc[fixed & df["budget"].isnull(), "budget"] = df.loc[fixed, "budget"].median()
    df.loc[hourly, "budget"] = 0.0
    df.loc[hourly & df["hourly_low"].isnull(), "hourly_low"] = df.loc[hourly, "hourly_low"].median()
    df.loc[hourly & df["hourly_high"].isnull(), "hourly_high"] = df.loc[hourly, "hourly_high"].median()
    df.loc[fixed, ["hourly_low", "hourly_high"]] = 0.0

    df["clean_title"] = df["title"].apply(preprocess_text)
    tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
    matrix = tfidf.fit_transform(df["clean_title"])
    return df, tfidf, matrix


def recommend(df, tfidf, matrix, query, top_n, job_type, countries):
    q = tfidf.transform([preprocess_text(query)])
    scores = (matrix @ q.T).toarray().ravel()   # cosine similarity (rows are L2-normalised)

    mask = pd.Series(True, index=df.index)
    if job_type == "Hourly":
        mask &= df["is_hourly"]
    elif job_type == "Fixed-price":
        mask &= ~df["is_hourly"]
    if countries:
        mask &= df["country"].isin(countries)
    scores = scores * mask.values

    idx = [i for i in scores.argsort()[::-1][:top_n] if scores[i] > 0]
    return df.iloc[idx][["title", "country", "is_hourly", "budget",
                         "hourly_low", "hourly_high", "link"]].copy()


# ---------------------------------------------------------------- UI helpers
def job_card(rank, row):
    title = html.escape(str(row["title"]))
    country = html.escape(str(row["country"]))
    link = str(row["link"])
    link = html.escape(link, quote=True) if link.startswith(("http://", "https://")) else "#"

    if row["is_hourly"]:
        type_badge = '<span class="badge hourly">⏱ Hourly</span>'
        pay = f"${row['hourly_low']:,.0f} – ${row['hourly_high']:,.0f} / hr"
    else:
        type_badge = '<span class="badge fixed">📌 Fixed price</span>'
        pay = f"${row['budget']:,.0f}"

    return (
        '<div class="job-card">'
        f'<div class="rank">{rank}</div>'
        '<div class="job-body">'
        f'<div class="job-title">{title}</div>'
        f'{type_badge}'
        f'<span class="badge pay">💰 {pay}</span>'
        f'<span class="badge">🌍 {country}</span>'
        '</div>'
        f'<a class="open-btn" href="{link}" target="_blank" rel="noopener noreferrer">View job ↗</a>'
        '</div>'
    )


# ---------------------------------------------------------------- page
st.markdown(
    '<div class="hero"><h1>💼 Upwork Job Recommender</h1>'
    "<p>Type a job title or skill and find the most similar Upwork jobs instantly.</p></div>",
    unsafe_allow_html=True,
)

source = CSV_PATH if os.path.exists(CSV_PATH) else None
if source is None:
    uploaded = st.file_uploader("Upload the dataset CSV (all_upwork_jobs_...csv)", type="csv")
    if uploaded is None:
        st.info("Place the CSV in the same folder as app.py, or upload it above.")
        st.stop()
    source = uploaded

df, tfidf, matrix = build_model(source)

with st.sidebar:
    st.header("🎛️ Filters")
    top_n = st.slider("Number of results", 5, 50, 10)
    job_type = st.radio("Job type", ["All", "Hourly", "Fixed-price"])
    countries = st.multiselect(
        "Country (optional)", df["country"].value_counts().head(40).index.tolist()
    )
    st.divider()

def set_query(text):
    st.session_state["query"] = text

query = st.text_input(
    "Search",
    key="query",
    placeholder="🔎  e.g. Python developer, SEO writer, WordPress...",
    label_visibility="collapsed",
)

st.caption("Try:")
cols = st.columns(5)
for col, example in zip(cols, ["Python developer", "SEO", "WordPress", "Data entry", "Video editor"]):
    col.button(example, on_click=set_query, args=(example,), use_container_width=True)

if query.strip():
    res = recommend(df, tfidf, matrix, query, top_n, job_type, countries)
    if res.empty:
        st.markdown(
            '<div class="empty"><h3>😕 No matching jobs found</h3>'
            "<p>Try different keywords or remove some filters.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-count">Top {len(res)} matches for “{html.escape(query.strip())}”</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "".join(job_card(i, row) for i, (_, row) in enumerate(res.iterrows(), 1)),
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        '<div class="empty"><h3>👆 Start by typing a job title or skill</h3>'
        "<p>Or click one of the quick examples above.</p></div>",
        unsafe_allow_html=True,
    )