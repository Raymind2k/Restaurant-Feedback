"""Streamlit dashboard: AI-Powered Restaurant Feedback & Growth Analyzer."""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from analyzer import (analyze_reviews, aspect_summary, extract_aspects,
                      generate_recommendations, top_complaint_phrases, trend_alerts)

st.set_page_config(page_title="Restaurant Feedback Analyzer", page_icon="🍽️", layout="wide")
SENT_COLORS = {"Positive": "#2e9e6b", "Neutral": "#9aa0a6", "Negative": "#d9534f"}
SAMPLE = Path(__file__).parent / "data" / "reviews.csv"


@st.cache_data(show_spinner="Analyzing reviews...")
def run_pipeline(raw: pd.DataFrame):
    df = analyze_reviews(raw)
    return df, extract_aspects(df)


# ------------------------------------------------------------------ data input
st.title("🍽️ Restaurant Feedback & Growth Analyzer")
st.caption("Turn customer reviews into sentiment insights, recurring concerns and action items.")

with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload reviews CSV", type="csv",
                          help="Required columns: date, rating, review_text. Optional: branch, platform")
    if up is not None:
        raw = pd.read_csv(up)
    elif SAMPLE.exists():
        raw = pd.read_csv(SAMPLE)
        st.info("Using sample data. Upload your own CSV to replace it.")
    else:
        st.error("No data found. Run `python generate_sample_data.py` or upload a CSV.")
        st.stop()

missing = {"date", "rating", "review_text"} - set(raw.columns)
if missing:
    st.error(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    st.stop()

df, aspects = run_pipeline(raw)

# ------------------------------------------------------------------ filters
with st.sidebar:
    st.header("Filters")
    dmin, dmax = df["date"].min().date(), df["date"].max().date()
    dr = st.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    if isinstance(dr, tuple) and len(dr) == 2:
        df = df[(df["date"].dt.date >= dr[0]) & (df["date"].dt.date <= dr[1])]
    for col in ("branch", "platform"):
        if col in df.columns:
            opts = sorted(df[col].dropna().unique())
            sel = st.multiselect(col.title(), opts, default=opts)
            df = df[df[col].isin(sel)]
    rating_rng = st.slider("Rating", 1, 5, (1, 5))
    df = df[df["rating"].between(*rating_rng)]

if df.empty:
    st.warning("No reviews match the current filters.")
    st.stop()

aspects_f = aspects[aspects["review_id"].isin(df["review_id"])]
summary = aspect_summary(aspects_f)

# ------------------------------------------------------------------ KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Reviews", f"{len(df):,}")
k2.metric("Avg rating", f"{df['rating'].mean():.2f} ★")
k3.metric("Positive", f"{(df['sentiment'] == 'Positive').mean():.0%}")
k4.metric("Negative", f"{(df['sentiment'] == 'Negative').mean():.0%}")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔍 Customer Concerns", "💡 Recommendations", "🗂️ Review Explorer"])

# ------------------------------------------------------------------ overview
with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        dist = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0).reset_index()
        dist.columns = ["sentiment", "count"]
        fig = px.pie(dist, names="sentiment", values="count", hole=.5, color="sentiment",
                     color_discrete_map=SENT_COLORS, title="Sentiment split")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        monthly = df.groupby("month").agg(avg_rating=("rating", "mean"),
                                          reviews=("rating", "size")).reset_index()
        fig = px.line(monthly, x="month", y="avg_rating", markers=True,
                      title="Average rating by month", range_y=[1, 5])
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        rc = df["rating"].value_counts().sort_index().reset_index()
        rc.columns = ["rating", "count"]
        st.plotly_chart(px.bar(rc, x="rating", y="count", title="Rating distribution"),
                        use_container_width=True)
    with c4:
        ms = df.groupby(["month", "sentiment"]).size().reset_index(name="n")
        st.plotly_chart(px.area(ms, x="month", y="n", color="sentiment", title="Sentiment over time",
                                color_discrete_map=SENT_COLORS), use_container_width=True)

    if "branch" in df.columns:
        b = df.groupby("branch").agg(avg_rating=("rating", "mean"), reviews=("rating", "size"),
                                     negative=("sentiment", lambda s: (s == "Negative").mean())).reset_index()
        b["negative"] = (b["negative"] * 100).round(1)
        b["avg_rating"] = b["avg_rating"].round(2)
        st.subheader("Branch comparison")
        st.dataframe(b.rename(columns={"negative": "negative %"}), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ concerns
with tab2:
    if summary.empty:
        st.info("No aspect mentions detected in the selected reviews.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            s = summary.assign(neg_pct=(summary["neg_rate"] * 100).round(1))
            fig = px.bar(s.sort_values("neg_pct"), x="neg_pct", y="aspect", orientation="h",
                         title="Negative mention rate by aspect (%)", color="neg_pct",
                         color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            a = aspects_f.merge(df[["review_id", "month"]], on="review_id")
            hm = (a.assign(neg=(a["aspect_sentiment"] == "Negative").astype(int))
                    .pivot_table(index="aspect", columns="month", values="neg", aggfunc="mean"))
            fig = px.imshow(hm, aspect="auto", color_continuous_scale="Reds",
                            title="Negative rate: aspect x month", labels=dict(color="neg rate"))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recurring complaint phrases")
        ph = top_complaint_phrases(df)
        if ph.empty:
            st.caption("Not enough negative reviews to mine phrases.")
        else:
            st.plotly_chart(px.bar(ph.sort_values("count"), x="count", y="phrase", orientation="h"),
                            use_container_width=True)

# ------------------------------------------------------------------ recommendations
with tab3:
    alerts = trend_alerts(df, aspects_f)
    if alerts:
        st.subheader("🚨 Emerging issues")
        for a in alerts:
            st.warning(a)

    recs = generate_recommendations(summary)
    st.subheader("Priority actions")
    if not recs:
        st.success("No aspect crosses the concern threshold. Keep monitoring.")
    for r in recs:
        icon = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}[r["severity"]]
        with st.expander(f"{icon} {r['aspect']}  |  {r['severity']} priority  |  {r['evidence']}", expanded=True):
            for act in r["actions"]:
                st.markdown(f"- {act}")
            ex = (aspects_f[(aspects_f.aspect == r["aspect"]) & (aspects_f.aspect_sentiment == "Negative")]
                  .merge(df[["review_id", "review_text"]], on="review_id")
                  .sort_values("aspect_score").head(3))
            if not ex.empty:
                st.caption("Example customer feedback:")
                for t in ex["review_text"]:
                    st.markdown(f"> {t}")

# ------------------------------------------------------------------ explorer
with tab4:
    q = st.text_input("Search review text")
    sent = st.multiselect("Sentiment", ["Positive", "Neutral", "Negative"], default=["Negative"])
    view = df[df["sentiment"].isin(sent)]
    if q:
        view = view[view["review_text"].str.contains(q, case=False, na=False)]
    cols = [c for c in ["date", "branch", "platform", "rating", "sentiment", "sentiment_score", "review_text"]
            if c in view.columns]
    st.dataframe(view.sort_values("date", ascending=False)[cols], hide_index=True, use_container_width=True)
    st.download_button("Download analyzed CSV", view[cols].to_csv(index=False), "analyzed_reviews.csv")
