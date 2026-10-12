"""NLP + analytics core: sentiment, aspect detection, complaint mining, recommendations."""
import re
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_sia = SentimentIntensityAnalyzer()

# Domain tuning: VADER is general-purpose, so teach it restaurant vocabulary.
_sia.lexicon.update({
    "wait": 0.0, "waited": -1.8, "waiting": -1.2, "slow": -2.0, "forever": -2.2, "late": -1.8,
    "cold": -1.6, "bland": -2.0, "stale": -2.2, "salty": -1.5, "greasy": -1.8, "undercooked": -2.5,
    "overpriced": -2.3, "expensive": -1.5, "pricey": -1.5, "rude": -2.5, "ignored": -2.2,
    "inattentive": -2.0, "noisy": -1.6, "crowded": -1.2, "cramped": -1.5, "loud": -1.2,
    "dirty": -2.3, "filthy": -3.0, "unclean": -2.2, "bland": -2.0, "tasty": 2.2, "crispy": 1.5,
    "spotless": 2.5, "affordable": 1.8, "quickly": 1.5, "fast": 1.5, "cozy": 1.8,
})
_NO_WAIT = re.compile(r"\bno wait\b|\bwithout (?:a )?wait\b|\bnot (?:a )?long wait\b", re.I)

# Keyword stems per business aspect (prefix-matched on word boundaries).
ASPECTS: Dict[str, List[str]] = {
    "Food Quality": ["taste", "tasty", "flavor", "dish", "curry", "pizza", "burger", "naan",
                     "chicken", "paneer", "fresh", "stale", "bland", "salty", "spic", "portion",
                     "cooked", "greasy", "delicious", "menu", "undercooked", "overcooked"],
    "Service": ["service", "staff", "waiter", "waitress", "server", "manager", "rude", "polite",
                "friendly", "attentive", "ignored", "host", "behavio"],
    "Wait Time": ["wait", "slow", "quick", "fast", "late", "delay", "forever", "seated", "delivery",
                  "took", "hour", "minutes"],
    "Price & Value": ["price", "pricey", "expensive", "overpriced", "cheap", "afford", "value",
                      "worth", "cost", "bill", "money", "rupee"],
    "Cleanliness": ["clean", "dirty", "filthy", "hygien", "spotless", "washroom", "restroom",
                    "unclean", "smell", "cutlery", "table was"],
    "Ambiance": ["ambian", "ambience", "atmosphere", "decor", "music", "noisy", "noise", "loud",
                 "cozy", "crowded", "cramped", "seating", "vibe", "lighting"],
}
_ASPECT_RE = {a: re.compile(r"\b(?:%s)" % "|".join(map(re.escape, kws)), re.I)
              for a, kws in ASPECTS.items()}

POS_T, NEG_T = 0.05, -0.05


def label_from_score(s: float) -> str:
    return "Positive" if s >= POS_T else "Negative" if s <= NEG_T else "Neutral"


def score_text(text: str) -> float:
    text = _NO_WAIT.sub("quick", str(text))
    return _sia.polarity_scores(text)["compound"]


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", str(text)) if s.strip()]


def analyze_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment columns to the review frame."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    out = out.dropna(subset=["review_text", "date", "rating"]).reset_index(drop=True)
    if "review_id" not in out:
        out["review_id"] = np.arange(1, len(out) + 1)
    out["sentiment_score"] = out["review_text"].map(score_text)
    out["sentiment"] = out["sentiment_score"].map(label_from_score)
    out["month"] = out["date"].dt.to_period("M").dt.to_timestamp()
    return out


def extract_aspects(df: pd.DataFrame) -> pd.DataFrame:
    """Aspect-based sentiment: one row per (review, aspect) mention."""
    rows = []
    for rid, text in zip(df["review_id"], df["review_text"]):
        per_aspect: Dict[str, List[float]] = {}
        for sent in _sentences(text):
            for aspect, rx in _ASPECT_RE.items():
                if rx.search(sent):
                    per_aspect.setdefault(aspect, []).append(score_text(sent))
        for aspect, scores in per_aspect.items():
            s = float(np.mean(scores))
            rows.append({"review_id": rid, "aspect": aspect,
                         "aspect_score": s, "aspect_sentiment": label_from_score(s)})
    return pd.DataFrame(rows, columns=["review_id", "aspect", "aspect_score", "aspect_sentiment"])


def aspect_summary(aspects: pd.DataFrame) -> pd.DataFrame:
    if aspects.empty:
        return pd.DataFrame(columns=["aspect", "mentions", "neg_rate", "avg_score"])
    g = aspects.groupby("aspect")
    s = pd.DataFrame({
        "mentions": g.size(),
        "neg_rate": g["aspect_sentiment"].apply(lambda x: (x == "Negative").mean()),
        "avg_score": g["aspect_score"].mean(),
    }).reset_index()
    return s.sort_values("neg_rate", ascending=False)


def top_complaint_phrases(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Most frequent 2-3 word phrases inside negative *sentences* of negative reviews."""
    neg = pd.Series([s for t in df.loc[df["sentiment"] == "Negative", "review_text"]
                     for s in _sentences(t) if score_text(s) <= NEG_T])
    if len(neg) < 3:
        return pd.DataFrame(columns=["phrase", "count"])
    vec = CountVectorizer(ngram_range=(2, 3), stop_words="english", min_df=2)
    try:
        X = vec.fit_transform(neg)
    except ValueError:
        return pd.DataFrame(columns=["phrase", "count"])
    counts = np.asarray(X.sum(axis=0)).ravel()
    res = pd.DataFrame({"phrase": vec.get_feature_names_out(), "count": counts})
    return res.sort_values("count", ascending=False).head(n).reset_index(drop=True)


# ---------------------------------------------------------------- recommendations
ACTIONS = {
    "Food Quality": ["Run a menu quality audit: check seasoning, temperature and freshness of flagged dishes.",
                     "Standardize recipes and introduce a pass-check before plates leave the kitchen."],
    "Service": ["Hold a short service-training refresher on attentiveness and order accuracy.",
                "Review floor staffing during peak hours and assign table-section ownership."],
    "Wait Time": ["Map the order-to-table timeline to find the bottleneck (kitchen, seating or delivery).",
                  "Add prep stations or pre-prep for top sellers; set a visible wait-time target."],
    "Price & Value": ["Review portion sizes vs. price on flagged items; consider combo or value meals.",
                      "Communicate value better (portion photos, chef's specials) before changing prices."],
    "Cleanliness": ["Introduce an hourly cleaning checklist for tables and washrooms with sign-off.",
                    "Schedule surprise hygiene inspections and share results with the team."],
    "Ambiance": ["Check noise levels and music volume at peak hours; consider acoustic treatment.",
                 "Rework seating layout to reduce crowding and improve comfort."],
}


def generate_recommendations(summary: pd.DataFrame, min_mentions: int = 10, top_k: int = 3) -> List[dict]:
    """Rank aspects by impact (negative rate x share of mentions) and attach actions."""
    if summary.empty:
        return []
    s = summary[summary["mentions"] >= min_mentions].copy()
    if s.empty:
        return []
    s["impact"] = s["neg_rate"] * (s["mentions"] / s["mentions"].sum())
    s = s.sort_values("impact", ascending=False)
    recs = []
    for _, r in s.head(top_k).iterrows():
        if r["neg_rate"] < 0.15:
            continue
        sev = "High" if r["neg_rate"] >= 0.4 else "Medium" if r["neg_rate"] >= 0.25 else "Low"
        recs.append({"aspect": r["aspect"], "severity": sev,
                     "evidence": f"{r['neg_rate']:.0%} of {int(r['mentions'])} mentions are negative",
                     "actions": ACTIONS[r["aspect"]]})
    return recs


def trend_alerts(df: pd.DataFrame, aspects: pd.DataFrame, window_days: int = 30) -> List[str]:
    """Flag aspects whose negative rate rose notably in the latest window vs. the prior one."""
    if aspects.empty or df.empty:
        return []
    m = aspects.merge(df[["review_id", "date"]], on="review_id")
    end = m["date"].max()
    cur = m[m["date"] > end - pd.Timedelta(days=window_days)]
    prev = m[(m["date"] <= end - pd.Timedelta(days=window_days)) &
             (m["date"] > end - pd.Timedelta(days=2 * window_days))]
    alerts = []
    for a in ASPECTS:
        c, p = cur[cur.aspect == a], prev[prev.aspect == a]
        if len(c) < 5 or len(p) < 5:
            continue
        cr, pr = (c.aspect_sentiment == "Negative").mean(), (p.aspect_sentiment == "Negative").mean()
        if cr - pr >= 0.10:
            alerts.append(f"{a}: negative mentions rose from {pr:.0%} to {cr:.0%} in the last {window_days} days.")
    return alerts
