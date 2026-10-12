# AI-Powered Restaurant Feedback & Growth Analyzer

Python · Pandas · NLP · Streamlit

Classifies restaurant reviews by sentiment, finds recurring customer concerns
(aspect-based analysis), and turns them into prioritized, actionable recommendations.

## Quick start
```bash
pip install -r requirements.txt
python generate_sample_data.py      # creates data/reviews.csv (synthetic)
streamlit run app.py
```

## Using your own data
Upload a CSV in the sidebar with columns:
`date, rating, review_text` (required) and `branch, platform` (optional).

## How it works
| Stage | File | Technique |
|---|---|---|
| Sentiment | `analyzer.py` | VADER compound score -> Positive / Neutral / Negative |
| Concerns | `analyzer.py` | Sentence-level aspect detection (food, service, wait time, price, cleanliness, ambiance) with per-aspect sentiment |
| Complaint mining | `analyzer.py` | n-gram frequency on negative reviews (scikit-learn) |
| Recommendations | `analyzer.py` | Impact = negative rate x mention share -> ranked actions + trend alerts |
| Dashboard | `app.py` | Streamlit + Plotly, filters, branch comparison, review explorer |

## Ideas to extend
- Swap VADER for a transformer: `pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")`
- Replace keyword aspects with BERTopic or zero-shot classification
- Add LLM-written weekly summaries using the Anthropic API
- Add a Google Places / Zomato scraper or API connector
- Unit tests for `analyzer.py` and Docker + Streamlit Cloud deployment
