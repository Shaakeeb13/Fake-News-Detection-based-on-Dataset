from flask import Flask, render_template, request
import pandas as pd
import pickle
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse

app = Flask(__name__)

# ── Text cleaning (must match train_model.py) ────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Load model artefacts ─────────────────────────────────────────────────────
print("Loading model and dataset…")
model      = pickle.load(open("model.pkl",      "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load pre-cleaned dataset (saved by train_model.py)
try:
    df = pd.read_pickle("dataset_clean.pkl")
    print(f"Dataset loaded: {len(df)} rows")
except FileNotFoundError:
    # Fallback: build from raw CSVs
    fake_df = pd.read_csv("Fake.csv"); fake_df["label"] = 0
    true_df = pd.read_csv("True.csv"); true_df["label"] = 1
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)
    df = df[df["content"].str.strip() != ""].reset_index(drop=True)

# Pre-compute TF-IDF matrix for cosine similarity search
dataset_matrix = vectorizer.transform(df["content"])
print("Dataset matrix ready.")


# ── Scraping helpers ─────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

NEWS_DOMAINS = [
    "timesofindia.indiatimes.com", "thehindu.com", "ndtv.com",
    "indiatoday.in", "hindustantimes.com", "indianexpress.com",
    "telegraphindia.com", "economictimes.indiatimes.com",
    "livemint.com", "deccanherald.com", "news18.com",
    "thewire.in", "scroll.in", "thequint.com", "firstpost.com",
    "theprint.in", "opindia.com", "newslaundry.com",
    "bbc.com/news", "reuters.com", "apnews.com",
]

def scrape_article(url):
    """Return (cleaned_text, raw_excerpt). Raises on failure."""
    resp = requests.get(url, headers=HEADERS, timeout=12)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove clutter
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "noscript", "iframe", "figure"]):
        tag.decompose()

    # Strategy 1: semantic article tag
    article_tag = soup.find("article")
    if article_tag:
        paragraphs = [p.get_text(" ", strip=True) for p in article_tag.find_all("p")]
        body = " ".join(p for p in paragraphs if len(p) > 40)
        if len(body) > 300:
            return clean_text(body), body[:3000]

    # Strategy 2: known content class/id patterns
    content_selectors = [
        "div.article-body", "div.story-content", "div.post-content",
        "div.entry-content", "div.content-body", "div.article__body",
        "div.article-content", "div[itemprop='articleBody']",
        "section.article-body", "div.story-body", "div.main-content",
    ]
    for sel in content_selectors:
        node = soup.select_one(sel)
        if node:
            paragraphs = [p.get_text(" ", strip=True) for p in node.find_all("p")]
            body = " ".join(p for p in paragraphs if len(p) > 40)
            if len(body) > 300:
                return clean_text(body), body[:3000]

    # Strategy 3: all <p> tags with reasonable length
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p") if len(p.get_text()) > 60]
    body = " ".join(paragraphs)
    if len(body) > 300:
        return clean_text(body), body[:3000]

    # Strategy 4: largest text block in any div
    divs = soup.find_all("div")
    best = max(divs, key=lambda d: len(d.get_text()), default=None)
    if best:
        body = best.get_text(" ", strip=True)
        return clean_text(body), body[:3000]

    raise ValueError("Could not extract article text.")


def search_news(query, max_results=6):
    """Search Bing for the query and return [(title, url), ...]."""
    params = {"q": query, "setlang": "en-US", "count": "20"}
    resp = requests.get("https://www.bing.com/search", params=params,
                        headers=HEADERS, timeout=12)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results = []
    for a in soup.select("li.b_algo h2 a, li.b_algo h3 a"):
        href  = a.get("href", "")
        title = a.get_text(strip=True)
        if href.startswith("http") and title:
            results.append((title, href))

    # Prioritise known news domains
    priority = [r for r in results if any(d in r[1] for d in NEWS_DOMAINS)]
    rest     = [r for r in results if r not in priority]
    ordered  = priority + rest

    # Deduplicate by domain
    seen_domains, unique = set(), []
    for title, url in ordered:
        domain = urllib.parse.urlparse(url).netloc
        if domain not in seen_domains:
            seen_domains.add(domain)
            unique.append((title, url))
        if len(unique) >= max_results:
            break
    return unique


def crawl_articles(search_results):
    """Try to scrape each result, return list of (title, url, cleaned, excerpt)."""
    crawled = []
    for title, url in search_results:
        try:
            cleaned, excerpt = scrape_article(url)
            if len(cleaned.split()) > 50:
                crawled.append((title, url, cleaned, excerpt))
        except Exception:
            continue
    return crawled


# ── Prediction helpers ───────────────────────────────────────────────────────
def ml_predict(text):
    """Run the trained ML model on cleaned text."""
    cleaned = clean_text(text)
    vec     = vectorizer.transform([cleaned])
    pred    = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    label   = "TRUE" if pred == 1 else "FAKE"
    confidence = float(max(proba)) * 100
    return {"label": label, "confidence": confidence}


def dataset_similarity(text):
    """Find the most similar article in the dataset via cosine similarity."""
    cleaned = clean_text(text)
    if not cleaned:
        return None
    vec        = vectorizer.transform([cleaned])
    sims       = cosine_similarity(vec, dataset_matrix)[0]
    best_idx   = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_row   = df.iloc[best_idx]
    label      = "TRUE" if best_row["label"] == 1 else "FAKE"
    return {
        "label": label,
        "title": best_row.get("title", "N/A"),
        "score": best_score,
    }


def combined_verdict(text):
    """
    Combine ML model + dataset similarity into a final verdict.
    High similarity (>0.75) -> trust dataset match.
    Otherwise -> trust ML model.
    """
    ml  = ml_predict(text)
    sim = dataset_similarity(text)

    if sim and sim["score"] > 0.75:
        method     = f"Dataset match (similarity {sim['score']:.2f})"
        result     = sim["label"]
        confidence = sim["score"] * 100
    else:
        method     = f"ML model (confidence {ml['confidence']:.1f}%)"
        result     = ml["label"]
        confidence = ml["confidence"]

    return {
        "result":     result,
        "confidence": confidence,
        "method":     method,
        "sim_score":  sim["score"]  if sim else 0.0,
        "sim_title":  sim["title"]  if sim else "N/A",
        "ml_label":   ml["label"],
        "ml_conf":    ml["confidence"],
    }


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    news_text       = request.form.get("news",  "").strip()
    browser_query   = request.form.get("query", "").strip()
    scraped_excerpt = None
    scraped_sources = []

    # ── Path A: search query ─────────────────────────────────────────────────
    if browser_query:
        try:
            search_results = search_news(browser_query)
        except Exception as e:
            return render_template("index.html", error=f"Search failed: {e}")

        if not search_results:
            return render_template("index.html",
                                   error="No search results found for that query.")

        articles = crawl_articles(search_results)
        scraped_sources = [f"{t} — {u}" for t, u, _, _ in articles]

        if not articles:
            return render_template("index.html",
                                   error="Search results found but no article text could be extracted.",
                                   scraped_sources=scraped_sources)

        # Use the longest successfully scraped article for best accuracy
        articles.sort(key=lambda x: len(x[2]), reverse=True)
        _, _, news_text, scraped_excerpt = articles[0]

    # ── Path B: pasted text ──────────────────────────────────────────────────
    elif not news_text:
        return render_template("index.html", error="Please enter some text or a search query.")

    # ✅ FIXED: Compute verdict before rendering
    verdict = combined_verdict(news_text)

    return render_template(
        "index.html",
        result          = verdict["result"],
        confidence      = f"{verdict['confidence']:.1f}",
        method          = verdict["method"],
        ml_label        = verdict["ml_label"],
        ml_conf         = f"{verdict['ml_conf']:.1f}",
        sim_score       = f"{verdict['sim_score']:.3f}",
        sim_title       = verdict["sim_title"],
        scraped_text    = scraped_excerpt,
        scraped_sources = scraped_sources,
    )


if __name__ == "__main__":
    app.run(debug=True)