"""SENTINEL – toxic comment detection API.  Run: python app.py"""
import json, os, re, urllib.error, urllib.parse, urllib.request
from pathlib import Path
import joblib
from flask import Flask, jsonify, redirect, request
from train import METRICS_PATH, MODEL_PATH, clean, train  # noqa: F401  (clean must be importable to unpickle)

def _load_dotenv():  # tiny .env reader (no extra dependency)
    f = Path(__file__).parent / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
_load_dotenv()

YT_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
THRESHOLD = float(os.environ.get("TOXIC_THRESHOLD", "0.5"))
MAX_LEN = 2000

app = Flask(__name__, static_folder="static", static_url_path="/static")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else train(quiet=True)

@app.get("/")
def index():
    return redirect("/static/LiveStream.html")

@app.get("/health")
def health():
    return jsonify(status="ok", threshold=THRESHOLD)

@app.get("/model-info")
def model_info():
    f = METRICS_PATH
    return jsonify(json.loads(f.read_text()) if f.exists() else {})

def _scores(texts):
    return [round(float(p), 4) for p in model.predict_proba([t[:MAX_LEN] for t in texts])[:, 1]] if texts else []

@app.post("/predict_batch")
def predict_batch():
    texts = (request.get_json(silent=True) or {}).get("texts", [])
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts) or len(texts) > 500:
        return jsonify(error="'texts' must be a list of up to 500 strings"), 400
    return jsonify(toxicity=_scores(texts), threshold=THRESHOLD)

@app.get("/api/youtube-comments")
def youtube_comments():
    """Fetch real comments of a YouTube video and score each one with the model."""
    vid, key = request.args.get("video", ""), os.environ.get("YOUTUBE_API_KEY")
    if not re.fullmatch(r"[\w-]{11}", vid):
        return jsonify(error="Invalid YouTube video id"), 400
    if not key:
        return jsonify(error="No YouTube API key set. Add YOUTUBE_API_KEY to a .env file (see README) or use 'Paste comments'."), 503
    params = {"part": "snippet", "videoId": vid, "maxResults": 50, "textFormat": "plainText",
              "order": request.args.get("order", "relevance"), "key": key}
    if request.args.get("page"):
        params["pageToken"] = request.args["page"]
    try:
        with urllib.request.urlopen(f"{YT_URL}?{urllib.parse.urlencode(params)}", timeout=10) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        try: msg = json.load(e)["error"]["message"]
        except Exception: msg = f"HTTP {e.code}"
        return jsonify(error="YouTube API: " + re.sub(r"[?&]key=[^&\s]+", "", msg)), 502
    except Exception as e:
        return jsonify(error=f"Could not reach YouTube: {e}"), 502
    rows = [(i["snippet"]["topLevelComment"]["snippet"]) for i in data.get("items", [])]
    scores = _scores([r["textDisplay"] for r in rows])
    return jsonify(comments=[{"author": r["authorDisplayName"], "text": r["textDisplay"], "likes": r.get("likeCount", 0),
                              "published": r.get("publishedAt"), "toxicity": t} for r, t in zip(rows, scores)],
                   nextPageToken=data.get("nextPageToken"), threshold=THRESHOLD)

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify(error="'text' must be a non-empty string"), 400
    score = float(model.predict_proba([text[:MAX_LEN]])[0][1])
    return jsonify(toxicity=round(score, 4), flagged=score >= THRESHOLD, threshold=THRESHOLD)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=bool(os.environ.get("DEBUG")))
