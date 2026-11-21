from app import app
c = app.test_client()
score = lambda t: c.post("/predict", json={"text": t}).get_json()["toxicity"]

def test_toxic_flagged():
    assert all(score(t) > .5 for t in ["you worthless idiot", "shut up you stupid clown", "u r such a b1tch"])

def test_clean_not_flagged():
    assert all(score(t) < .5 for t in ["Great stream, loved it!", "baddie", "that boss fight was brutal", "this slaps"])

def test_bad_input():
    assert c.post("/predict", json={"text": "  "}).status_code == 400
    assert c.post("/predict", json={"text": 5}).status_code == 400

def test_endpoints():
    assert c.get("/health").get_json()["status"] == "ok"
    assert c.get("/model-info").get_json()["accuracy"] > .9

import io, json as _json
import app as appmod

def test_youtube_needs_key(monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    assert c.get("/api/youtube-comments?video=oZzQC8NVTeM").status_code == 503
    assert c.get("/api/youtube-comments?video=bad").status_code == 400

def test_youtube_comments_scored(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "x")
    item = lambda a, t: {"snippet": {"topLevelComment": {"snippet": {"authorDisplayName": a, "textDisplay": t, "likeCount": 2, "publishedAt": "2026-01-01T00:00:00Z"}}}}
    fake = _json.dumps({"items": [item("Ann", "great video, thanks"), item("Bob", "you stupid idiot")], "nextPageToken": "T"}).encode()
    monkeypatch.setattr(appmod.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(fake))
    j = c.get("/api/youtube-comments?video=oZzQC8NVTeM").get_json()
    assert [x["author"] for x in j["comments"]] == ["Ann", "Bob"] and j["nextPageToken"] == "T"
    assert j["comments"][0]["toxicity"] < .5 < j["comments"][1]["toxicity"]

def test_batch():
    j = c.post("/predict_batch", json={"texts": ["nice one", "you worthless idiot"]}).get_json()
    assert j["toxicity"][0] < .5 < j["toxicity"][1]
