"""Train the SENTINEL toxicity classifier -> model.joblib (+ metrics.json).

  python train.py                    # real data: data/labeled_data.csv + chat/slang augmentation
  python train.py --data my.csv      # any CSV with text,label (1 = toxic) or Jigsaw comment_text,toxic

Dataset: Davidson et al. 2017, "Automated Hate Speech Detection and the Problem of
Offensive Language" (MIT licence). Classes hate/offensive are merged into "toxic".
"""
import argparse, html, itertools, json, random, re
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).parent
MODEL_PATH, METRICS_PATH = ROOT / "model.joblib", ROOT / "metrics.json"
DEFAULT_DATA = ROOT / "data" / "labeled_data.csv"
URL_RE, USER_RE, RT_RE = re.compile(r"https?://\S+|www\.\S+"), re.compile(r"@\w+"), re.compile(r"\brt\b:?")

def clean(text: str) -> str:
    text = html.unescape(str(text)).lower()
    text = RT_RE.sub(" ", URL_RE.sub(" <url> ", text))
    text = USER_RE.sub(" <user> ", text)
    text = re.sub(r"[^\w\s<>'*$@!]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def chat_augmentation() -> pd.DataFrame:
    """Live-chat style examples (slang, praise, banter vs. real abuse) the tweet data lacks."""
    rng = random.Random(7)
    insults = ["idiot", "moron", "loser", "trash", "scumbag", "imbecile", "dumbass", "clown", "pathetic fool",
               "worthless pig", "jerk", "creep", "bastard", "freak", "piece of garbage", "waste of space"]
    toxic_t = ["you are a {}", "shut up you {}", "{}", "what a bunch of {}s", "i hate you, {}", "go die you {}",
               "nobody likes you, {}", "you stupid {}", "get lost, {}", "only a {} would say that",
               "kill yourself {}", "you are so ugly and dumb, {}", "streamer is a {}", "ban this {}"]
    toxic = [t.format(i) for t, i in itertools.product(toxic_t, insults)]
    toxic += ["i will find you and hurt you", "you should be ashamed, disgusting human", "stfu nobody asked",
              "this is garbage and so are you", "u r a b1tch", "you people are all animals", "f off you loser",
              "uninstall the game and your life", "go back to where you came from", "i hope you get what you deserve, scum"]
    pos = ["great video", "really helpful, thanks", "loved the explanation", "this was so well made",
           "interesting point of view", "congratulations on the launch", "can you share the source?",
           "i disagree but i see your point", "the audio is a bit quiet", "first time watching, subscribed",
           "what camera did you use?", "nice work to the whole team", "this made my day", "i learned a lot from this",
           "could you make a follow up on this topic?", "the editing is fantastic", "thank you for the tutorial",
           "the stream quality is excellent", "hello from India, great stream!", "love this song", "that goal was amazing",
           "who else is watching in 2026?", "please add subtitles", "baddie", "baddie vibes", "she is a baddie",
           "that fit is bad ass", "this slaps", "you are killing it", "that was sick", "this is fire", "no cap this is good",
           "bad take but i respect it", "i'm dead lol", "dying laughing", "that boss fight was brutal", "gg well played",
           "lets goooo", "bro cooked", "absolute legend", "the die roll was unlucky", "this is the way", "chat is on fire",
           "kill the boss now!", "that was a shot to the heart, so emotional", "bad internet today, stream is laggy",
           "stay safe everyone", "love from Kolkata", "music is on point", "mod please pin the schedule"]
    ends = ["", "!", " :)", ", keep it up", " everyone", " - really enjoyed it", ", thanks a lot"]
    clean_ = [p + e for p, e in itertools.product(pos, ends)]
    rng.shuffle(clean_)
    return pd.DataFrame({"text": toxic + clean_, "label": [1] * len(toxic) + [0] * len(clean_)})

def load_data(path):
    path = path or (DEFAULT_DATA if DEFAULT_DATA.exists() else None)
    if path is None:
        return chat_augmentation(), None
    df = pd.read_csv(path)
    if {"tweet", "class"} <= set(df.columns):           # Davidson: 0 hate, 1 offensive, 2 neither
        df = pd.DataFrame({"text": df.tweet, "label": (df["class"] != 2).astype(int)})
    elif "comment_text" in df.columns:                  # Jigsaw
        df = df.rename(columns={"comment_text": "text", "toxic": "label"})
    return df[["text", "label"]].dropna().drop_duplicates("text"), True

def build_pipeline(C=4.0):
    feats = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        # character n-grams catch obfuscation like "b1tch" / "id10t"
        ("char", TfidfVectorizer(preprocessor=clean, analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=3)),
    ])
    return Pipeline([("tfidf", feats), ("clf", LogisticRegression(max_iter=2000, C=C, class_weight="balanced"))])

def train(data=None, quiet=False):
    df, real = load_data(data)
    extra = chat_augmentation() if real else pd.DataFrame(columns=["text", "label"])
    tr, te = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)
    tr, val = train_test_split(tr, test_size=0.15, stratify=tr.label, random_state=1)
    fit = lambda d, C: build_pipeline(C).fit(pd.concat([d, extra]).text, pd.concat([d, extra]).label)

    best = max((1.0, 4.0, 10.0), key=lambda C: f1_score(val.label, fit(tr, C).predict(val.text)))  # tune on validation
    pipe = fit(pd.concat([tr, val]), best)
    prob, pred = pipe.predict_proba(te.text)[:, 1], pipe.predict(te.text)
    rep = classification_report(te.label, pred, target_names=["clean", "toxic"], output_dict=True)
    metrics = {"dataset": "Davidson et al. 2017 + chat augmentation", "train_size": int(len(tr) + len(val) + len(extra)),
               "test_size": int(len(te)), "best_C": best, "accuracy": round(rep["accuracy"], 4),
               "toxic_precision": round(rep["toxic"]["precision"], 4), "toxic_recall": round(rep["toxic"]["recall"], 4),
               "toxic_f1": round(rep["toxic"]["f1-score"], 4), "clean_f1": round(rep["clean"]["f1-score"], 4),
               "roc_auc": round(float(roc_auc_score(te.label, prob)), 4),
               "confusion_matrix [[TN,FP],[FN,TP]]": confusion_matrix(te.label, pred).tolist()}
    if not quiet:
        print(classification_report(te.label, pred, target_names=["clean", "toxic"])); print(json.dumps(metrics, indent=2))
    pipe = fit(df, best)                                 # ship a model refit on all data
    joblib.dump(pipe, MODEL_PATH); METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return pipe

if __name__ == "__main__":
    import train as _t   # pickle references become `train.clean`, not `__main__.clean`
    ap = argparse.ArgumentParser(); ap.add_argument("--data")
    _t.train(ap.parse_args().data)
