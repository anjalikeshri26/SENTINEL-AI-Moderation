# SENTINEL – Real-time AI Comment Moderation

An NLP service that scores live-stream chat messages for toxicity and holds abusive ones for review before they reach viewers.
Built with a reproducible ML pipeline, real training data and a live demo.

## Results (held-out test set, 4,957 comments never seen in training)

| Metric | Score |
|---|---|
| Accuracy | **95.7%** |
| ROC-AUC | **0.986** |
| Toxic precision / recall / F1 | 0.986 / 0.962 / 0.974 |
| Clean F1 | 0.878 |

Full numbers are written to `metrics.json` on every training run.

## How it works
- **Model:** TF-IDF features (word 1-2-grams + character 2-5-grams) into a class-balanced Logistic Regression (scikit-learn).
  Character n-grams make it robust to obfuscation such as `b1tch`. Regularisation strength is tuned on a validation split.
- **Data:** [Davidson et al. 2017](https://github.com/t-davidson/hate-speech-and-offensive-language) (24,783 labelled tweets, MIT licence; hate + offensive merged into "toxic"),
  plus hand-written live-chat examples (slang such as "baddie", "this slaps", gaming talk) so everyday chat isn't over-flagged.
- **API (Flask):** `POST /predict` `{"text": "..."}` -> `{"toxicity": 0.94, "flagged": true}`; also `POST /predict_batch`, `GET /api/youtube-comments?video=ID`, `GET /model-info` and `GET /health`.
- **Frontend:** a simulated live stream whose chat is moderated in real time, with a strictness slider, live stats, click-to-reveal held messages,
  and an optional real YouTube embed. User text is rendered with `textContent` (no HTML injection).

## Real YouTube comments
The page embeds the original video and pulls that video's actual comments through the YouTube Data API v3, scoring each one with the model.
1. Google Cloud Console -> create a project -> enable **YouTube Data API v3** -> Credentials -> **Create API key** (free tier, 1 quota unit per request).
2. Copy `.env.example` to `.env` and set `YOUTUBE_API_KEY=your_key`.
3. Run `python app.py`; comments load automatically. Paste any other YouTube link to moderate that video instead.
No key? Use **Paste comments** (one per line) to moderate real comments you copy from the video.

## Run
```bash
pip install -r requirements.txt
python app.py            # open http://localhost:5000
python train.py          # optional: retrain on data/labeled_data.csv and refresh metrics.json
pytest                   # tests
```
Env vars: `TOXIC_THRESHOLD` (default 0.5), `PORT`. Other data: `python train.py --data file.csv` (`text,label` or Jigsaw `comment_text,toxic`).

## Known limitations
- The dataset labels casual profanity as "offensive", so swearing is flagged even when not aimed at anyone.
- It is a bag-of-n-grams model: it misses sarcasm, implicit abuse and context (e.g. "nobody wants you here, trash" scores below the cutoff).
- The test set is 83% toxic, so precision on mostly-friendly real chat will be lower than reported.
- Next steps: fine-tune a transformer (e.g. DistilBERT) on Jigsaw; add multilingual / Hinglish support.

## Credits
Dataset: Davidson et al., ICWSM 2017.
