# Solution | NLP Automated Customer Reviews

**Team:** Hamza Shabbir & Yasaman Najafi Jozani · Ironhack AI Engineering, Project 4

**Live app:** https://project-brief-nlp-automated-customers-reviews-sbmev8fypayseged.streamlit.app/
**Our model on Hugging Face:** https://huggingface.co/hamza-shabbir94/amazon-review-sentiment-analysis

From 34,660 Amazon reviews (Kaggle `1429_1.csv`) we built: a sentiment classifier, 5 product clusters, an AI-written buyer's guide per cluster, and a web app.

---

## Results at a glance

| Task | What we did | Result |
|---|---|---|
| 1. Sentiment | 5 models compared on the same test split | Best macro-F1 **0.605** (TF-IDF v2 + LogReg) |
| 2. Clustering | MiniLM embeddings of product name + tags, KMeans | **5 clusters** for 40 products |
| 3. Summaries | Facts sheet (pandas) → Qwen2.5-1.5B-Instruct | **5 articles** in `outputs/articles/` |
| 4. Deployment | Streamlit app with model picker | Live on Streamlit Community Cloud |

---

## Project structure

```
├── src/
│   ├── config.py              # all paths and column names (import, never hardcode)
│   └── preprocessing.py       # load, clean, label functions
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb   # raw CSV -> data/processed/
│   ├── 02_sentiment_analysis.ipynb      # Task 1 -> models/tfidf-v2/
│   ├── 03_product_clustering.ipynb      # Task 2 -> reviews with a cluster
│   └── 04_review_summarization.ipynb    # Task 3 -> outputs/articles/
├── streamlit_app/             # Task 4: app.py + requirements.txt
├── models/tfidf-v2/           # our model folder (gitignored, uploaded to Hugging Face)
├── outputs/
│   ├── metrics/sentiment_model_comparison.csv
│   ├── figures/               # clusters_pca.png, clustering_choose_k.png
│   └── articles/              # cluster_0.md ... cluster_4.md
└── data/raw/1429_1.csv        # not in Git, download from Kaggle
```

Each notebook saves a file that the next one reads, so run them in order: 01 → 02 → 03 → 04.

---

## 0. Data cleaning (notebook 01)

- `src/preprocessing.py`: dropped rows missing rating, text, name or title, and duplicate review texts: **34,660 → 27,863 reviews**.
- Corrupted product names (two products glued into one `name`) are cleaned in notebook 03; the clean name is the product key, not `id`.
- Labels from stars: **1–2 negative, 3 neutral, 4–5 positive**.
- Strong imbalance: **~93% positive**, so we judge models by **macro-F1**, not accuracy.

## 1. Sentiment analysis (notebook 02)

One stratified split (70/30, `random_state=42`) for every model, 8,359 test reviews.

| Model | Accuracy | Macro-F1 | F1 neg | F1 neu | F1 pos |
|---|---|---|---|---|---|
| **TF-IDF v2 + LogReg** (ours) | 0.925 | **0.605** | 0.495 | 0.355 | 0.966 |
| RoBERTa zero-shot (pretrained) | 0.929 | 0.562 | **0.559** | 0.155 | 0.970 |
| TF-IDF v1 + LogReg | 0.870 | 0.520 | 0.354 | 0.271 | 0.935 |
| TF-IDF v1 + LinearSVC | 0.923 | 0.509 | 0.291 | 0.271 | 0.963 |
| TF-IDF v1 + Naive Bayes | 0.780 | 0.456 | 0.275 | 0.214 | 0.880 |

**What made v2 better than v1:** keep negations ("not", "no", "too"), expand "n't" → "not", use title + text, bigrams, `class_weight="balanced"`, `C=10`. Cleaning lives inside the sklearn `Pipeline`, so the app gets raw text.

**RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`) is used without training. Lower macro-F1, but best on negative reviews and right on all our hand-written test sentences.

**Output:** `models/tfidf-v2/` with `sentiment_model.joblib`, `text_cleaning.py`, `requirements.txt` and a `README.md` model card. Saved locally only; we upload it to Hugging Face ourselves.

## 2. Product clustering (notebook 03)

- First try (average word vectors of all reviews per product) failed: every product looked the same ("great, love it").
- Final: one text per product = **clean name + category tags** → `all-MiniLM-L6-v2` → **KMeans, k = 5** (chosen from 4–6 by silhouette, elbow and cluster sizes).

| Cluster | Content | Products | Reviews |
|---|---|---|---|
| 0 | Kindle e-readers | 4 | 3,795 |
| 1 | Echo, smart home, chargers | 12 | 6,555 |
| 2 | Kindle Oasis & covers | 6 | 293 |
| 3 | Fire 7" & older tablets | 11 | 1,590 |
| 4 | Fire HD 8 & Kids tablets | 7 | 15,630 |

Silhouette ≈ 0.2 (groups overlap), so every cluster was checked by hand. PCA plot: `outputs/figures/clusters_pca.png`.

## 3. AI summaries (notebook 04)

**Pandas finds the facts, the LLM only writes.**

1. Per cluster, a facts sheet: top 3 products (avg rating, #reviews, % negative), weakest product, real praise/complaint quotes. Products with too few reviews are not ranked.
2. `Qwen/Qwen2.5-1.5B-Instruct` writes the article. 3 prompts tested (basic / structured / structured + example); **v2 structured** won.
3. Each `cluster_N.md` ends with the facts sheet it was built from, so every number can be checked.

**Known issues:** articles are ~950–1,250 words (target 300) and some stop mid-sentence; some quotes land under the wrong product; one article adds generic praise not in the facts.

## 4. Deployment (Streamlit)

- `streamlit_app/app.py`: pick a model, paste a review, get 😞 / 😐 / 😊 with 3 confidence bars.
- **Models are not in GitHub.** On first start the app downloads them from Hugging Face and caches them (`@st.cache_resource`):
  - RoBERTa: `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - Ours: `hamza-shabbir94/amazon-review-sentiment-analysis` (`sentiment_model.joblib` + `text_cleaning.py`)
- Hosting: GitHub → Streamlit Community Cloud, main file `streamlit_app/app.py`. CPU-only torch keeps it small.

---

## How to run

```bash
# 1. environment (Python 3.12/3.13)
pip install -r requirements.txt

# 2. data: put Kaggle 1429_1.csv in data/raw/

# 3. notebooks in order
jupyter lab   # run 01 -> 02 -> 03 -> 04

# 4. app
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py

# 5. (optional) publish our model
hf auth login
hf upload hamza-shabbir94/amazon-review-sentiment-analysis models/tfidf-v2 .
```

> scikit-learn must be the same version (1.7.2) when training and loading the `.joblib` model.

---

## Lessons learned

- **Test with your own sentences, not only metrics.** Our first model scored well but called "Worst customer service I have ever experienced" positive.
- **Check what cleaning removes.** Default stopwords deleted "not", "no", "too".
- **Describe a product by what it is, not how people feel**, when clustering.
- **Give the LLM facts, not raw reviews**, and still review its output.
- One shared Python environment from day one.

## Limitations & next steps

- Mixed 3-star (neutral) reviews are still hard for every model (best F1 0.36).
- Only Amazon devices; unclear how it works on other products.
- Next: fine-tune a transformer on our data, cap article length, name the clusters inside notebook 03.
