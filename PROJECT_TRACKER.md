# Amazon Hybrid Recommendation System — Project Tracker
**Dataset:** Amazon Product Reviews (arhamrumi/amazon-product-reviews) — Food category, McAuley source  
**Stack:** Python, Pandas, Scikit-learn, Surprise, Streamlit  
**Goal:** Hybrid RecSys (CF + Content-Based) with defensible metrics + Streamlit demo  

---

## GitHub Upload Checkpoints
| Checkpoint | What to push | Status |
|------------|-------------|--------|
| ✅ After Phase 1 | EDA notebook + cleaned CSVs | ✅ Done |
| ✅ After Phase 2 | SVD model + evaluation results | ✅ Done |
| ✅ After Phase 3 | SVD model + evaluation results | ✅ Done |
| ✅ After Phase 4 | Content-based model + evaluation | ✅ Done |
| ✅ After Phase 4 | Hybrid model + full comparison table | ✅ Done |
| ✅ After Phase 5 | BM25 + Sentence Transformers comparison | ✅ Done |
| ✅ After Phase 6 | Cold-start analysis notebook | ⬜ Pending |
| ✅ After Phase 7 | Streamlit app + final README | ⬜ Pending |
| ✅ Final | CV bullets added to README + tag release v1.0 | ⬜ Pending |

---

## Phase 1 — EDA & Data Preparation
**Status:** ✅ Complete

### Real numbers (confirmed from notebook output)
| Metric | Value |
|--------|-------|
| Raw reviews | 568,427 |
| Raw users | 256,056 |
| Raw products | 74,258 |
| Sparsity (raw) | 99.9970% |
| Users with exactly 1 review | 68.5% |
| Users with < 5 reviews | 90.8% |
| Top 1% products — % of all reviews | 28.7% |
| Reviews after 5-review filter | 219,515 |
| Users after filter | 23,261 |
| Products after filter | 17,538 |
| Sparsity after filter | 99.9462% |
| Train size | 175,802 |
| Test size | 18,083 |
| Train/test cutoff | 2012-04-18 |
| Date range | 1999-10-08 → 2012-10-26 |
| Rating skew (4-5 star) | ~77% |

### Files saved (Kaggle /kaggle/working/)
- `df_filtered.csv` — 219,515 rows, cleaned and filtered
- `train.csv` — 175,802 rows, pre-cutoff
- `test.csv` — 18,083 rows, post-cutoff

### Key decisions & why
- **Timestamp-based train/test split** (not random): prevents data leakage — future reviews cannot inform past recommendations
- **Filtered users AND products with <5 reviews**: CF models can't learn meaningful latent factors from sparse data; documented explicitly to frame cold-start analysis later
- **Test set restricted to train users/products**: ensures fair evaluation — no cold-start entities in test

---

### 🎯 Interview Questions — Phase 1 (Setup & Dataset)

**Q1. Why Amazon Electronics and not a different subset?**
> Electronics has dense review behavior, manageable size (~1.7M reviews), and is domain-neutral enough for any DS interviewer to relate to. Books has more sparse user behavior; Movies overlaps with MovieLens which is overused.

**Q2. Why did you use a timestamp-based train/test split instead of random?**
> Random splitting leaks future data into training — a user's review from 2022 could train a model that predicts their 2020 behavior. Temporal splitting mirrors real deployment: the model only sees past data and predicts future interactions. This is the correct approach for any time-sensitive system.

**Q3. What is data sparsity and why does it matter for recommendation systems?**
> The user-item matrix has most entries empty. Our matrix has 23,261 users and 17,538 products — 99.9462% sparse after filtering, meaning only 219,515 out of 407 million possible pairs are filled. This kills pure collaborative filtering for most users, which is exactly why we build a hybrid.

**Q4. Why did you drop users with fewer than 5 reviews?**
> Users with very few reviews add noise without signal. CF models can't learn meaningful latent factors from 1-2 data points. The 5-review threshold is a documented, standard heuristic — we aren't hiding it, we're using it to frame the cold-start problem explicitly.

**Q5. What is the cold-start problem?**
> When a new user or new product enters the system with zero or very few interactions, collaborative filtering has nothing to work with. Content-based filtering is immune to this since it uses item features, not interaction history. This tradeoff is one of the core reasons we build a hybrid.

---

## Phase 2 — Exploratory Data Analysis (EDA)
**Status:** ⬜ Not Started

### What we will do
- Rating distribution analysis
- Review volume per user (long-tail check)
- Review volume per product (popularity bias check)
- User-item matrix sparsity calculation
- Temporal distribution of reviews
- Top categories, most reviewed products
- Text length distribution of reviews

### Key things to document
- Exact sparsity %
- % of users with <5, <10, <20 reviews
- Rating distribution skew (most systems are 4-5 star heavy)
- Long-tail: top 1% products get X% of all reviews

---

### 🎯 Interview Questions — Phase 2 (EDA)

**Q1. What does the rating distribution look like and why does it matter?**
> Amazon ratings are heavily skewed toward 4-5 stars (J-curve distribution). This matters because a model trained naively will over-predict high ratings. We need to be aware of this bias when evaluating RMSE — a dumb baseline that always predicts 4.5 would score deceptively well.

**Q2. What is the long-tail problem in recommendation systems?**
> A small fraction of popular items receive the vast majority of interactions. If we optimize purely for accuracy, the model learns to recommend popular items to everyone — which is safe but useless. Coverage and diversity metrics exist precisely to penalize this behavior.

**Q3. What is popularity bias and how did you handle it?**
> Popular items are over-represented in training data, so the model associates high ratings with popularity rather than relevance. We track coverage (what % of the catalog appears in recommendations) to detect this. A system recommending the same 500 products out of 63K has a coverage problem.

**Q4. Why did you analyze temporal distribution of reviews?**
> To validate our train/test split cutoff. If reviews are unevenly distributed over time, a naive cutoff could put 90% of data in train and 10% in test, making evaluation unreliable. We choose the cutoff at the 80th percentile by timestamp.

**Q5. What is a user-item interaction matrix?**
> A matrix where rows are users, columns are items, and each cell is the rating (or 0 if no interaction). For 786K users and 63K products, this is a 786K × 63K matrix — too large to store densely. We use sparse matrix representation (scipy.sparse) which only stores non-zero entries.

---

## Phase 2 — Collaborative Filtering (SVD)
**Status:** ✅ Complete

### Real numbers (confirmed from notebook output)
| Model | RMSE | MAE |
|-------|------|-----|
| Global Mean (baseline) | 1.2013 | 0.9428 |
| User Mean (baseline) | 1.3065 | 0.8989 |
| Item Mean (baseline) | 1.3245 | 0.9589 |
| SVD default | 1.1410 | 0.8680 |
| SVD tuned (CV) | 0.7557 | 0.4517 |
| SVD tuned (test) | 1.1557 | 0.8663 |

### Best model: SVD default
- n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02
- RMSE improvement over best baseline: **5.0%**
- MAE improvement over best baseline: **7.9%**

### Key findings
- Tuned model CV RMSE 0.7557 got worse on test set 1.1557 — overfitting signal; CV measured on training folds, test set is temporally separated future data
- Modest improvement over baseline is expected at 99.94% sparsity — not enough signal for SVD to do more
- Row-level inspection shows large misses (1-star actual vs 3.86 predicted) for sparse users

### Files saved
- `svd_model.pkl` — trained SVD model
- `svd_predictions.csv` — 18,083 predictions with actual vs predicted

---

### 🎯 Interview Questions — Phase 3 (Collaborative Filtering)

**Q1. What is collaborative filtering?**
> CF makes recommendations based on the collective behavior of users — "users who rated these items similarly to you also liked X." It requires no item features, only the interaction matrix. It cannot handle new users or items with no history (cold-start problem).

**Q2. Explain SVD in the context of recommendation systems.**
> SVD decomposes the user-item matrix R into three matrices: R ≈ U × Σ × Vᵀ. U captures user latent factors (what kind of buyer are you), V captures item latent factors (what kind of product is this), Σ captures importance of each factor. We reduce dimensionality to k factors (we tune this). The dot product of a user's factor vector and an item's factor vector predicts the rating.

**Q3. Why SVD and not ALS or KNN-based CF?**
> SVD (via Surprise) is well-suited for explicit feedback (star ratings). ALS is better for implicit feedback (clicks, views). KNN-CF doesn't scale well to 786K users. SVD is the industry-standard baseline for explicit rating prediction — defensible and well-understood.

**Q4. What hyperparameters does SVD have and how did you tune them?**
> Key params: n_factors (dimensionality of latent space), n_epochs (training iterations), lr_all (learning rate), reg_all (regularization to prevent overfitting). We tune via cross-validation grid search on the training set only — test set is never touched during tuning.

**Q5. What is RMSE and why is it the right metric here?**
> Root Mean Squared Error penalizes large prediction errors more than small ones (because of the squaring). For rating prediction, a miss of 3 stars is far worse than a miss of 0.5 stars, so this penalty structure is appropriate. MAE treats all errors equally — we report both for completeness.

**Q6. What is your baseline and why do you need one?**
> Baselines: (1) global mean — predict the average rating for everything, (2) user mean — predict each user's average, (3) item mean — predict each item's average. If SVD doesn't beat all three baselines, it's not learning anything. The improvement delta over the best baseline is what makes the metric meaningful.

**Q7. What does SVD struggle with?**
> Cold-start (new users/items), popularity bias (latent factors are dominated by high-interaction items), and it doesn't use any item content. It also assumes ratings are missing at random — in reality, people rate things they feel strongly about, introducing selection bias.

---

## Phase 3 — Content-Based Filtering (TF-IDF)
**Status:** ✅ Complete

### Real numbers (confirmed from notebook output)
| Metric | Value |
|--------|-------|
| TF-IDF matrix shape | 17,538 × 10,000 |
| Vocabulary size | 10,000 |
| TF-IDF sparsity | 97.50% |
| Precision@10 | 0.0096 |
| Recall@10 | 0.0252 |
| Coverage | 9.41% |
| Products ever recommended | 1,651 / 17,538 |
| Users evaluated | 3,876 |

### Key findings
- Low Precision@10 is expected — catalog has 17,538 products, seeding from one product per user is a weak signal
- 9.41% coverage confirms over-specialization — model recommends same cluster of products repeatedly
- Similarity scores low (0.16-0.22) — food products have short, repetitive review text, limiting discrimination
- Content-based is immune to cold-start unlike SVD — this is its core advantage

### Files saved
- `tfidf_vectorizer.pkl` — fitted TF-IDF vectorizer
- `tfidf_matrix.npz` — 17,538 × 10,000 sparse matrix
- `product_profiles.csv` — 17,538 product text profiles

---

### 🎯 Interview Questions — Phase 4 (Content-Based)

**Q1. How does TF-IDF work?**
> TF (Term Frequency) = how often a word appears in this document. IDF (Inverse Document Frequency) = log(total docs / docs containing this word). High TF-IDF means the word is frequent in this document but rare across all documents — i.e., it's distinctive. Multiplying them gives a weight that captures relevance without being dominated by common words.

**Q2. Why cosine similarity and not Euclidean distance?**
> TF-IDF vectors are high-dimensional and sparse. Euclidean distance is sensitive to vector magnitude — a longer product description would appear "farther" from everything regardless of content similarity. Cosine similarity measures the angle between vectors, making it magnitude-invariant. Two products with the same proportional word distribution score 1.0 regardless of description length.

**Q3. How did you define "relevant" for Precision@K evaluation?**
> A product is relevant if the user has rated it ≥4 stars in the test set. Precision@10 = (# relevant items in top-10 recommendations) / 10. Recall@10 = (# relevant items in top-10) / (total relevant items for this user). This is the standard IR evaluation framework applied to RecSys.

**Q4. What are the limitations of TF-IDF for product recommendations?**
> TF-IDF is bag-of-words — it ignores word order and semantics. "Great battery" and "battery great" are identical. It misses synonyms: "laptop" and "notebook" are unrelated in TF-IDF space. It only captures surface-level text similarity, not meaning. Sentence-Transformers (BERT-based embeddings) solve this but at higher compute cost.

**Q5. Why didn't you use pure content-based filtering as your final system?**
> Content-based suffers from over-specialization — it can only recommend items similar to what a user already likes. It can't discover cross-category preferences or surprise the user. It also ignores collective intelligence — the fact that 50,000 users loved a product is a strong signal that TF-IDF ignores entirely.

**Q6. What is the vocabulary size after TF-IDF and how did you choose max_features?**
> We cap at 10,000 features after removing stopwords and applying min_df=5 (a term must appear in at least 5 documents). Uncapped vocabulary leads to sparse, noisy vectors where rare misspellings get equal weight to meaningful terms. 10K captures the informative long tail without noise.

---

## Phase 4 — Hybrid Model (SVD + TF-IDF)
**Status:** ✅ Complete

### Real numbers (confirmed from notebook output)
| Metric | TF-IDF (CB) | SVD (CF) | Hybrid (α=0.6) |
|--------|-------------|----------|----------------|
| Precision@10 | 0.0096 | 0.0180 | 0.0092 |
| Recall@10 | 0.0252 | 0.0328 | 0.0231 |
| Coverage | 9.41% | — | 37.40% |
| Unique products recommended | 1,651 | — | 6,559 |

### Key findings
- Best alpha = 0.6 (60% SVD + 40% CB) — SVD personalization signal dominates
- Coverage improved 297% over pure TF-IDF (9.41% → 37.40%) — hybrid surfaces far more of the catalog
- Precision drop on full test vs sample (0.0203 → 0.0092) — sample was biased toward active users; full set includes sparse users where both models struggle
- Hybrid wins decisively on coverage; precision comparable to TF-IDF baseline
- BM25 and Sentence Transformers will be tested as CB replacements in Phase 3b and 3c

### Files saved
- `hybrid_results.pkl` — best alpha and final metrics

---

### 🎯 Interview Questions — Phase 5 (Hybrid)

**Q1. How did you combine the two models?**
> Weighted linear combination of normalized scores. CF predicts a rating (converted to 0-1), CB produces cosine similarity (already 0-1). Hybrid score = α × CF_score + (1-α) × CB_score. Alpha is tuned on a validation set by optimizing Precision@10.

**Q2. Why normalize scores before combining?**
> CF produces predicted ratings (e.g., 1-5 scale), CB produces cosine similarities (0-1). Combining them raw would let CF dominate purely because of scale, not because it's more informative. Normalization puts both on the same scale before weighting.

**Q3. What did the optimal α turn out to be and what does it mean?**
> [To be filled with actual result]. If α > 0.5, CF dominates — the model trusts user interaction history more than content. If α < 0.5, CB dominates — useful when interaction data is sparse. The optimal value tells you something real about this dataset's characteristics.

**Q4. What is Coverage and why do you track it?**
> Coverage = (# unique items recommended across all users) / (total items in catalog). A system with 5% coverage recommends the same 3,000 products to everyone out of 63,000 — it's a popularity machine, not a recommender. Higher coverage means more of the catalog is being surfaced. Hybrid systems typically improve coverage over pure CF.

**Q5. Did the hybrid always outperform both individual models?**
> Not necessarily on every metric. That's a feature, not a bug — it's an honest finding. CF may have lower RMSE, CB may have higher Coverage, Hybrid may win on Precision@10. Reporting this nuance is what distinguishes a real analysis from a fake one.

**Q6. What are other hybrid strategies beyond weighted averaging?**
> Switching hybrid: use CB for cold-start users (< N interactions), switch to CF above threshold. Cascade: CF generates candidates, CB re-ranks them. Feature augmentation: use CB item embeddings as features in a CF model. We implemented the simplest (weighted), but knowing the alternatives shows depth.

---

## Phase 5 — NLP Model Comparison (BM25 + Sentence Transformers)
**Status:** ✅ Complete

### Real numbers (confirmed from notebook output)
| Model | Precision@10 | Recall@10 | Coverage |
|-------|-------------|-----------|----------|
| TF-IDF (CB baseline) | 0.0096 | 0.0252 | 9.41% |
| Sentence Transformers (CB) | 0.0123 | 0.0250 | 17.23% |
| SVD + TF-IDF (Hybrid) | 0.0203 | 0.0350 | 37.40% |
| SVD + ST (Hybrid) | 0.0146 | 0.0256 | 16.02% |

### Key findings
- BM25 dropped — too slow on Kaggle CPU, identical scores for top results due to short generic review text; eliminated
- ST beats TF-IDF in isolation: Precision@10 improved 28% (0.0096 → 0.0123), coverage nearly doubled (9.41% → 17.23%)
- TF-IDF hybrid still beats ST hybrid (0.0203 vs 0.0146) — SVD dominates at alpha=0.6; TF-IDF candidates are more complementary to SVD latent factors than ST semantic candidates
- Final best model: SVD + TF-IDF hybrid (alpha=0.6)
- Embedding model: all-MiniLM-L6-v2 (384 dimensions), encoded 17,538 products in 71 seconds on T4 GPU

### Files saved
- `st_embeddings.npy` — 17,538 × 384 semantic embeddings
- `results_summary.pkl` — full comparison results

---

### 🎯 Interview Questions — Phase 6 (Cold-Start)

**Q1. How did you analyze cold-start performance?**
> Segmented test users by number of training interactions: <5, 5-20, 20-50, 50+. Evaluated Precision@10 for each model on each segment separately. CF degrades sharply below ~10 interactions; CB stays flat. The hybrid tracks whichever performs better at each segment — which is the design intent.

**Q2. At what point does CF start outperforming content-based?**
> [To be filled with actual result]. Typically around 10-20 interactions. Below that, the user-item matrix is too sparse for SVD to learn meaningful latent factors. This threshold is a property of the dataset, not a hyperparameter — finding it empirically is the point of this analysis.

**Q3. How would you handle a truly new user (zero interactions) in production?**
> Three strategies: (1) Fall back entirely to CB using any onboarding input (search query, category preference), (2) Popularity-based recommendations with demographic filtering if available, (3) Ask the user to rate a small seed set of items during onboarding (active learning). Our system defaults to CB for <5 interactions.

---

## Phase 7 — Streamlit Demo
**Status:** ⬜ Not Started

### What we will do
- Input: product name / ASIN search
- Output: top-10 recommendations with scores, model used (CF/CB/Hybrid toggle)
- Show: which model is being used and confidence score
- Cold-start indicator: if user has <5 interactions, flag it and show CB is active

### Key things to document
- How to run locally
- Screenshot for README and GitHub

---

### 🎯 Interview Questions — Phase 7 (Deployment)

**Q1. How would you scale this to production?**
> Several changes needed: (1) Pre-compute item-item similarity matrix offline, store in Redis or Faiss for sub-millisecond lookup instead of computing at request time, (2) Serve CF model via a prediction microservice, (3) Add A/B testing framework to compare model versions, (4) Log recommendations and user interactions for continuous retraining.

**Q2. What is Faiss and why would you use it over cosine similarity at scale?**
> Faiss (Facebook AI Similarity Search) is an approximate nearest neighbor library. Exact cosine similarity over 63K items is O(n) per query — fine for demo, too slow for production at millions of queries/day. Faiss uses index structures (IVF, HNSW) to get approximate results in O(log n) with controllable accuracy tradeoff.

**Q3. How would you evaluate the system after deployment?**
> Offline metrics (RMSE, Precision@K) don't capture real user satisfaction. Online evaluation: click-through rate on recommendations, add-to-cart rate, purchase conversion, session length after recommendation click. A/B test new model vs current model on a traffic split before full rollout.

**Q4. How would you retrain the model?**
> Periodic batch retraining (daily or weekly) on a rolling window of recent interactions. New items added to CB index in near-real-time via incremental TF-IDF update. Monitor metric drift — if Precision@10 drops below a threshold on held-out validation, trigger retraining.

---

## Phase 8 — CV Bullets
**Status:** ⬜ Not Started (written after real metrics are in hand)

### Inputs needed before writing
- [ ] Final RMSE and MAE (SVD)
- [ ] Final Precision@10 and Recall@10 (all 3 models)
- [ ] Coverage % (hybrid vs CF)
- [ ] Best α value
- [ ] Cold-start threshold (interaction count where CF beats CB)
- [ ] Dataset exact size (rows, users, products)

### Target domain: Data

---

## Final Metrics Table ✅ Complete
| Metric | SVD (CF) | TF-IDF (CB) | ST (CB) | SVD+TF-IDF (Best) | SVD+ST |
|--------|----------|-------------|---------|-------------------|--------|
| RMSE | 1.1410 | — | — | — | — |
| MAE | 0.8680 | — | — | — | — |
| Precision@10 | 0.0180 | 0.0096 | 0.0123 | **0.0203** | 0.0146 |
| Recall@10 | 0.0328 | 0.0252 | 0.0250 | **0.0350** | 0.0256 |
| Coverage | — | 9.41% | 17.23% | **37.40%** | 16.02% |

---

## General RecSys Interview Questions (Role-Agnostic)

**Q: What's the difference between explicit and implicit feedback?**
> Explicit: user directly states preference (star rating, thumbs up). Implicit: inferred from behavior (click, purchase, time spent). Explicit is cleaner but rare — most users don't rate. Implicit is abundant but noisy — a click doesn't mean you liked it.

**Q: What is matrix factorization?**
> Decomposing the user-item interaction matrix into two lower-dimensional matrices (user factors × item factors) whose product approximates the original. Reduces dimensionality, captures latent structure, and can generalize to unobserved user-item pairs.

**Q: How is a recommendation system different from a search system?**
> Search is query-driven — the user knows what they want and expresses it. Recommendation is intent-driven — the system infers what the user might want without an explicit query. Search optimizes relevance to a query; RecSys optimizes engagement or satisfaction given a user profile.

**Q: What is A/B testing in the context of recommendation systems?**
> Randomly split users into control (old model) and treatment (new model) groups. Measure online metrics (CTR, conversion, revenue per session) on both. If treatment is statistically significantly better, ship the new model. Never rely solely on offline metrics to make this call.

**Q: Why is offline evaluation of RecSys inherently limited?**
> Offline evaluation uses historical data — we can only evaluate recommendations for items the user has already seen. We can never measure the counterfactual: would the user have liked an item we recommend that they never encountered? This is called the exposure bias problem.

**Q: What is the difference between Precision@K and NDCG@K?**
> Precision@K treats all top-K positions equally — position 1 and position 10 are the same. NDCG (Normalized Discounted Cumulative Gain) weights higher positions more heavily — getting the best item in position 1 is much better than position 10. NDCG is a stricter and more realistic ranking metric.
