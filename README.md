# Amazon Hybrid Product Recommendation System

## Overview
A hybrid recommendation system combining Collaborative Filtering (SVD) and Content-Based Filtering (TF-IDF) built on 2,19,515 Amazon product reviews. Benchmarks three NLP models and deploys a live Streamlit demo.

## Results
| Model | Precision@10 | Recall@10 | Coverage |
|-------|-------------|-----------|----------|
| TF-IDF (Content-Based) | 0.0096 | 0.0252 | 9.41% |
| SVD (Collaborative Filtering) | 0.0180 | 0.0328 | — |
| Sentence Transformers | 0.0123 | 0.0250 | 17.23% |
| SVD + TF-IDF (Hybrid) | 0.0203 | 0.0350 | 37.40% |

SVD RMSE: 1.1410, MAE: 0.8680 — 5.0% improvement over global mean baseline

## Dataset
Amazon Product Reviews — 5,68,427 raw reviews, filtered to 2,19,515 across 23,261 users and 17,538 products. Sparsity: 99.9462%.

Source: https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews

## Project Structure
- notebooks/phase1.ipynb — EDA, cleaning, temporal train/test split
- notebooks/phase2.ipynb — SVD collaborative filtering + baseline comparison
- notebooks/phase3.ipynb — TF-IDF content-based filtering
- notebooks/phase4.ipynb — Hybrid model, alpha tuning
- notebooks/phase5.ipynb — NLP comparison: TF-IDF vs BM25 vs Sentence Transformers
- models/ — saved SVD, TF-IDF vectorizer, hybrid results
- app/app.py — Streamlit demo

## Key Design Decisions
- Temporal train/test split at 80th percentile — prevents data leakage
- 5-review filter on users and products — documents cold-start boundary
- Alpha=0.6 optimal blend — 60% SVD, 40% TF-IDF
- Ablation study across 3 NLP models — ST wins standalone, TF-IDF wins in hybrid

## Stack
Python | Pandas | Scikit-learn | Surprise | Sentence-Transformers | Streamlit | Scipy

## Author
Vishesh | IIT Kharagpur
