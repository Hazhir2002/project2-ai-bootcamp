Project: Sentiment analysis of product reviews (warranty/guarantee focus)

Contents

- `sentiment-analysis-of-comments.ipynb` - main notebook performing EDA, keyword detection for warranty/guarantee, simple labeling, aggregation, and CSV export.
- `train_data.csv`, `test_data.csv`, `title_brand.csv` - raw data files.
- `GoogleNews-vectors-negative300.bin` - pretrained word2vec binary used to expand keywords.

Quick start

1. Create a virtualenv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the notebook with JupyterLab or Jupyter Notebook and execute cells in order.

Notes and suggested next steps (to meet an NLP deep-learning project standard):

- Data preprocessing: tokenization, stopword removal, lemmatization/stemming, handling out-of-vocab tokens.
- Feature engineering: text vectorization (Word2Vec/fastText/GloVe), TF-IDF, contextual embeddings (BERT/Cohere/transformers).
- Model training: add supervised model training (e.g., fine-tune BERT/DistilBERT, or train an LSTM/CNN) to predict sentiment from `reviewText`.
- Evaluation: train/validation split, confusion matrix, precision/recall/F1, ROC-AUC for binary classification.
- Reproducibility: add code to set random seeds, save trained model weights, and include hyperparameter settings.
- Deployment: export model and a simple inference script or REST API.

If you want, I can implement the model training pipeline (data splits, tokenization, a small transformer fine-tuning loop with `transformers`, evaluation metrics, and model saving).
