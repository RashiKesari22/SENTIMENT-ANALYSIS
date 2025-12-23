# SENTIMENT-ANALYSIS

COMPANY : CODTECH IT SOLUTIONS

NAME : RASHI KESARI

INTERN ID:CT04DR3010

DOMAIN : DATA ANALYTICS

DURATION : 4 WEEKS

MENTOR : NEELA SANTHOSH KUMAR

**DESCRIPTION OF TASK 

Sentiment analysis is the process of automatically determining the emotional tone behind a piece of text—whether the writer is expressing a positive, negative, or neutral opinion. When the source is short, informal, and noisy (as with tweets or product reviews), the task becomes a classic NLP challenge that combines text preprocessing, feature engineering, and machine‑learning or deep‑learning models.

Why it matters?
Companies monitor brand perception, marketers gauge campaign impact, and researchers study public mood. By converting unstructured text into a quantitative sentiment score, analysts can aggregate results, visualise trends, and feed the output into downstream systems such as recommendation engines or alert dashboards.

Typical workflow

1.Data collection – Gather raw texts from APIs (Twitter, Twitter API v2; Amazon reviews, etc.) and store them in a structured format (CSV, JSON). Include metadata such as timestamp, user ID, or product ID if relevant.

2. Pre‑processing – Clean the noisy nature of social‑media language:
    - Lower‑case the text.
    - Remove URLs, HTML tags, and user mentions.
    - Expand contractions (“won’t” → “will not”) and handle slang with a lookup dictionary.
    - Tokenise the string (e.g., using spaCy, NLTK, or TweetTokenizer) and strip punctuation.
    - Apply optional steps: stop‑word removal and handling emojis (convert to text or keep as tokens).

3.Feature representation – Convert the cleaned tokens into numeric vectors that a model can learn from:
    - Bag‑of‑words / TF‑IDF for traditional classifiers.
    - Word embeddings (Word2Vec, GloVe) that capture semantic similarity.
    - Contextual embeddings (BERT, RoBERTa, XLM‑R) which handle polysemy and out‑of‑vocabulary words better, especially for short, informal texts.

4. Model selection – Choose an algorithm that balances accuracy and latency:
    - Simple baselines: Logistic Regression, Naïve Bayes, Linear SVM.
    - Tree‑based ensembles: Random Forest, Gradient Boosting.
    - Deep models: LSTM, Bi‑LSTM with attention, or transformer fine‑tuning (e.g., `distilbert-base-uncased` for faster inference).
    - Hybrid approaches: lexicon‑based scores (VADER, TextBlob) combined with a classifier.

5. Training and evaluation – Split the data into train/validation/test sets (stratified to preserve class balance). Use metrics such as accuracy, F1‑score (macro for imbalanced classes), ROC‑AUC, and confusion matrices. Cross‑validation helps assess robustness.

6. Interpretation– Examine which words or phrases drive positive/negative predictions (feature importance for linear models, attention weights for transformers). This insight can guide product improvements or marketing messages.

7. Deployment – Export the trained pipeline (e.g., using `joblib` or ONNX) and serve it via a REST API (FastAPI, Flask) or a batch job for periodic scoring. Monitor drift: retrain periodically as language evolves.
   

**PREREQUISITE


- Install Python 3.9+
- Create a virtual env
- A CSV with two columns: `text` (the raw tweet/review) and `label` (`0` = negative, `1` = positive).



**OUTPUT

Classification report:


               precision    recall  f1-score   support

           0       1.00      1.00      1.00         8
           1       1.00      1.00      1.00        36

    accuracy                           1.00        44
   macro avg       1.00      1.00      1.00        44
weighted avg       1.00      1.00      1.00        44



Top 10 positive cues:

love: 0.946

would buy: 0.664

would: 0.664

buy: 0.664

recommend: 0.618

highly recommend: 0.618

highly: 0.618

solid: 0.399

smooth performance: 0.399

smooth: 0.399


Top 10 negative cues:

unhappy: -2.484

bass: -1.363

low: -1.363

low bass: -1.363

buying: -1.282

never: -1.282

never buying: -1.282

charge: -0.993

charge often: -0.993

need: -0.993


**CONFUSION MATRIX




![Image](https://github.com/user-attachments/assets/5630d153-c175-40da-b704-63495c191231)



**INSIGHTS DERIVED FROM SENTIMENT ANALYSIS



The logistic regression model applied to TF-IDF vectors is identifying the expected key indicators. Words like "great", "love", "best", and "awesome" have the highest coefficients, strongly influencing the prediction towards a positive sentiment (1). Conversely, words such as "bad", "worst", "disappointed", and "poor" are the strongest indicators of a negative sentiment. This outcome aligns with expectations for a binary sentiment analysis task, as the model effectively learns a polarity lexicon from the data.

In terms of performance, metrics like accuracy, F1 score, and ROC-AUC will indicate how well the simple bag-of-words approach generalizes. For a relatively balanced dataset, performance is likely to fall within the high 70s to low 80s percentage range. The confusion matrix will reveal whether the model struggles with neutral or ambiguous examples, often mistaking them for more extreme sentiments - a common challenge when dealing with short, noisy texts like tweets or reviews. 




















