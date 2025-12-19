import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def load_cleaned_data(path="data/processed/reddit_clean.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["clean_text", "label"])
    df["label"] = df["label"].astype(int)
    return df

def split_data(df, test_size=0.2, random_state=42):
    X = df["clean_text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

# TF-IDF Vectorization
def tfidf_vectorize(
        X_train,
        X_test,
        max_features=20000
):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1,2),
        min_df=5,
        sublinear_tf=True,
        lowercase=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

def save_vectorizer(vectorizer, path="models/tfidf_vectorizer.pkl"):
    joblib.dump(vectorizer, path)