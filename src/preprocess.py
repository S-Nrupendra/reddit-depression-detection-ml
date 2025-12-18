import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def ensure_nltk_data():
    try:
        nltk.data.find("corpora/stopwords")
        nltk.data.find("corpora/wordnet")
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("stopwords")
        nltk.download("wordnet")
        nltk.download("omw-1.4")



STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def merge_title_body(df):
    df["title"] = df["title"].fillna("")
    df["body"] = df["body"].fillna("")

    df["text"] = df.apply(
        lambda row: row["body"] if row["body"].strip() != "" else row["title"],
        axis=1
    )
    df = df[df["text"].str.strip() != ""]

    return df

def basic_cleaning(df):
    df = df.dropna(subset=["label"])
    df = df.drop_duplicates(subset=["text"])

    # Drop very short texts (< 3 words)
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    df = df[df["word_count"] >= 3]

    df = df.drop(columns=["word_count"])

    return df

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower() # lowercase
    text = re.sub(r"http\S+|www\S+", " ", text) # Remove URLS
    text = re.sub(r"[^a-z\s]", " ", text)       # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()    # Remove extra spaces
    tokens = [word for word in text.split() if word not in STOPWORDS] # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]          # Lemmatization

    return " ".join(tokens)

def apply_cleaning(df):
    df["clean_text"] = df['text'].apply(clean_text)

    df = df[df["clean_text"].str.strip() != ""]

    return df

def preprocess_and_save(
    input_path="../data/raw/reddit_depression_dataset.csv",
    output_path="../data/processed/reddit_clean.csv"
):
    ensure_nltk_data()
    print("Loading dataset...")
    df = pd.read_csv(input_path)

    print("Merging title + body...")
    df = merge_title_body(df)

    print("Performing basic cleaning...")
    df = basic_cleaning(df)

    print("Deep text cleaning...")
    df = apply_cleaning(df)

    print(f"Saving cleaned dataset to: {output_path}")
    df.to_csv(output_path, index=False)

    print("Preprocessing complete.")
    print(f"Final shape: {df.shape}")

    return df