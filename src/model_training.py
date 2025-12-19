import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Logistic Regression
def train_logistic_regression(X_train_tfidf, y_train):
    model = LogisticRegression(
        class_weight="balanced",  # handle class imbalance
        max_iter=1000,
        C=1.0,                    # good default regularization
        solver="liblinear"        # stable for sparse text
    )

    model.fit(X_train_tfidf, y_train)
    return model

# Naive Bayes
def train_naive_bayes(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(results["confusion_matrix"])

    return results

def save_model(model, path):
    joblib.dump(model, path)