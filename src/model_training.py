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
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced", # Very important for imbalanced data
        n_jobs=-1
    )
    model.fit(X_train, y_train)
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