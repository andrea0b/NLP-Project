"""Classification model utilities for sentiment analysis."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    classification_report, f1_score, balanced_accuracy_score,
    accuracy_score, confusion_matrix
)




def compare_models(X_train, X_test, y_train, y_test, models=['linear_svc', 'logistic_regression', 'sgd'],
                   X_train_vec=None, X_test_vec=None, X_train_scaled=None, X_test_scaled=None):
    """Train and compare multiple classification models.

    If vectorized/scaled data is provided (from outside), reuse it.
    Otherwise, create new vectorizer and scaler.

    Returns:
        - results_df: Metrics for all models
        - best_model: The trained model with highest macro_f1 score
    """
    results = []
    trained_models = {}

    # If not provided, create vectorizer and scaler
    if X_train_vec is None:
        tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            stop_words='english'
        )
        X_train_vec = tfidf.fit_transform(X_train)
        X_test_vec = tfidf.transform(X_test)

    if X_train_scaled is None:
        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train_vec)
        X_test_scaled = scaler.transform(X_test_vec)

    # Train and evaluate each model
    for model_type in models:
        if model_type == 'linear_svc':
            model = LinearSVC(C=0.1, class_weight='balanced', random_state=42, dual=False, max_iter=5000)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1)
        elif model_type == 'sgd':
            model = SGDClassifier(loss='hinge', class_weight='balanced', random_state=42, max_iter=2000, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        macro_f1 = f1_score(y_test, y_pred, average='macro')
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'macro_f1': macro_f1,
            'weighted_f1': f1_score(y_test, y_pred, average='weighted'),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_negative': f1_score(y_test, y_pred, labels=[-1], average=None)[0],
            'f1_positive': f1_score(y_test, y_pred, labels=[1], average=None)[0],
            'model': model_type
        }
        results.append(metrics)
        trained_models[model_type] = model  # Store all trained models

    results_df = pd.DataFrame(results)

    # Find and return the best model (highest macro_f1)
    best_idx = results_df['macro_f1'].idxmax()
    best_model_name = results_df.loc[best_idx, 'model']
    best_model = trained_models[best_model_name]

    return results_df, best_model
