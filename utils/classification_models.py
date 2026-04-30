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


class SentimentClassifier:
    """Wrapper for sentiment classification with evaluation metrics."""

    def __init__(self, model_type='linear_svc', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.tfidf = None
        self.scaler = None
        self.model = None

    def build_tfidf(self, X_train, max_features=5000, ngram_range=(1, 3)):
        """Build and fit TF-IDF vectorizer."""
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            stop_words='english'
        )
        X_train_vec = self.tfidf.fit_transform(X_train)
        return X_train_vec

    def transform_tfidf(self, X):
        """Transform text using fitted TF-IDF."""
        return self.tfidf.transform(X)

    def scale_features(self, X_train_vec, X_test_vec=None):
        """Apply standard scaling (needed for SVM)."""
        self.scaler = StandardScaler(with_mean=False)
        X_train_scaled = self.scaler.fit_transform(X_train_vec)
        if X_test_vec is not None:
            X_test_scaled = self.scaler.transform(X_test_vec)
            return X_train_scaled, X_test_scaled
        return X_train_scaled

    def build_model(self):
        """Create the classification model."""
        if self.model_type == 'linear_svc':
            self.model = LinearSVC(
                C=0.1,
                class_weight='balanced',
                random_state=self.random_state,
                dual=False,
                max_iter=5000
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'sgd':
            self.model = SGDClassifier(
                loss='hinge',
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=2000,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        return self.model

    def train(self, X_train_scaled, y_train):
        """Train the model."""
        if self.model is None:
            self.build_model()
        self.model.fit(X_train_scaled, y_train)
        return self

    def predict(self, X_test_scaled):
        """Make predictions."""
        return self.model.predict(X_test_scaled)

    def evaluate(self, y_true, y_pred):
        """Compute evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1_negative': f1_score(y_true, y_pred, labels=[-1], average=None)[0],
            'f1_positive': f1_score(y_true, y_pred, labels=[1], average=None)[0],
        }

    def get_classification_report(self, y_true, y_pred):
        """Get full classification report."""
        return classification_report(
            y_true, y_pred,
            target_names=['Negative (-1)', 'Neutral (0)', 'Positive (1)']
        )

    def get_confusion_matrix(self, y_true, y_pred):
        """Get confusion matrix."""
        return confusion_matrix(y_true, y_pred)


def compare_models(X_train, X_test, y_train, y_test, models=['linear_svc', 'logistic_regression', 'sgd']):
    """Train and compare multiple classification models."""
    results = []

    # Build TF-IDF using training data
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        stop_words='english'
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    # Scale features
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_vec)
    X_test_scaled = scaler.transform(X_test_vec)

    # Train and evaluate each model
    for model_type in models:
        clf = SentimentClassifier(model_type=model_type)
        clf.model = clf.build_model()
        clf.tfidf = tfidf
        clf.scaler = scaler
        clf.train(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        metrics = clf.evaluate(y_test, y_pred)
        metrics['model'] = model_type
        results.append(metrics)

    return pd.DataFrame(results)
