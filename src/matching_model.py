# src/matching_model.py
"""
Lightweight supervised matching model interface.
The implementation is deliberately simple and does **not** train a model here –
it only provides the class skeleton that downstream code can use once the user
has prepared training data.

Key methods:
    - fit(X, y): train a classifier (default: sklearn's GradientBoostingClassifier).
    - predict_proba(X): return probability of the positive class.
    - save(path): serialize the trained model to disk (joblib).
    - load(path): class method to load a saved model.

The design keeps dependencies minimal (only scikit‑learn and joblib) and works
with the feature dictionary produced by ``src/matching_features.py``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import joblib
from sklearn.ensemble import GradientBoostingClassifier


class MatchingModel:
    """A thin wrapper around a scikit‑learn classifier for entity matching.

    The model expects feature vectors as ``list[float]`` or ``list[int]``.
    Consumers should convert the ``dict`` returned by ``build_feature_dict``
    (in ``matching_features.py``) to a flat list preserving a consistent order.
    """

    def __init__(self, classifier: Any = None):
        # Use a sensible default if none is supplied.
        self.classifier = classifier or GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )
        self._feature_order: List[str] | None = None

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    def fit(self, X: List[Dict[str, Any]], y: List[int]):
        """Fit the underlying classifier.

        Parameters
        ----------
        X: list of feature dictionaries – each dict must contain the same keys.
        y: list of binary labels (1 = match, 0 = non‑match).
        """
        if not X:
            raise ValueError("Feature list X is empty")
        # Determine a deterministic ordering of feature names.
        self._feature_order = sorted(X[0].keys())
        X_matrix = [[sample.get(k, 0) for k in self._feature_order] for sample in X]
        self.classifier.fit(X_matrix, y)
        return self

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------
    def _ensure_order(self):
        if self._feature_order is None:
            raise RuntimeError("Model has not been fitted – feature order unknown")

    def predict_proba(self, X: List[Dict[str, Any]]) -> List[float]:
        """Return the probability of the positive class for each sample."""
        self._ensure_order()
        X_matrix = [[sample.get(k, 0) for k in self._feature_order] for sample in X]
        # scikit‑learn returns ``[[p0, p1], ...]`` – we need the second column.
        probs = self.classifier.predict_proba(X_matrix)[:, 1]
        return probs.tolist()

    # ---------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialise the model (including feature order) to ``path`` using joblib."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "classifier": self.classifier,
            "feature_order": self._feature_order,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "MatchingModel":
        """Load a model saved with :meth:`save`.

        Returns a new ``MatchingModel`` instance with the restored classifier
        and feature ordering.
        """
        payload = joblib.load(path)
        obj = cls(classifier=payload["classifier"])
        obj._feature_order = payload["feature_order"]
        return obj

    # ---------------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------------
    def set_feature_order(self, order: List[str]):
        """Manually set the feature order – useful when loading a model that was
        trained elsewhere and the ordering must be forced.
        """
        self._feature_order = order
        return self

    def __repr__(self) -> str:
        return f"MatchingModel(classifier={self.classifier.__class__.__name__})"
