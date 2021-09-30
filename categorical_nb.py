## Inherits everything from CategoricalNB and makes a few changes
## Mainly, adds an extra argument: weights. This is required for
## making a prediction on a fitted model, and replaces the class_priors
## that are generated when fitting.

from sklearn.naive_bayes import CategoricalNB
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.fixes import logsumexp
import numpy as np


class CategoricalNBMod(CategoricalNB):
    def _joint_log_likelihood(self, X, weights):
        if not X.shape[1] == self.n_features_:
            raise ValueError(
                "Expected input with %d features, got %d instead"
                % (self.n_features_, X.shape[1])
            )
        # Also check shape of weights
        elif weights is not None:
            if len(weights) != len(self.classes_):
                raise ValueError("Number of priors must match number of" " classes.")
        jll = np.zeros((X.shape[0], self.class_count_.shape[0]))
        for i in range(self.n_features_):
            indices = X[:, i]
            jll += self.feature_log_prob_[i][:, indices].T
        total_ll = jll + np.log(weights)
        return total_ll

    def predict(self, X, weights):
        """
        Perform classification on an array of test vectors X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values for X
        """
        check_is_fitted(self)
        X = self._check_X(X)
        jll = self._joint_log_likelihood(X, weights)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_log_proba(self, X, weights):
        """
        Return log-probability estimates for the test vector X.
        Parameters **UPDATED TO USE PRIORS (I.E. WEIGHTS)**
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        X = self._check_X(X)
        jll = self._joint_log_likelihood(X, weights)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X, weights):
        """
        Return probability estimates for the test vector X.
        Parameters **UPDATED TO USE PRIORS (I.E. WEIGHTS)**
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        return np.exp(self.predict_log_proba(X, weights))
