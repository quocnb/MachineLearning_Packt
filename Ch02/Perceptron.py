import numpy as np

class Perceptron(object):

    """Perceptron classifier.
        Parameters
        ------------
        :param eta: float, from 0.0 to 1.0
            Learning rate
        :param n_iter: int
            Passes over the training dataset.
        w_ : {1d-array}
            Weights after fitting.
        errors_ : list
            Number of misclassifications in every epoch.
    """
    eta = 0.01
    n_iter = 10
    _w = []
    error = []

    def __init__(self, eta=0.01, n_iter=10):
        """
        Perceptron classifier init
        :param eta: float, from 0.0 to 1.0
            Learning rate
        :param n_iter: int
            Passes over the training dataset
        """
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Fit training data
        :param X: {array-like}, shape = [m_samples, n_features]
        Training vectors
        :param y: {array-like}, shape = [n_samples]
        :return: self object
        """
        self._w = np.zeros(1 + X.shape[1])
        self.error = []
        for _ in range(self.n_iter):
            errors = 0
            for (xi, yi) in zip(X, y):
                update = self.eta * (yi - self.predict(xi))
                self._w[1:] += update * xi
                self._w[0] += update
                errors += int(update != 0.0)
            self.error.append(errors)
        return self

    def predict(self, X):
        """
        Predict results (label) of X
        :param X: {array-like}
        :return: {1 or -1 array-like}
        """
        return np.where(self.net_input(X) >= 0, 1, -1)

    def net_input(self, X):
        """
        Calculate net input
        :param X: {array-like}
            Input data
        :return:
            product of X  and w_
        """
        return np.dot(X, self._w[1:]) + self._w[0]
