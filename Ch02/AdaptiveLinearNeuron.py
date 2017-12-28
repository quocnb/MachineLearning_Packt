import numpy as np
from numpy.random import seed

class Adaline(object):
    """ Adaptive Linear Neuron classifier
    :param eta: float
        Learning rate (between 0.0 and 1.0)
    :param n_iter: int
        Passes over the training dataset.
    :param _w: {1d-array}
        Weights after fitting.
    :param _error: list
        Number of misclassifications in every epoch.
    """
    eta = 0.01
    n_iter = 10
    _w = []
    cost = []
    _w_initialized = False

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Fit training data
        :param X: {array-like}, shape = [m_samples, n_features]
            Training vectors
        :param y: {array-like}, shape = m_samples
            Target values
        :return:
            self.object
        """
        self._w = np.zeros(X.shape[1] + 1)
        self.cost = []
        for i in range(self.n_iter):
            outputs = self.net_input(X)
            errors = y - outputs
            self._w[1:] += self.eta * X.T.dot(errors)
            self._w[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2
            self.cost.append(cost)
        return self

    def net_input(self, X):
        """
        Calculate net input
        :param X:
            Training data
        :return:
        """
        return np.dot(X, self._w[1:]) + self._w[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0, 1, -1)


class AdalineSGD(object):
    """ ADAptive LInear NEuron Stochastic Gradient Descent
    @:param eta: float
        Learning rate (0.0 -> 1.0)
    @:param n_iter: int
        Passes over the traning set
    @:param shuffle: bool (default = true)
        Shuffles traning data every epoch
    @:param random_state: int (default = None)
        Set random state for shuffling and initializing weights
    Attributes
    _w : 1d-array
        Weights after fiting
    errors: list
        Number of misclassification in every epoch
    """
    eta = 0.01
    n_iter = 10
    cost = []
    _w = []
    errors = []
    shuffle = True
    random_state = None

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self._w_initialized = False
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        """
        Traning Data (X, y)
        :param X: {array-like}, shape = (m_sample x n_features)
            Traning vectors, m_sample = number of sample, n_features = number of features
        :param y: {array-like}, shape = (m_sample)
            Target values
        :return:
            self: object
        """
        self._initialize_weights(X.shape[1])
        for n in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            epoch_cost = []
            for xi, target in zip(X, y):
                epoch_cost.append(self._update_weights(xi, target))
            avg_epoch_cost = sum(epoch_cost) / len(y)
            self.cost.append(avg_epoch_cost)
        return self

    def partial_fit(self, X, y):
        """
        Fit traning data without reinstializing weights
        :param X
            Training example
        :param y
            Training target values
        :return:
        """
        if not self._w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    @staticmethod
    def _shuffle(X, y):
        """
        Shuffle traning data
        :param X:
            Training data
        :param y:
            Traning target values
        :return:
            New shuffled X, y
        """
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, n):
        """
        Init weights
        :param n: int
             number of columns of weights
        :return:
        self : object
        """
        self._w = np.zeros(1+n)
        self._w_initialized = True

    def _update_weights(self, xi, target):
        """
        Update weights via (xi, target) data
        :param xi:
        :param target:
        :return:
            cost
        """
        output = self._net_input(xi)
        error = target - output
        self._w[1:] += self.eta * xi.dot(error)
        self._w[0] += self.eta * error
        epoch_cost = 1/2 * error**2
        return epoch_cost

    def _net_input(self, X):
        """
        Calculator net input
        :param X:
            Training data
        :return:
            Predict target results ( = X * w)
        """
        return np.dot(X, self._w[1:]) + self._w[0]

    def predict(self, X):
        return np.where(self._net_input(X) >= 0, 1, -1)
