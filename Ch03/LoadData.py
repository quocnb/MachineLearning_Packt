from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def load_iris_and_fit_data(test_size=0.3, random_state=0):
    X, y = load_iris(True)
    X = X[:, [2, 3]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std, X_test_std, y_train, y_test


def load_random():
    np.random.seed(0)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)

    X_positive = X_xor[y_xor == 1]
    X_negative = X_xor[y_xor == -1]
    plt.scatter(X_positive[:, 0], X_positive[:, 1], c='b', marker='x', label='1')
    plt.scatter(X_negative[:, 0], X_negative[:, 1], c='r', marker='s', label='-1')
    plt.ylim(-3.0)
    plt.legend(loc=2)
    plt.show()

    return X_xor, y_xor
