from Perceptron import Perceptron
from IrisDataLoader import load_iris_data
from IrisDataLoader import plot_decision_regions
import matplotlib.pyplot as plt

X, y = load_iris_data()

plt.scatter(X[:50, 0], X[:50, 1], marker='o', color='r', label='setosa')
plt.scatter(X[50:, 0], X[50:, 1], marker='x', color='b', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron()
ppn.fit(X, y)
plt.plot(range(1, len(ppn.error) + 1), ppn.error, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassification')
plt.show()

plot_decision_regions(X, y, ppn)
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.legend(loc='upper left')
plt.show()
