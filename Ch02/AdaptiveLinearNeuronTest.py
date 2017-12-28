from IrisDataLoader import load_iris_data
from IrisDataLoader import plot_decision_regions
from AdaptiveLinearNeuron import Adaline
from AdaptiveLinearNeuron import AdalineSGD
import matplotlib.pyplot as plt
import numpy as np

X, y = load_iris_data()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ada1 = Adaline(eta=0.01, n_iter=10).fit(X, y)

ax[0].plot(range(1, len(ada1.cost) + 1), np.log10(ada1.cost), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = Adaline(eta=0.0001, n_iter=10).fit(X, y)

ax[1].plot(range(1, len(ada2.cost) + 1), ada2.cost, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.show()

# Apply standardization
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

ada = Adaline(n_iter=15, eta=0.01)
ada.fit(X_std, y)


plot_decision_regions(X_std, y, classifier=ada)
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.title('Adaline - Gradient Descent')
plt.legend(loc='upper right')
plt.show()
plt.plot(range(1, len(ada.cost) + 1), ada.cost, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()


ada_sgd = AdalineSGD(eta=0.01, n_iter=10, random_state=1)
ada_sgd.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('Adaline Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada_sgd.cost) + 1), ada_sgd.cost, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()
