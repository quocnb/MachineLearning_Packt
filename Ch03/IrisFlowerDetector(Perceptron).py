from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from LoadData import load_iris_and_fit_data
from PlotData import plot

X_train_std, X_test_std, y_train, y_test = load_iris_and_fit_data()

perceptron = Perceptron(max_iter=40, eta0=0.01, random_state=0)
perceptron.fit(X_train_std, y_train)

plot(X_train_std, X_test_std, y_train, y_test, perceptron, range(105, 150))

y_pred = perceptron.predict(X_test_std)
print('Missclassified sample: %d' % (y_pred != y_test).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
