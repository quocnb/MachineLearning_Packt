from PlotData import plot_decision_regions
from PlotData import plot
from sklearn.svm import SVC
from LoadData import load_random
from LoadData import load_iris_and_fit_data
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

X_xor, y_xor = load_random()
svm = SVC(kernel='rbf', C=10.0, random_state=0, gamma=0.1)
svm.fit(X_xor, y_xor)

plt.figure(1)
plot_decision_regions(X_xor, y_xor, svm)
plt.legend(loc=2)
plt.title('C=10.0, gamma=0.1')

svm2 = SVC(kernel='rbf', C=10.0, random_state=0, gamma=0.2)
svm2.fit(X_xor, y_xor)

plt.figure(2)
plot_decision_regions(X_xor, y_xor, svm2)
plt.legend(loc=2)
plt.title('C=10.0, gamma=0.2')


svm3 = SVC(kernel='rbf', C=1.0, random_state=0, gamma=0.2)
svm3.fit(X_xor, y_xor)

plt.figure(3)
plot_decision_regions(X_xor, y_xor, svm3)
plt.legend(loc=2)
plt.title('C=1.0, gamma=0.2')

svm4 = SVC(kernel='rbf', C=1.0, random_state=0, gamma=1000)
svm4.fit(X_xor, y_xor)

plt.figure(4)
plot_decision_regions(X_xor, y_xor, svm4)
plt.legend(loc=2)
plt.title('C=1.0, gamma=1000')


plt.show()

X_train_std, X_test_std, y_train, y_test = load_iris_and_fit_data()
svm.fit(X_train_std, y_train)
plot(X_train_std, X_test_std, y_train, y_test, svm)

svm4.fit(X_train_std, y_train)
plot(X_train_std, X_test_std, y_train, y_test, svm4)

y_pred = svm.predict(X_test_std)
print('Accuracy (gamma = 0.01) = %.2f' % accuracy_score(y_test, y_pred))

y_pred4 = svm4.predict(X_test_std)
print('Accuracy (gamma = 1000) = %.2f' % accuracy_score(y_test, y_pred4))
