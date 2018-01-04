from sklearn.neighbors import KNeighborsClassifier
from LoadData import load_iris_and_fit_data
from PlotData import plot
from sklearn.metrics import accuracy_score

X_train_std, X_test_std, y_train, y_test = load_iris_and_fit_data()

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

plot(X_train_std, X_test_std, y_train, y_test, knn)

y_pred = knn.predict(X_test_std)
print('Accuracy (Forest) = %.2f' % accuracy_score(y_test, y_pred))
