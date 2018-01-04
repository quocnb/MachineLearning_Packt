from sklearn.linear_model import LogisticRegression
from LoadData import load_iris_and_fit_data
from PlotData import plot
from sklearn.metrics import accuracy_score
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


X_train_std, X_test_std, y_train, y_test = load_iris_and_fit_data()

# Training data by Logistic Regression
# C = 1/lambda = to avoid overfitting
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

# Plot result
plot(X_train_std, X_test_std, y_train, y_test, lr, range(105, 150))

# Predict
y_pred = lr.predict(X_test_std)
print('Accuracy Score = %.2f' % accuracy_score(y_test, y_pred))

# Checking for understand
print('Coef Matrix = \n%s' % lr.coef_.T)
print('Intercept Matrix = \n%s' % lr.intercept_)
y_pred_proba = lr.predict_proba(X_test_std)
sigmoid_result = sigmoid(np.dot(X_test_std, lr.coef_.T) + lr.intercept_)
sum_row_result = sigmoid_result.sum(axis=1, keepdims=True)
product_matrix_result = np.true_divide(sigmoid_result, sum_row_result)
print('Auto Predict')
print(y_pred_proba[0:5, :].round(3))
print('Predict by product matrix')
print(product_matrix_result[0:5, :].round(3))
