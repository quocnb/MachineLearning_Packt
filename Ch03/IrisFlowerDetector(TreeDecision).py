from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from LoadData import load_iris_only
from PlotData import plot
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = load_iris_only()
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)
plot(X_train, X_test, y_train, y_test, tree)
y_pred = tree.predict(X_test)
print('Accuracy (Tree) = %.2f' % accuracy_score(y_test, y_pred))

# Export decision tree
#export_graphviz(tree, out_file='tree.dot', feature_names=['petal length', 'petal width'])
# Install GraphViz and run 'dot -Tpng tree.dot -o tree.png' in the terminal for convert .dot to .png file


forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)
plot(X_train, X_test, y_train, y_test, forest)
y_pred = forest.predict(X_test)
print('Accuracy (Forest) = %.2f' % accuracy_score(y_test, y_pred))
