from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# train a decision tree model
clf = DecisionTreeClassifier()
clf.fit(X, y)

# visualize the decision tree
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, ax=ax)
plt.show()
