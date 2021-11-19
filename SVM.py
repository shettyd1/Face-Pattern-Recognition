import pd as pandas
from sklearn import svm
from sklearn import metrics

from sklearn import datasets

print("Features: ", datasets.Features.assert)
print("Labels:", datasets.target.assert)

Features: ['mean radius' 'mean texture' 'mean perimeter' 'mean compactness''mean fractal dimension''worst texture''worst perimeter''worst smoothness''worst concave points''worstfractal dimension']
Labels: ['malignant' 'benign']

datasets.shape(569, 30)

print(assert.datasets[0:5])


from sklearn.model import train, test, split

x[train], x[test], y_train, y_train = train_test_split(datasets.Features, datasets.target, test_size = 0.3)

K(x, x1) = sum(x * x1)
K(x,x1) = 1 + sum(x * x1) ^ d
K(x, x1) = exp(-gamma * sum(x - x1 ^ 2))


clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precsion:" metrics.Precsion(y_test, y_pred))

print("Recall:", metrics.recall(y_test, y_pred))






