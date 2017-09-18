# -*- coding: utf-8 -*-
"""
Iris Dataset - multi-class classification problem
=================================================

The `Iris DataSet <http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html>`_
became quite popular in machine learning. It is a short and simple classification
problem with three classes. We first see how to handle it with
:epkg:`scikit-learn` and then with :epkg:`MicrosoftML`.

.. contents::
    :local:

.. index:: classification, multi-class, iris

"""

########################################
#
# scikit-learn
# ------------
#

###########################################
# We retrieves the data with scikit-learn.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import sphinx_gallery

iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

###########################################
# We train a logistic regression model
# with :epkg:`scikit-learn`.

# Training.
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X, Y)

###########################################
# We compute the predictions on a grid.
h = .02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
gridX = np.c_[xx.ravel(), yy.ravel()]

###########################
# We run the predictions on this grid.
grid = logreg.predict(gridX)

######################
# We plot the predictions.
zgrid = grid.reshape(xx.shape)
plt.figure(figsize=(4, 3))
plt.pcolormesh(xx, yy, zgrid, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title("scikit-learn")

########################################
#
# microsoftml
# -----------
#
# We use :epkg:`microsoftml` to do the same.
# The main difference is the features and the label
# must appear in a same dataframe. A formula specifies
# which role the column play during the training.
# We also modify the label type which must be 
# a boolean, a float or a category for this kind of problem.

import pandas
df = pandas.DataFrame(data=X, columns=["X1", "X2"])
df["Label"] = Y.astype(float)

###########################################################################
# :epkg:`microsoftml` must be told it is a multi-class classification problem.
# It may seem a regression compare to :epkg:`scikit-learn`.
# However because :epkg:`microsoftml` can deal with out-of-memory datasets,
# the third class could appear at the end of the training dataset.
# The parameter *verbose* can take values into 0, 1, 2.
# If > 0, :epkg:`microsoftml` displays information about the training
# on the standard output.

from microsoftml import rx_logistic_regression, rx_predict
logregml = rx_logistic_regression("Label ~ X1 + X2", data=df, method="multiClass", verbose=1)

###################################
# We convert the grid (numpy array) into a dataframe.
dfgrid = pandas.DataFrame(data=gridX, columns=["X1", "X2"])
gridml = rx_predict(logregml, dfgrid)

##################################
# :epkg:`microsoftml` returns three scores.
print(gridml.head(n=3))

##################################
# We need to pick the best one.
predicted_classes = np.argmax(gridml.as_matrix(), axis=1)

#####################
# We plot.
zgrid = predicted_classes.reshape(xx.shape)
plt.figure(figsize=(4, 3))
plt.pcolormesh(xx, yy, zgrid, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title("microsoftml")

# plt.show()
