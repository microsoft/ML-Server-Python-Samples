"""
Features selection with mutual information
==========================================

The best scenario is all the feature you need are
already computed and in your initial datasets.
But that's usually not the case. Many times,
more features need to be computed and merged altogether
before starting to train a model. In this simple case,
we are going to add square features.

.. index:: concat, features

Build fake data
---------------
"""
from sklearn.datasets import make_circles
data, labels = make_circles(1000, noise=0.1, factor=0.2, random_state=1)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
ax.scatter(data[labels==0, 0], data[labels==0, 1], label="class 0")
ax.scatter(data[labels==1, 0], data[labels==1, 1], label="class 1")

#############################
# We put the data into a dataframe.

import pandas
df = pandas.DataFrame(data=data, columns=["X1", "X2"])
df["Label"] = labels


#################################
# We train a logistic regression.

from microsoftml import rx_logistic_regression, rx_predict
logreg = rx_logistic_regression("Label ~ X1 + X2", data=df)

#################################
# And we display the results.

import numpy

def colorie(X, model, ax, fig, additional_columns=None, additional_names=None):
    if isinstance(X, pandas.DataFrame):
        X = X.as_matrix()
    xmin, xmax = numpy.min(X[:, 0]), numpy.max(X[:, 0])
    ymin, ymax = numpy.min(X[:, 1]), numpy.max(X[:, 1])
    hx = (xmax - xmin) / 100
    hy = (ymax - ymin) / 100
    xx, yy = numpy.mgrid[xmin:xmax:hx, ymin:ymax:hy]
    grid = numpy.c_[xx.ravel(), yy.ravel()]
    names = ["X1", "X2"]
    if additional_columns:
        add = numpy.zeros((grid.shape[0], 3))
        for i, f in enumerate(additional_columns):
            vf = numpy.vectorize(f)
            add[:, i] = vf(grid[:, 0], grid[:, 1])
        grid = numpy.hstack([grid, add])
        names += additional_names
            
    dfgrid = pandas.DataFrame(data=grid, columns=names)
    probs = rx_predict(model, dfgrid).as_matrix()[:, 1].reshape(xx.shape)

    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu", vmin=0, vmax=1)
    ax_c = fig.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])


X = df[["X1", "X2"]].as_matrix()
Y = df[["Label"]]

fig = plt.figure(figsize=(7, 5))
ax = plt.subplot()
colorie(X, logreg, ax, fig)
ax.scatter(X[:, 0], X[:, 1], c=Y, s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)
ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")

################################
# The model does not know how to classify because
# it is looking for a linear separation between the 
# two classes and that cannot be found.
# Let's add more features to make the problem be linear.

df["X1X1"] = df["X1"] * df["X1"]
df["X2X2"] = df["X2"] * df["X2"]
df["X1X2"] = df["X1"] * df["X2"]

logreg = rx_logistic_regression("Label ~ X1 + X2 + X1X1 + X1X2 + X2X2", data=df)

fig = plt.figure(figsize=(7, 5))
X = df[["X1", "X2"]].as_matrix()
ax = plt.subplot()

colorie(X, logreg, ax, fig, additional_columns=[
            (lambda x,y: x*x),
            (lambda x,y: x*y),
            (lambda x,y: y*y)],
            additional_names=["X1X1", "X1X2", "X2X2"])
ax.scatter(X[:, 0], X[:, 1], c=Y, s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)
ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")

###################################
# Let's see the coefficients.

print(logreg.coef_)

#################################
# That's a lot better! But is there to remove less useful
# variable before training?
# Let's see :epkg:`microsoft:mutualinformation_select` in order to keep
# only three features (including the bias).

from microsoftml import mutualinformation_select
logreg = rx_logistic_regression("Label ~ X1 + X2 + X1X1 + X1X2 + X2X2", data=df,
                    ml_transforms=[
                            mutualinformation_select(cols=["X1", "X2", "X1X1", "X1X2", "X2X2"], 
                                                     label="Label", num_features_to_keep=3)])

fig = plt.figure(figsize=(7, 5))
X = df[["X1", "X2"]].as_matrix()
ax = plt.subplot()

colorie(X, logreg, ax, fig, additional_columns=[
            (lambda x,y: x*x),
            (lambda x,y: x*y),
            (lambda x,y: y*y)],
            additional_names=["X1X1", "X1X2", "X2X2"])
ax.scatter(X[:, 0], X[:, 1], c=Y, s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)
ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")

###################################
# And the coefficients.

print(logreg.coef_)

#################################
# The method keeps the features which are correlated with
# the target. In our case, the best one would be :math:`x_1^2 + x_2^2`
# which is not part of our sample.
