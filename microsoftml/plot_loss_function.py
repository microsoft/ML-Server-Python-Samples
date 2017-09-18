"""
Loss Function and Fast Linear
=============================

A couple of learners expose a parameter called *loss_function*
which affects the error the model tries to minimize.
Let's see on a short example how these parameters impact
the training. We will illustrate that in a case of a linear classifier.
The impact is higher for linear learners
in the case of outliers.
  
.. contents::
    :local:

.. index:: regression, wine

Dummy data
----------

We prepare some dummy data. We need two classes quite 
well separated. We choose a line and choose a class depending on 
the side the point :math:`(x,y)` falls into.
"""
import matplotlib.pyplot as plt
import pandas
import numpy
import numpy.random as rand

def formula(x, y, e):
    return x*2+y-1.75+e

N = 200
x = rand.rand(N)
y = rand.rand(N) * 2
e = (rand.rand(N)-0.5)

data = pandas.DataFrame(dict(x=x, y=y, line=formula(x, y, e)))
data["clas"] = data.line.apply(lambda z: 1 if z > 0 else 0).astype(float)
data = data.drop("line", axis=1).copy()

print(data.groupby("clas").count())

ax = data[data.clas==0].plot(x="x", y="y", color="red", label="clas=0", kind="scatter")
data[data.clas==1].plot(x="x", y="y", color="blue", label="clas=1", ax=ax, kind="scatter")
ax.plot([0, 1], [1.75, -0.25], "--")
ax.set_title("Initial cloud of points")

#########################################
# Learn a model
# -------------
#
# Let's see how a fast linear model is doing.

from microsoftml import rx_fast_linear, rx_predict

model = rx_fast_linear("clas ~ x + y", data=data)
pred = rx_predict(model, data, extra_vars_to_write=["x", "y"])

print(pred.head())

#####################################
# We plot the model decision by filling the background of the graph.

def plot_color_class(data, model, ax, fig, side=True):
    X = data[["x", "y"]].as_matrix()
    xmin, xmax = numpy.min(X[:, 0]), numpy.max(X[:, 0])
    ymin, ymax = numpy.min(X[:, 1]), numpy.max(X[:, 1])
    dx, dy = (xmax - xmin) / 10, (ymax - ymin) / 10
    xmin -= dx
    xmax += dx
    ymin -= dy
    ymax += dy
    hx = (xmax - xmin) / 100
    hy = (ymax - ymin) / 100
    xx, yy = numpy.mgrid[xmin:xmax:hx, ymin:ymax:hy]
    grid = numpy.c_[xx.ravel(), yy.ravel()]
    dfgrid = pandas.DataFrame(data=grid, columns=["x", "y"])
    probs = rx_predict(model, dfgrid).as_matrix()[:, 1].reshape(xx.shape)

    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu", vmin=0, vmax=1)
    if side:
        ax_c = fig.colorbar(contour)
        ax_c.set_label("$P(y = 1)$")
        ax_c.set_ticks([0, .25, .5, .75, 1])
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    data[data.clas==0].plot(x="x", y="y", color="red", label="clas=0", kind="scatter", ax=ax)
    data[data.clas==1].plot(x="x", y="y", color="blue", label="clas=1", ax=ax, kind="scatter")
    ax.plot([0, 1], [1.75, -0.25], "--")

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
plot_color_class(data, model, ax, fig)
ax.set_title("Initial cloud of points\nDefault settings for logisitic regression")

########################################
# Let's add outliers
# ------------------
#
# This problem was design to be linear. Let's 
# make it more difficult by adding outlier far from
# the original linear separation.

xo = numpy.arange(0.3, 0.4, 0.02)
yo = [2.2] * len(xo)
claso = [0] * len(xo)

outlier = pandas.DataFrame(dict(x=xo, y=yo, clas=claso))
new_data = pandas.concat([data, outlier])

print(outlier.tail())

ax = new_data[new_data.clas==0].plot(x="x", y="y", color="red", label="clas=0", kind="scatter")
new_data[new_data.clas==1].plot(x="x", y="y", color="blue", label="clas=1", ax=ax, kind="scatter")
ax.plot([0, 1], [1.75, -0.25], "--")
ax.set_title("Cloud of points with outliers")

################################
# This is obviously outliers. Let's see how the model
# is behaving on those.

model = rx_fast_linear("clas ~ x + y", data=new_data)
pred = rx_predict(model, new_data, extra_vars_to_write=["x", "y"])

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
plot_color_class(new_data, model, ax, fig)
ax.set_title("Cloud of points with outliers\nDefault settings for fast linear")

##########################
# The prediction is significantly impacted 
# by the new points. We switch to another loss function
# :epkg:`microsoftml.hinge_loss`.
# It is linear error and not a log loss anymore.
# It is less sensitive to high values
# and as a consequence less sensitive to outliers.

from microsoftml import hinge_loss

consts = [0, 0.1, 1, 10]
fig, ax = plt.subplots(2, len(consts) // 2, figsize=(15, 15))
for i, const in enumerate(consts):
    a = ax[i // 2, i % 2]
    model = rx_fast_linear("clas ~ x + y", data=new_data,
                           loss_function=hinge_loss(const))
    pred = rx_predict(model, new_data, extra_vars_to_write=["x", "y"])

    plot_color_class(new_data, model, a, fig, side=False)
    a.set_title("Cloud of points with outliers\Hinge Loss\nmargin={0}".format(const))

##########################
# We can also use :epkg:`microsoftml.smoothed_hinge_loss`.

from microsoftml import smoothed_hinge_loss

consts = [0, 0.1, 1, 10]
fig, ax = plt.subplots(2, len(consts) // 2, figsize=(15, 15))
for i, smooth_const in enumerate(consts):
    a = ax[i // 2, i % 2]
    model = rx_fast_linear("clas ~ x + y", data=new_data,
                           loss_function=smoothed_hinge_loss(smooth_const))
    pred = rx_predict(model, new_data, extra_vars_to_write=["x", "y"])

    plot_color_class(new_data, model, a, fig, side=False)
    a.set_title("Cloud of points with outliers\nSmooth Hinge Loss\nconst={0}".format(smooth_const))
