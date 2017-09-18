"""
Grid Search
===========

All learners have what we call 
`hyperparameters <https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)>`_
which impact the way a model is trained. Most of the time, they have a default
value which works on most of the datasets but that does not mean that's the best
possible value for the current dataset. Let's see how to choose them
on a the following dataset 
`Epileptic Seizure Recognition Data Set  <https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition>`_.

.. contents::
    :local:


The data
--------

"""
import matplotlib.pyplot as plt
import pandas
import os
here = os.path.dirname(__file__) if "__file__" in locals() else "."
data_file = os.path.join(here, "data", "epilepsy", "data.csv")

data = pandas.read_csv(data_file, sep=",")
data["y"] = data["y"].astype("category")
print(data.head(2))
print(data.shape)

#########################################
# The variable of interest is ``y``.

print(set(data["y"]))

###################"
# 1 is epilepsy, 2-5 is not and the distribution is uniform.
# We convert that problem into a binary classification problem,
# 1 for epileptic, 0 otherwise.

data["y"] = data["y"].apply(lambda x: 1 if x == 1 else 0)

print(data[["y", "X1"]].groupby("y").count())

##########################
# We split into train/test.
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
train, test = train_test_split(data)

###################
# First model
# -----------
#
# Let's fit a logistic regression.

import numpy as np
from microsoftml import rx_fast_trees, rx_predict
features = [c for c in train.columns if c.startswith("X")]

model = rx_fast_trees("y ~ " + "+".join(features), data=train)

pred = rx_predict(model, test, extra_vars_to_write=["y"])
print(pred.head())

#################
# Let's compute the confusion matrix.
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(pred["y"], pred["PredictedLabel"])
print(conf)

########################
# The prediction is quite accurate. Let's see if we can improve it.
#
# Optimize hyperparameters
# ------------------------
#
# We split the training set into a smaller train set and a smaller test set.
# The dataset is then split into three buckets: A, B, C. A is used to train a model,
# B is used to optimize hyperparameters, C is used to validate.
# We define a function which train and test on buckets A, B.


def train_test_hyperparameter(trainA, trainB, **hyper):
    # Train a model and 
    features = [c for c in train.columns if c.startswith("X")]
    model = rx_fast_trees("y ~ " + "+".join(features), data=trainA, verbose=0, **hyper)
    pred = rx_predict(model, trainB, extra_vars_to_write=["y"])
    conf = confusion_matrix(pred["y"], pred["PredictedLabel"])
    return (conf[0,0] + conf[1,1]) / conf.sum()

#############################
# We look into one parameter *num_leaves* to see if the number of trees impacts
# the performance. 

trainA, trainB = train_test_split(train)

hyper_values = [5, 10, 15, 20, 25, 30, 35, 40, 50, 100, 200]
perfs = []
for val in hyper_values:
    acc = train_test_hyperparameter(trainA, trainB, num_leaves=val)
    perfs.append(acc)
    print("-- Training with hyper={0} performance={1}".format(val, acc))
    
#########################################
# We finally plot the curve.

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
ax.plot(hyper_values, perfs, "o-")
ax.set_xlabel("num_leaves")
ax.set_ylabel("% correctly classified")

###################################
# Let's choose the best value, we check our finding on the test sets
# after we train of the whole training set.
tries = max(zip(perfs, hyper_values))
print("max={0}".format(tries))

model = rx_fast_trees("y ~ " + "+".join(features), data=train, num_leaves=tries[1])
pred = rx_predict(model, test, extra_vars_to_write=["y"])
conf = confusion_matrix(pred["y"], pred["PredictedLabel"])
print(conf)

########################
# The process we followed relies on one training per
# value of the hyperparameter. This could be improved by
# running `cross validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)>`_
# and each of them.


