"""
Use a regression to predict wine quality
========================================

I will use `wine quality data set <https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv>`_
from the `UCI Machine Learning Repository <https://archive.ics.uci.edu/ml/datasets.html>`_.  
The dataset contains quality ratings (labels) for a 1599 red wine samples. 
The features are the wines' physical and chemical properties (11 predictors). 
We want to use these properties to predict the quality of the wine. 
The experiment is shown below and can be found in the 
`Cortana Intelligence Gallery <https://gallery.cortanaintelligence.com/Experiment/Predict-Wine-Quality-Classification-10>`_

.. index:: regression, wine

*Sources:* 

- `Predicting Wine Quality with Azure ML and R <http://blog.revolutionanalytics.com/2016/04/predicting-wine-quality.html>`_
- `Predicting Wine
  Quality <https://github.com/shaheeng/ClassificationModelEvaluation/blob/master/PredictWineQuality_RevBlog3/Predicting%20Wine%20Quality%20-%20Shaheen.ipynb>`_
  (notebook)
  
*Contents:*
  
.. contents::
    :local:

Processing the data
-------------------

Let's start with collecting and preparing the data.
We save them in a single file in order to avoid downloading them
many times.
"""
import matplotlib.pyplot as plt
import pandas
import os

if not os.path.exists("wines_backup.csv"):
    # if not exist, we create wines.csv which combines red and white wines into a single file
    columns = ["facidity", "vacidity", "citric", "sugar", "chlorides", "fsulfur", 
               "tsulfur", "density", "pH", "sulphates", "alcohol", "quality"]
    red = pandas.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                         names=columns, sep=";", skiprows=1)
    white = pandas.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
                         names=columns, sep=";", skiprows=1)
    red["color"] = "red"
    white["color"] = "white"
    wines = pandas.concat([white, red])
    wines.to_csv("wines_backup.csv", sep="\t", index=False)
else:
    wines = pandas.read_csv("wines_backup.csv", sep="\t")
    
print(wines.head(n=5))

#############################
# The goal is to predict the quality of the wines.
# Let's see how this variable is distributed.

fig, ax = plt.subplots(1, 1)
wines["quality"].hist(bins=7, ax=ax)
ax.set_xlabel("quality")
ax.set_ylabel("# wines")

######################################
# Is there any differance between red and white wines?

red = wines[wines.color=="red"]["quality"]
white = wines[wines.color=="white"]["quality"]

fig, ax = plt.subplots(1, 1)
ax.hist([red, white], label=["red", "white"], alpha=0.5,
        histtype='bar', bins=7, color=["red", "green"])
ax.legend()
ax.set_xlabel("quality")
ax.set_ylabel("# wines")

############################
# There are more white wines and more high quality white wines.
# Let's see if the quality is correlated to the alcohol degree?

fig, ax = plt.subplots(1, 1)
ax.scatter(x=wines.alcohol, y=wines.quality)
ax.set_xlabel("alcohol")
ax.set_ylabel("quality")

###############################
# Quite difficult to see don't you think?

fig, ax = plt.subplots(1, 1)
wines.plot.hexbin(x='alcohol', y='quality', ax=ax, gridsize=25)

#####################################
# The alcohol does not explain the quality all by itself.
#
# Predict the quality of the wine
# -------------------------------
#
# The quality is a mark between 1 and 9.
# We use a fast tree regression to predict it.
# But before anything starts, we need to split the dataset
# into train and test.

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
wines_train, wines_test = train_test_split(wines)

##############################
# And we train. We drop the color which is a non numerical
# features. We will add it later.

from microsoftml import rx_fast_trees
cols = wines.columns.drop(["quality", "color"])
model = rx_fast_trees("quality ~" + "+".join(cols), data=wines_train, method="regression")

######################
# We predict.

from microsoftml import rx_predict
pred = rx_predict(model, wines_test, extra_vars_to_write=["quality"])
print(pred.head())

###########################
# The column 'Score' is the prediction.
# We estimate its quality with the metric `R2 <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html>`_
# and we plot them.

from sklearn.metrics import r2_score
r2 = r2_score(pred.quality, pred.Score)
print("R2=", r2)

fig, ax = plt.subplots(1, 1)
ax.scatter(x=pred.quality, y=pred.Score)
ax.set_xlabel("quality")
ax.set_ylabel("prediction")

#########################
# Still not easy to read.
fig, ax = plt.subplots(1, 1)
pred.plot.hexbin(x='quality', y='Score', ax=ax, gridsize=25)

################################
# It seems to be doing a relatively good job to predict
# marks 5, 6, 7. As we saw with the distribution, 
# the dataset contain many examples for these marks
# and not many for the others.
#
# .. index:: feature importance
#
# Feature Importance
# ------------------
# 
# Let's see which variables contribute the most to the prediction.

feature_importance = [(k, v) for k, v in model.summary_["keyValuePairs"].items()]

import numpy
fig, ax = plt.subplots(1, 1)
ind = numpy.arange(len(feature_importance))
ax.barh(ind, [f[1] for f in feature_importance], 0.35)
ax.set_yticks(ind + 0.35 / 2)
ax.set_yticklabels([f[0] for f in feature_importance])
ax.set_title("Feature importances")

#####################################
# Alcohol is the dominant feature but the others still play
# an important part in the prediction.
# 
# Does the color help?
# --------------------
#
# To answer that question, we need to add the wine color
# as a new feature. Because it is a categorical feature, we 
# need to convert it into a numerical one.
# We use the transform :epkg:`microsoftml:categorical`
# to convert column *color* into *color_num*.

from microsoftml import categorical
cols = list(wines.columns.drop(["quality", "color"]))  # We still drop column color.
cols.append("color_num")  # But we add the new one.
model = rx_fast_trees("quality ~" + "+".join(cols), data=wines_train, method="regression",
                      ml_transforms=[categorical(cols=dict(color_num="color"))])
pred = rx_predict(model, wines_test, extra_vars_to_write=["quality"])
r2_color = r2_score(pred.quality, pred.Score)
print("R2 with colors=", r2_color)

#####################################
# Performance is not better. Let's confirm that with 
# the feature importances.

feature_importance = [(k, v) for k, v in model.summary_["keyValuePairs"].items()]

import numpy
fig, ax = plt.subplots(1, 1)
ind = numpy.arange(len(feature_importance))
ax.barh(ind, [f[1] for f in feature_importance], 0.35)
ax.set_yticks(ind + 0.35 / 2)
ax.set_yticklabels([f[0] for f in feature_importance])
ax.set_title("Feature importances with colors")

#########################
# Color does not help or we can say that the prediction model
# is color blind.
