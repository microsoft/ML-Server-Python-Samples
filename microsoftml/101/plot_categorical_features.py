# -*- coding: utf-8 -*-
"""
Categorical Features
====================

Many machine learned models only consider numerical features
and many machine learning problem do have non numerical features.
One kind is categories. Somebody's university is not
numerical, it is one item among a list of unordered possibilities.
Before training any model, we need to convert that kind of features
into numerical features.

.. index:: classification, binary, categories

One famous problem is the 
`adult data set <https://archive.ics.uci.edu/ml/datasets/adult>`_.
The goal is to predict whether somebody earns more than 50K a year or not
based on his age, education, occupation, relationship, ... Most of the features 
are categorical.
"""

columns = ["age", "workclass", "fnlwgt", "education", "educationnum", "maritalstatus",
           "occupation", "relationship", "race", "sex", "capitalgain", "capitalloss", 
           "hoursperweek", "nativecountry", "Label"]

import pandas
import os

def preprocess_data(df):
    # The data contains '?' for missing values.
    # We replace them and remove them.
    # We convert every numerical features into either str or float.
    # We remove extra spaces and put every thing in lower case.
    for c in df.columns:
        df[c] = df[c].apply(lambda x: numpy.nan if x == '?' else x)
    df = df.dropna()
    for c in df.columns:
        try:
            newc = df[c].astype(float)
            print("numerical", c)
        except Exception as e:
            print("categorical", c)
            newc = df[c].astype(str).apply(lambda s: s.strip(". ").lower())
        df[c] = newc
    return df

if os.path.exists("adult.train.csv"):
    train = pandas.read_csv("adult.train.csv")
else:
    train = pandas.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                            header=None, names=columns)
    train = preprocess_data(train)
    # We store the data on disk to avoid loading it every time
    # we execute the script.
    train.to_csv("adult.train.csv", index=False)
print(train.head())

#############################
# We do the same for the test data.

if os.path.exists("adult.test.csv"):
    test = pandas.read_csv("adult.test.csv")
else:
    test = pandas.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                            header=None, names=columns)
    test = preprocess_data(test)
    # We store the data on disk to avoid loading it every time
    # we execute the script.
    test.to_csv("adult.test.csv", index=False)
    
print(test.head())

###########################
# We convert the label into boolean.
train["Label"] = train["Label"] == ">50k"
test["Label"] = test["Label"] == ">50k"

#####################################
# The data contains numerical and categorical features.
# Let's choose a random forest because it usually works better 
# than a linear model on non continuous features.
# Let's first train a model on a couple of numerical features.

from microsoftml import rx_fast_trees, rx_predict
trees = rx_fast_trees("Label ~ age + fnlwgt + educationnum + capitalgain + capitalloss", data=train)

#########################
# Let's see the :epkg:`confusion matrix`.

y_pred = rx_predict(trees, test)
print(y_pred.head())
print(y_pred.tail())

from sklearn.metrics import confusion_matrix
conf = confusion_matrix(test["Label"], y_pred["PredictedLabel"])
print(conf)

########################
# Not very good. We need to use categorical features.
# Let's choose education without any conversion.

try:
    trees2 = rx_fast_trees("Label ~ age + fnlwgt + educationnum + capitalgain + capitalloss + education", data=train)
except Exception as e:
    print(e)

##########################
# As expected it fails. We need to convert it into a numerical feature.
# We have two options. The first one is to use 
# :epkg:`scikit-learn` on the input dataframe
# (see `OneHotEncoder <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html>`_ for example).
# Or we can ask :epkg:`microsoftml` to that instead 
# with :epkg:`microsoft:categorical`.
# We create a new variable *educationCat* and we replace it in the formula.

from microsoftml import categorical

trees2 = rx_fast_trees("Label ~ age + fnlwgt + educationnum + capitalgain + capitalloss + education_cat",
                       data=train,
                       ml_transforms=[categorical(cols=dict(education_cat="education"))])

########################################
# We look into the :epkg:`confusion matrix`.
y_pred2 = rx_predict(trees2, test)
conf = confusion_matrix(test["Label"], y_pred2["PredictedLabel"])
print(conf)

########################################
# Still not very good. We add more.

cats = {}
for col in ["workclass", "education", "maritalstatus", "occupation", 
            "relationship", "race", "sex", "nativecountry"]:
    cats[col + "_cat"] = col

formula = "Label ~ age + fnlwgt + educationnum + capitalgain + capitalloss +" + \
          " + ".join(sorted(cats.keys()))
          
print(cats)
print(formula)

trees3 = rx_fast_trees(formula, data=train,
                       ml_transforms=[categorical(cols=cats)])
y_pred3 = rx_predict(trees3, test)
conf = confusion_matrix(test["Label"], y_pred3["PredictedLabel"])
print(conf)

########################################
# .. index:: ROC
# 
# This is better. We draw the :epkg:`ROC` curve.

from sklearn.metrics import roc_curve
fpr, tpr, th = roc_curve(test["Label"], y_pred3["Probability"])

import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr, label=">50k")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('ROC - Adult Data Set')
plt.legend(loc="lower right")

#####################################
# The advantage of using the categorical transform
# inside the model is the data transformation does not 
# have to be applied on the test data. It is part of the model
# or the `pipeline <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_
# to follow :epkg:`scikit-learn` concept.
