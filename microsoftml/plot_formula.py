"""
microsoftml and formula
=======================

:epkg:`microsoftml` is using :epkg:`patsy` to process formulas.
It has some limitations such as a high number of features.
Let's see how to work around that issue.

.. index:: sparse, patsy

Build fake data
---------------

We just need fake to begin with, 1 label and 2000 random features.
"""

from numpy.random import randn
matrix = randn(2000, 2001)

import pandas
data = pandas.DataFrame(data=matrix, columns=["Label"] + ["f%s" % i for i in range(1, matrix.shape[1])])
data["Label"] = (data["Label"] > 0.5).apply(lambda x: 1.0 if x else 0.0)

print("problem dimension:", data.shape)
print(data[["Label", "f1", "f2", data.columns[-1]]].head())

###################################################
# Let's train a logistic regression.

formula = "Label ~ {0}".format(" + ".join(data.columns[1:]))
print(formula[:50] + " + ...")

from microsoftml import rx_logistic_regression

try:
    logregml = rx_logistic_regression(formula, data=data)
except Exception as e:
    # The error is expected because patsy cannot handle
    # so many features.
    print(e)

#########################################
# Let's skip patsy's parser to manually define the formula
# with object `ModelDesc <http://patsy.readthedocs.io/en/latest/API-reference.html?highlight=lookupfactor#patsy.ModelDesc>`_.

from patsy.desc import ModelDesc, Term
from patsy.user_util import LookupFactor

patsy_features = [Term([LookupFactor(n)]) for n in data.columns[1:]][:10]
model_formula = ModelDesc([Term([LookupFactor("Label")])], [Term([])] + patsy_features)

print(model_formula.describe() + " + ...")
logregml = rx_logistic_regression(model_formula, data=data)

