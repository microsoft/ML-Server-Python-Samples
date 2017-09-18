"""
Input schemas is important
==========================

:epkg:`microsoftml` is handling many types of features.
It does add a couple of transformations without the 
user knowing which may see some errors difficult
to interpret due to that.

.. index:: sparse, patsy

Build fake data
---------------

We just need fake to begin with, 1 label, 3 random features
and one categorical feature which looks like numeric.
"""
from numpy.random import randn
matrix = randn(2000, 3)

import random
import pandas
data = pandas.DataFrame(data=matrix, columns=["Label"] + ["f%s" % i for i in range(1, matrix.shape[1])])
data["Label"] = (data["Label"] > 0.5).apply(lambda x: 1.0 if x else 0.0)
data["cat"] = [["0", "1"][random.randint(0,1)] for i in range(0, data.shape[0])]

################################
# We define this column as a category.
data["cat"] = data["cat"].astype("category")

print("problem dimension:", data.shape)
print(data.head())

###################################################
# Let's train a logistic regression.

formula = "Label ~ {0}".format(" + ".join(data.columns[1:]))
print(formula)

from microsoftml import rx_logistic_regression
logregml = rx_logistic_regression(formula, data=data)

#########################################
# Let's predict now.

from microsoftml import rx_predict
scores= rx_predict(logregml, data=data)
print(scores.head())

#########################################
# Let's change the type of the category into numerical
# and predict again.

data["cat"] = data["cat"].astype(float)
try:
    scores= rx_predict(logregml, data=data)
except Exception as e:
    # It fails because the input schema is not the same.
    print(e)

##################################
# The same schema must in use for training and testing.


