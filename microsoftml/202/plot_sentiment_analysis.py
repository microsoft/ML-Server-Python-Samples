"""
Sentiment Analysis
==================

:epkg:`microsoftml` is provided with a couple of 
pretrained models. One of them is predicting sentiment.
Let's see how to use it on the UCI datasets:
`Sentiment Labelled Sentences Data Set <https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences>`_.

.. contents::
    :local:


Build the dataset
-----------------

The dataset first needs to be downloaded and unzipped.
Once it is done, the script can begin.
"""
import matplotlib.pyplot as plt
import pandas
import os
here = os.path.dirname(__file__) if "__file__" in locals() else "."

files = [("amazon", os.path.join(here, "data/sentiment_analysis/amazon_cells_labelled.txt")),
         ("imdb", os.path.join(here, "data/sentiment_analysis/imdb_labelled.txt")),
         ("yelp", os.path.join(here, "data/sentiment_analysis/yelp_labelled.txt"))]
    
dfs = []             
for provider, name in files:
    df = pandas.read_csv(name, sep="\t")
    df.columns = ["sentance", "label"]
    df["provider"] = provider
    dfs.append(df)
    
data = pandas.concat(dfs, axis=0)
print(data.head())

print("shape", data.shape)

###################################
# 1 means a positive sentiment, 0 negative.
#
# Pretrained model for Sentiment Analysis
# ---------------------------------------
#
# :epkg:`microsoftml` includes a pretrained model
# to predict that intent. Even if it was not trained on this data,
# let's see what kind of outputs it produces.
# We call that transformation a featurization because
# we convert text data into numerical data: the result
# of the pretrained model. We create a column
# *sentiment* from the column *sentance*.

from microsoftml import rx_featurize, get_sentiment

sentiment_scores = rx_featurize(data=data,
                    ml_transforms=[
                        get_sentiment(cols=dict(sentiment="sentance"))
                    ])

print(sentiment_scores.head())

#####################################
# Let's now how it correlates with the expected value.

import seaborn
fig, ax = plt.subplots(1, 1)
seaborn.violinplot(x="provider", y="sentiment", hue="label", split=True,
                        data=sentiment_scores, palette="Set2", ax=ax)
                        
####################################
# The model is good at predicting negative sentiment.
# It also works better on the yelp dataset meaning
# the data used to trained the model was closer 
# to yelp sentences.
#
# Featurize and Predict
# ---------------------
#
# Let's see now how a random forest would behave on this dataset.
# We first need to split into train and test dataset.
# Then we convert the text into features, we train
# a model and we evaluate it.

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

train, test = train_test_split(sentiment_scores)

##################
# We create the columns *features* which contains n-grams probabilities
# computed from the sentances. Using n-grams is the default behavior
# of function :epkg:`microsoftml:rx_featurize`.
from microsoftml import rx_fast_trees, featurize_text

model = rx_fast_trees("label~features", data=train, ml_transforms=[
                        featurize_text(language="English", 
                                       cols=dict(features="sentance"))
                    ])

####################
# We predict.

from microsoftml import rx_predict
pred = rx_predict(model, test, extra_vars_to_write=["score", "sentiment", "label"])
print(pred.head())

######################
# We have now two predictions. The first one coming from
# the pretrained model, the second one coming from the model
# we just trained. Let's compare them with a ROC curve
# as it is a binary classification problem.

from sklearn.metrics import roc_curve
fig, ax = plt.subplots(1, 2, figsize=(10,5))

# Positive sentiment.
pfpr_p, ptpr_p, pth_p = roc_curve(pred["label"], pred["sentiment"])
pfpr_m, ptpr_m, pth_m = roc_curve(pred["label"], pred["Probability"])
ax[0].plot(pfpr_p, ptpr_p, label="pretrained model")
ax[0].plot(pfpr_m, ptpr_m, label="random forest")
ax[0].legend()
ax[0].set_title("Prediction of positive sentiments")

# Negative sentiment.
nfpr_p, ntpr_p, nth_p = roc_curve(1 - pred["label"], 1 - pred["sentiment"])
nfpr_m, ntpr_m, nth_m = roc_curve(1 - pred["label"], 1 - pred["Probability"])
ax[1].plot(nfpr_p, ntpr_p, label="pretrained model")
ax[1].plot(nfpr_m, ntpr_m, label="random forest")
ax[1].legend()
ax[1].set_title("Prediction of negative sentiments")

########################
# The performance is similar on both,
# but do they agree?

from sklearn.metrics import confusion_matrix
conf = confusion_matrix(pred["sentiment"] > 0.5, pred["PredictedLabel"])
print(conf)

################################
# They seem to disagree a lot which brings the idea
# of using the pretrained model output to increase the
# performance of our trained model. That's called 
# `transfer learning <https://en.wikipedia.org/wiki/Transfer_learning>`_.
#
# Transfer Learning: leaverage a pre-trained model
# ------------------------------------------------
#

model2 = rx_fast_trees("label~features+sentiment", data=train, ml_transforms=[
                            featurize_text(language="English", 
                                       cols=dict(features="sentance"))
                    ])

pred2 = rx_predict(model2, test, extra_vars_to_write=["score", "sentiment", "label"])
print(pred2.head())

####################
# Let's display adds the new ROC on the previous graphs.

pfpr_tl, ptpr_tl, pth_tl = roc_curve(pred2["label"], pred2["Probability"])
nfpr_tl, ntpr_tl, nth_tl = roc_curve(1-pred2["label"], 1-pred2["Probability"])

fig, ax = plt.subplots(1, 2, figsize=(10,5))

# Positive sentiment.
ax[0].plot(pfpr_p, ptpr_p, label="pretrained model")
ax[0].plot(pfpr_m, ptpr_m, label="random forest")
ax[0].plot(pfpr_tl, ptpr_tl, label="transfer learning")
ax[0].legend()
ax[0].set_title("Prediction of positive sentiments")

# Negative sentiment.
ax[1].plot(nfpr_p, ntpr_p, label="pretrained model")
ax[1].plot(nfpr_m, ntpr_m, label="random forest")
ax[1].plot(nfpr_tl, ntpr_tl, label="transfer learning")
ax[1].legend()
ax[1].set_title("Prediction of negative sentiments")

###############################
# That's better! Let's see what features the model
# considers as the most important.

feature_importance = [(v, k) for k, v in model2.summary_["keyValuePairs"].items()]

################
# We keep the ten first ones.
feature_importance.sort(reverse=True)
feature_importance = feature_importance[:10]

##################
# We plot them.
import numpy
fig, ax = plt.subplots(1, 1)
ind = numpy.arange(len(feature_importance))
ax.barh(ind, [f[0] for f in feature_importance], 0.35)
ax.set_yticks(ind + 0.35 / 2)
ax.set_yticklabels([f[1] for f in feature_importance])
ax.set_title("Feature importances")


