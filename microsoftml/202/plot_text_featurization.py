"""
Text Featurization
==================

Many datasets contains free text features which has to be converted
into numerical features. That's what we call 
`text featurization <http://blog.revolutionanalytics.com/2017/08/text-featurization-microsoftml.html>`_.

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

train, test = train_test_split(data)

##################
# We create the columns *features* which contains n-grams probabilities
# computed from the sentances.
from microsoftml import rx_fast_trees, featurize_text, n_gram

model_word2 = rx_fast_trees("label~features", data=train, ml_transforms=[
                        featurize_text(language="English", 
                                       cols=dict(features="sentance"),
                                       word_feature_extractor=n_gram(2, weighting="TfIdf"))
                    ])

####################
# We predict.

from microsoftml import rx_predict
pred2 = rx_predict(model_word2, test, extra_vars_to_write=["score", "label"])
print(pred2.head())

#####################################
# We repeat the same with a different n-gram length.

model_word3 = rx_fast_trees("label~features", data=train, ml_transforms=[
                        featurize_text(language="English", 
                                       cols=dict(features="sentance"),
                                       word_feature_extractor=n_gram(3, weighting="TfIdf"))
                        ])
pred3 = rx_predict(model_word3, test, extra_vars_to_write=["score", "label"])
print(pred2.head())

######################
# We have now two predictions for different n-grams length.
# Let's compare them with a ROC curve
# as it is a binary classification problem.

from sklearn.metrics import roc_curve
fig, ax = plt.subplots(1, 2, figsize=(10,5))

# Positive sentiment.
pfpr2, ptpr2, pth2 = roc_curve(pred2["label"], pred2["Probability"])
pfpr3, ptpr3, pth3 = roc_curve(pred3["label"], pred3["Probability"])
ax[0].plot(pfpr2, ptpr2, label="2-grams")
ax[0].plot(pfpr3, ptpr3, label="3-grams")
ax[0].legend()
ax[0].set_title("Prediction of positive sentiments")

# Negative sentiment.
nfpr2, ntpr2, nth2 = roc_curve(1 - pred2["label"], 1 - pred2["Probability"])
nfpr3, ntpr3, nth3 = roc_curve(1 - pred3["label"], 1 - pred3["Probability"])
ax[1].plot(nfpr2, ntpr2, label="2-grams")
ax[1].plot(nfpr3, ntpr3, label="3-grams")
ax[1].legend()
ax[1].set_title("Prediction of negative sentiments")

###################################
# It looks quite similar. Let's try other featurization
# and we measure the training time.
import time

origin = time.clock()
model_word1 = rx_fast_trees("label~features", data=train, ml_transforms=[
                        featurize_text(language="English", 
                                       cols=dict(features="sentance"),
                                       word_feature_extractor=n_gram(1, weighting="TfIdf"))
                        ])
model_word1_time = time.clock() - origin
pred1 = rx_predict(model_word1, test, extra_vars_to_write=["score", "label"])

from microsoftml import n_gram_hash

# 
origin = time.clock()
model_wordh1 = rx_fast_trees("label~features", data=train, ml_transforms=[
                        featurize_text(language="English", 
                                       cols=dict(features="sentance"),
                                       word_feature_extractor=n_gram_hash(ngram_length=1))
                        ])
model_wordh1_time = time.clock() - origin
predh1 = rx_predict(model_wordh1, test, extra_vars_to_write=["score", "label"])

# 
origin = time.clock()
model_wordh1h8 = rx_fast_trees("label~features", data=train, ml_transforms=[
                        featurize_text(language="English", 
                                       cols=dict(features="sentance"),
                                       word_feature_extractor=n_gram_hash(8, ngram_length=1))
                        ])
model_wordh1h8_time = time.clock() - origin
predh1h8 = rx_predict(model_wordh1h8, test, extra_vars_to_write=["score", "label"])

# 
origin = time.clock()
model_char3 = rx_fast_trees("label~features", data=train, ml_transforms=[
                        featurize_text(language="English", 
                                       cols=dict(features="sentance"),
                                       char_feature_extractor=n_gram(3, weighting="TfIdf"))
                        ])
model_char3_time = time.clock() - origin
predc3 = rx_predict(model_char3, test, extra_vars_to_write=["score", "label"])

#
prediction = [("2-grams word", pred2),
              ("3-grams char", predc3),
              ("1-gram word", pred1),
              ("1-gram-hash word 8 bits", predh1h8),
              ("1-gram-hash word", predh1)]

from sklearn.metrics import roc_curve
fig, ax = plt.subplots(1, 1, figsize=(5,5))

for label, pred in prediction:
    pfpr, ptpr, pth = roc_curve(pred["label"], pred["Probability"])
    ax.plot(pfpr, ptpr, label=label)
ax.legend()
ax.set_title("Prediction of positive sentiments")

#############################
# The 3-grams at character level outperforms the others.
# Let's compare the training time.

dft = pandas.DataFrame(data=dict(training=[model_word1_time, model_wordh1_time, 
                                           model_wordh1h8_time, model_char3_time],
                                 experiment=["1-gram word", "1-grams-hash word", 
                                             "1-grams-hash word 8 bits", "3-grams char"]))
print(dft)                                 


####################################
# The model 1-gram-hash is significantly faster and performs
# almost the same as the model 1-gram. This functionality is 
# useful when the distinct number of terms is too big to fit
# in memory. The hash function will reduce that amount to 
# ``2 ** bits``. By design, several terms
# will be then hashed into the same value but the probability that
# several frequent terms share the same hash is very low if the number
# of bits is big enough.
# Previous figure shows that 256 grams (hash with 8 bits)
# is too low to get good performance. The number of collisions 
# between important terms is too high.
#
# Combination of two sets of features
# -----------------------------------
#
# Instead of keeping one set of features, let's combine the two.
# two options are available. The first one relies one formula,
# the second one on transform :epkg:`microsoftml.concat`.

from microsoftml import concat

model_char_word = rx_fast_trees("label~features", data=train, ml_transforms=[
                        featurize_text(language="English", 
                                       cols=dict(feat_word="sentance"),
                                       word_feature_extractor=n_gram(2, weighting="TfIdf")),
                        featurize_text(language="English", 
                                       cols=dict(feat_char="sentance"),
                                       char_feature_extractor=n_gram(3, weighting="TfIdf")),
                        # The last transform concatenates the two vectors into a single one
                        # called features and used in the formula.
                        concat(cols=dict(features=["feat_word", "feat_char"]))
                        ])

################################
# We could remove the transform ``concat`` by changing the formula into
# ``label ~ feat_word + feat_char``.
# We plot the new predictions.

predcw = rx_predict(model_char_word, test, extra_vars_to_write=["score", "label"])

prediction = [("2-grams word", pred2),
              ("3-grams char", predc3),
              ("3-grams char + 2-grams word", predcw)]

from sklearn.metrics import roc_curve
fig, ax = plt.subplots(1, 1, figsize=(5,5))

for label, pred in prediction:
    pfpr, ptpr, pth = roc_curve(pred["label"], pred["Probability"])
    ax.plot(pfpr, ptpr, label=label)
ax.legend()
ax.set_title("Prediction of positive sentiments")

