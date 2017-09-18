"""
Image Featurization Applied to Image Classification
===================================================

Here is the scenario this sample addresses: train a model 
to classify or recognize the type of an image using labeled 
observations from a training set provided. Specifically, 
this sample trains a multiclass linear model using the 
:epkg:`microsoftml:rx_logistic_regression` algorithm to 
distinguish between fish, helicopter and fighter jet images. 
The multiclass training task uses the feature vectors of the 
images from the training set to learn how to classify these images.

.. index:: image, featurization

The procedure for training the model has the following steps:

- Locate the images to use in the training set and get their feature vectors.
- Label the images in the training set by type.
- Train a multiclass classifier using the :epkg:`microsoftml:rx_logistic_regression` algorithm.
- Specify a new image not in the training set to classify and use the trained model 
  to predict its type.
"""
import os

try:
    root = os.path.dirname(__file__)
except NameError:
    # __file__ does not exist in a notebook
    root = "."

# An absolute path must be used if the current folder
# is not the script's one.
image_location = os.path.abspath(os.path.join(root, "Data", "Pictures"))

###########################
# Let's take all images in the local data folder.
# The last subfolder corresponds to the class the image belongs to.
# Let's get that information too.
import glob
images = glob.glob(image_location + "/**/*.jpg", recursive=True)
images_type = [img.replace("\\", "/").split("/")[-2] for img in images]
print(list(zip(images_type, images))[0])
print(list(zip(images_type, images))[-1])

#######################
# Now since we're going to train on these images, we need to have a label.
# and to place everything in a dataframe.
# We also use float as labels.

import pandas
images_df = pandas.DataFrame(data=dict(image=images, image_type=images_type))

label_int = {'Fish': 0., 'Helicopter': 1., 'FighterJet':2., 'Flower': 3.}
images_df["Label"] = images_df["image_type"].apply(lambda t: label_int[t])
images_df = images_df[["Label", "image_type", "image"]]
print(images_df)

####################################
# Let's display the first and last images.
import matplotlib.pyplot as plt
from PIL import Image
fig, ax = plt.subplots(1, 2)
ax[0].imshow(Image.open(images_df.loc[0, "image"]))
ax[1].imshow(Image.open(images_df.loc[images_df.shape[0]-1, "image"]))


###########################
# We take out one image from this training set.
# We'll use it later to test the model on one example.
import random
h = random.randint(0, images_df.shape[0]-1)
train_df = images_df.drop(h).reset_index(drop=True)
test_df = images_df[h:h+1].reset_index(drop=True)
print(test_df)

fig, ax = plt.subplots(1, 1)
ax.imshow(Image.open(test_df.loc[0, "image"]))


########################################################
# We train a multiclass classifier using the :epkg:`microsoftml:rx_logistic_regression`
# algorithm. Just for kicks, and to compare from the previous sample, 
# we'll use the Resnet-50 model.

from microsoftml import rx_featurize, load_image, resize_image, extract_pixels, featurize_image
from microsoftml import rx_logistic_regression

image_model = rx_logistic_regression(formula="Label~Features", data=train_df, 
                                     method="multiClass", ml_transforms=[
                        load_image(cols=dict(Features="image")),
                        resize_image(cols="Features", width=227, height=227),
                        extract_pixels(cols="Features"),
                        featurize_image(cols="Features", dnn_model="Alexnet")])

############################
# Note that ``type="multiClass"`` indicates that this is a multiclass training task.
# Finally, let's give it an image and its feature vector to classify. 
# Note that this image was not part of the original training set. 
# See the actual code for details.
# Now use the model to predict the type of the image.

from microsoftml import rx_predict
prediction = rx_predict(image_model, data=test_df)
print(prediction)

###############################
# The highest score gives the predicted label.
import numpy as np
label_str = {v: k for k, v in label_int.items()}
predicted_classes = np.argmax(prediction.as_matrix(), axis=1)
print(predicted_classes)
print(label_str[predicted_classes[0]])
