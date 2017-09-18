"""
Image Featurization Applied to Find Similar Images
==================================================

Here is the scenario this sample addresses: You have a catalog 
of images in a repository. When you get a new image, you want 
to find the image from your catalog that most closely matches 
this new image.
The procedure for finding the best match has the following steps:
    
- Locate the images in the catalogue and get their feature vectors.
- Locate the new image and get its feature vector.
- Find out which image or set of images from the catalog has the 
  smallest "distance" from the new image. There are a number of 
  ways to calculate this distance. A simple one is the Euclidean 
  distance, which we use in this sample.

In this sample, our intial catalog consists a set of pictures of fish and helicopters.
First, create a dataframe with the locations of these images:

.. index:: image, similarity
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

############################
# Specify paths to the images we want to featurize.

images = []
for im in ["Fish/Fish1.jpg", "Fish/Fish2.jpg", 
           "Helicopter/Helicopter1.jpg", "Helicopter/Helicopter2.jpg"]:
    images.append(os.path.join(image_location, im))

###############################
# Let's plot the image to see what they look like.

import matplotlib.pyplot as plt
from PIL import Image
fig, ax = plt.subplots(2, 2)
for i, im in enumerate(images):
    ax[i // 2, i % 2].imshow(Image.open(im))

################
# Setup a dataframe with the path to the image.

import pandas
image_df = pandas.DataFrame(data=dict(image=images))
print(image_df)

#########################
# Then, get the corresponding feature vectors for each 
# of the catalog images into a dataframe.
# We follow the process mentioned at :ref:`l-imgfeat`.
# We load, resize, convert into pixels and finally build
# vectors from images.

from microsoftml import rx_featurize, load_image, resize_image, extract_pixels, featurize_image
image_vector = rx_featurize(data=image_df, ml_transforms=[
    load_image(cols=dict(Features="image")),
    resize_image(cols="Features", width=227, height=227),
    extract_pixels(cols="Features"),
    featurize_image(cols="Features", dnn_model="Alexnet")])

print(image_vector.head())

###################################
# Secondly, create a dataframe with the location of 
# the new image to match and get its feature vector into a dataframe.

images_match = []
for im in ["Fish/Fish4.jpg"]:
    images_match.append(os.path.join(image_location, im))
    
fig, ax = plt.subplots(1, 1)
ax.imshow(Image.open(images_match[0]))

image_match_df = pandas.DataFrame(data=dict(image=images_match))

image_match_vectors = rx_featurize(data=image_match_df, ml_transforms=[
    load_image(cols=dict(Features="image")),
    resize_image(cols="Features", width=227, height=227),
    extract_pixels(cols="Features"),
    featurize_image(cols="Features", dnn_model="Alexnet")])
    
print(image_match_vectors.head())

###########################
# Thirdly, compare the new image with the images in the 
# catalogue to find the best match. 
# We have 2 sets of feature vectors: 
#
# - ``image_vectors`` contains the feature vectors for the catalog images; 
# - ``image_match_vectors`` contains the feature vector of the new image to be compared.
#
# The best match is defined (for our purposes) as the image pair 
# with the least Euclidean distance between their image feature 
# vectors where one of the feature vectors is for the new image. 
# We implement these calculations using 
# `cdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist>`_.

matimg = image_vector.drop("image", axis=1).as_matrix()
matmat = image_match_vectors.drop("image", axis=1).as_matrix()

from scipy.spatial.distance import cdist
distance = cdist(matimg, matmat)
print(distance)

#######################
# It contains 4 values corresponding to the Euclidian
# distance between the new image and the first four images
# we used as reference.
#
# .. note:: The actual values can change slightly depending on the machine 
#       used to run the code, but the order relations between the distance 
#       values should be invarient.
#
# And the winner is...

arg = distance.argmin()
print(arg)

fig, ax = plt.subplots(1, 1)
ax.imshow(Image.open(images[arg]))

