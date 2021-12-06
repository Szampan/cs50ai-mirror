from re import M
import cv2
import numpy as np
import os
import sys
import tensorflow as tf

import time                             # ADDED
from tensorflow import keras            # ADDED
from tensorflow.keras import layers     # ADDED

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers.core import Activation

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

start_time = time.time()                # ADDED

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    print('▬▬▬▬LOAD DATA▬▬▬▬')
    print(data_dir)
    images = []
    labels = []
    dimensions = (IMG_WIDTH, IMG_HEIGHT)
    category_dirs = [i for i,j,k in os.walk(data_dir)][1:]      
    
    print(f'Cagegory dirs ({len(category_dirs)}): {category_dirs}')

    for dir in category_dirs:  
        dir_label = dir[len(data_dir):]    
        dir_images = []
        img_names = [img_name for img_name in os.listdir(dir) if os.path.isfile(os.path.join(dir, img_name))]

        for img_name in img_names:    
            img = cv2.imread(os.path.join(dir, img_name), cv2.IMREAD_COLOR)
            resized_img = cv2.resize(img, dimensions)
            dir_images.append(resized_img)  
            
        images += dir_images
        labels += [dir_label] * len(dir_images)

    print('▬▬ Image labels:', set(labels))
    return (images,labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    print('▬▬▬▬GET MODEL▬▬▬▬')

    model = keras.Sequential()
    model.add(layers.Conv2D(8, 3, activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(layers.MaxPooling2D(2)) 
    model.add(layers.Conv2D(16, 3, activation="relu"))
    model.add(layers.Conv2D(32, 3, activation="relu"))
    model.add(layers.Conv2D(64, 3, activation="relu"))
    
    model.add(layers.MaxPooling2D(2)) 
    model.add(layers.Conv2D(128, 3, activation="relu"))
        
    model.add(layers.MaxPooling2D(2)) 

    # model.add(layers.GlobalMaxPooling2D())    # Flatten layer alternative
    model.add(layers.Flatten())  

    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))

    # Output layer
    model.add(layers.Dense(NUM_CATEGORIES, activation="softmax"))

    model.summary()
    print()

    print('▬▬▬▬COMPILE:▬▬▬▬')
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model  


if __name__ == "__main__":
    main()
    print("--- %s seconds ---" % (time.time() - start_time))