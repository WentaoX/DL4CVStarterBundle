# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

from utilities.callbacks import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from utilities.nn.cnn import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output directory")
args = vars(ap.parse_args())

# Show information on the process ID
print("[INFO]: Process ID: {}".format(os.getpid()))

# Load the training and testing data, then scale it into the range [0, 1]
print("[INFO]: Loading CIFAR-10 data....")
((train_x, train_y), (test_x, test_y)) = cifar10.load_data()
train_x = train_x.astype("float") / 255.0
test_x = test_x.astype("float") / 255.0

# Convert the labels from integers to vectors
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

# Initialize the label names for the CIFAR-10 dataset
label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Initialize the SGD optimizer, but without any learning rate decay
print("[INFO]: Compiling model....")
optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Construct the set of callbacks
fig_path = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
json_path = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(fig_path, json_path)]

# train the network
print("[INFO]: Training....")
model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=64, epochs=100, callbacks=callbacks, verbose=1)
