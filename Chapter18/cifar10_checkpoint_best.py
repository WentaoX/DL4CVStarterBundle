from sklearn.preprocessing import LabelBinarizer
from utilities.nn.cnn import MiniVGGNet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True,
                help="path to weights directory")
args = vars(ap.parse_args())

# Load the training and testing data, then scale it into the range [0, 1]
print("[INFO]: Loading CIFAR-10 data....")
((train_x, train_y), (test_x, test_y)) = cifar10.load_data()
train_x = train_x.astype("float") / 255.0
test_x = test_x.astype("float") / 255.0

# Convert the labels from integers to vectors
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

# Initialize the SGD optimizer, but without any learning rate decay
print("[INFO]: Compiling model....")
optimizer = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Construct the callback to save only the 'best' model to disk based on the validation loss
checkpoint = ModelCheckpoint(args['weights'], monitor="val_loss", mode="min", save_best_only=True, verbose=1)
callbacks = [checkpoint]

# Train the network
print("[INFO]: Training network....")
H = model.fit(train_x, train_y, validation_data=(test_x, test_y),
              batch_size=64, epochs=40, callbacks=callbacks, verbose=2)
