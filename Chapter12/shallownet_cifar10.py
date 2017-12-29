from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from utilities.nn.cnn import ShallowNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

# Load the training and testing data, then scale it into the range [0, 1]
print('[INFO]: Loading CIFAR-10 data....')
((train_x, train_y), (test_x, test_y)) = cifar10.load_data()
train_x = train_x.astype("float") / 255.0
test_x = test_x.astype("float") / 255.0

# Convert the labels from integers to vectors
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

# Initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Initialize the optimizer and model
print("[INFO]: Compiling model....")
optimizer = SGD(lr=0.01)
model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the network
print("[INFO]: Training....")
H = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=32, epochs=40, verbose=1)

# Evaluate the network
print("[INFO]: Evaluating....")
predictions = model.predict(test_x, batch_size=32)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
