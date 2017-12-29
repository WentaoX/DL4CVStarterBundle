from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.preprocessing import SimplePreprocessor
from utilities.datasets import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained model")
args = vars(ap.parse_args())

# Initialize the class labels
classLabels = ["cat", "dog", "panda"]

# Grab a random sample of images from the dataset
print("[INFO]: Sampling images....")
image_paths = np.array(list(paths.list_images(args["dataset"])))
indexes = np.random.randint(0, len(image_paths), size=(10,))
image_paths = image_paths[indexes]

# Initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
itap = ImageToArrayPreprocessor()

# Load the dataset and scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, itap])
(data, labels) = sdl.load(image_paths)
data = data.astype("float") / 255.0

# Load the pre-trained network
print("[INFO]: Loading pre-trained network....")
model = load_model(args["model"])

# Make predictions on the images
print("[INFO] Predicting...")
predictions = model.predict(data, batch_size=32).argmax(axis=1)

# loop over the sample images
for (i, image_path) in enumerate(image_paths):
    # Load the example image, draw the prediction, and display it
    image = cv2.imread(image_path)
    cv2.putText(image, "Label: {}".format(classLabels[predictions[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
