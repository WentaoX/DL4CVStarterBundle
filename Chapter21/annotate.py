from imutils import paths
import argparse
import imutils
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input directory of images")
ap.add_argument("-a", "--annotation", required=True,
                help="path to output directory of annotations")
args = vars(ap.parse_args())

# Grab the image paths then initialize the dictionary of character counts
image_paths = list(paths.list_images(args["input"]))
counts = {}

# Loop over the image paths
for (i, image_path) in enumerate(image_paths):
    # Display an update to the user
    print("[INFO]: Processing image {}/{}".format(i + 1, len(image_paths)))

    try:
        # Load the image and convert it to grayscale, then pad the image to ensure digits caught only the border of the
        # image are retained
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        # Threshold the image to reveal the digits
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Find contours in the image, keeping only the four largest ones
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if imutils.is_cv2() else contours[1]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

        # Loop over the contours
        for c in contours:
            # Compute the bounding box for the contour then extract the digit
            (x, y, w, h) = cv2.boundingRect(c)
            roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

            # Display the character, making it larger enough for us to see, then wait for a keypress
            cv2.imshow("ROI", imutils.resize(roi, width=28))
            key = cv2.waitKey(0)

            # If the '`' key is pressed, then ignore the character
            if key == ord("`"):
                print("[INFO]: Ignoring character....")
                continue

            # Grab the key that was pressed and construct the path the output directory
            key = chr(key).upper()
            dir_path = os.path.sep.join([args["annotation"], key])

            # If the output directory does not exist, create it
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            # Write the labeled character to file
            count = counts.get(key, 1)
            p = os.path.sep.join([dir_path, "{}.png".format(str(count).zfill(6))])
            cv2.imwrite(p, roi)

            # Increment the count for the current key
            counts[key] = count + 1

    # We are trying to control-c out of the script, so break from the loop (you still need to press a key for the
    # active window to trigger this)
    except KeyboardInterrupt:
        print("[INFO]: Manually leaving script")
        break
    # An unknown error has occurred for this particular image
    except:
        print("[INFO]: Skipping image...")
