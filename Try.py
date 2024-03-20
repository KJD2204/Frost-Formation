import cv2
import numpy as np
import glob
import os

# Define the input and output folders
input_folder = 'tif'
output_image_folder = 'annotated_images'
output_text_folder = 'text_files'

# Create the output folders if they don't exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_text_folder, exist_ok=True)

# Get a list of all .tif files in the input folder
files = glob.glob(os.path.join(input_folder, '*.tif'))

# Process each file
for file in files:
    # Load the image
    image = cv2.imread(file, -1)

    # Convert the image from 14-bit to 8-bit
    image = cv2.convertScaleAbs(image, alpha=(255.0/16383.0))

    # Threshold the image to make it binary
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Define the minimum area for a spot to be considered
    min_area = 100  # Change this value to suit your needs

    # Define the aspect ratio tolerance
    tolerance = 0.25  # Change this value to suit your needs

    # Define the minimum average intensity for a spot to be considered
    min_intensity = 160  # Change this value to suit your needs

    # List to store the rectangles
    rectangles = []

    # Open the output file
    with open(os.path.join(output_text_folder, os.path.basename(file) + '.txt'), 'w') as f:
        # Check if any contours were found
        if contours:
            # Iterate over all contours
            for contour in contours:
                # Only consider contours that are larger than the minimum area
                if cv2.contourArea(contour) > min_area:
                    # Get the bounding rectangle for each contour
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate the aspect ratio
                    aspect_ratio = float(w)/h

                    # Only consider contours that are approximately circular
                    if abs(1 - aspect_ratio) <= tolerance:
                        # Check the average intensity of the spot
                        spot = image[y:y+h, x:x+w]
                        if np.mean(spot) >= min_intensity:
                            # Check if the rectangle overlaps with any existing rectangle
                            for rx, ry, rw, rh in rectangles:
                                if x < rx + rw and x + w > rx and y < ry + rh and y + h > ry:
                                    break
                            else:
                                # If no overlap, draw the rectangle and add it to the list
                                image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                rectangles.append((x, y, w, h))

                                # Write the rectangle information to the file
                                f.write(f'{x + w/2} {y + h/2} {w} {h}\n')

        # Save the annotated image
        cv2.imwrite(os.path.join(output_image_folder, os.path.basename(file)), image)