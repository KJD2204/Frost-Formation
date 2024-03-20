import cv2
import os
import numpy as np

# Path to the directory containing the .tif frames
frames_dir = './tif'

output_file = 'output_video.mp4'

# Get the list of .tif files in the frames directory
frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.tif')]

# Sort the frame files in ascending order
frame_files.sort()

# Get the first frame to determine the video dimensions
first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]), -1)
first_frame = cv2.convertScaleAbs(first_frame, alpha=(255.0/16383.0))
height, width = first_frame.shape

# Create a VideoWriter object to write the frames into an mp4 video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_file, fourcc, 1, (width, height), isColor=False)

# Iterate over each frame file and write it to the video
for frame_file in frame_files:
    frame_path = os.path.join(frames_dir, frame_file)
    frame = cv2.imread(frame_path, -1)
    frame = cv2.convertScaleAbs(frame, alpha=(255.0/16383.0))

    # Adjust contrast and brightness
    frame = cv2.convertScaleAbs(frame, alpha=1, beta=25)  # Change alpha (contrast) and beta (brightness) as needed

    video_writer.write(frame)

# Release the VideoWriter object and close the video file
video_writer.release()