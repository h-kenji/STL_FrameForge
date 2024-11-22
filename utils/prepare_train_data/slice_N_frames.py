import cv2
import os

# Define input video file, output directory and desired frames per segment
input_video = "input.mp4"
N = 30  # Number of frames per segment
output_dir = "segments"

# Create a VideoCapture object to read the video
cap = cv2.VideoCapture(input_video)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get and print the video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Input video (" + input_video + ") properties:\n")
print("FPS= " + str(fps))
print("\nResolution= " + str(width) + "x" + str(height))
print("\nTotal Frames= " + str(total_frames))

# Directory to save output segments
os.makedirs(output_dir, exist_ok=True)

part = 1
frame_count = 0
out = None

# Loop through frames and slice into segments
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Start a new video writer for each segment
    if frame_count % N == 0:
        if out:
            out.release()  # Release the previous segment writer
        segment_path = os.path.join(output_dir, f"output_part_{part}.avi")
        out = cv2.VideoWriter(segment_path, fourcc, fps, (width, height))
        part += 1

    # Write the frame to the current segment
    out.write(frame)
    frame_count += 1

# Release resources
if out:
    out.release()
cap.release()
print("Video slicing completed.")
