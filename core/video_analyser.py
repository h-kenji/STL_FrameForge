import cv2
import numpy as np
import os
from collections import deque
from core.frame_analysis import analyze_frame

def bad_frame(last_frames,Y,output_folder):
    """Takes last good frames, adds Y black frames, and saves them as an .avi file."""
    height, width, layers = last_frames[0].shape
    output_folder = "test"  # For practical purposes, output goes straight to the model's test directory
    os.makedirs(output_folder, exist_ok=True)
    video_name = os.path.join(output_folder, "corrupted_vid.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_name, fourcc, 20.0, (width, height))

    # Write last good frames
    for frame in last_frames:
        out.write(frame)

    # Add Y black frames. Note that there's no need to then be specificaly black, we just need Y frames to constitute the Y_test split in the model.
    black_frame = np.zeros_like(last_frames[0])
    for _ in range(Y):
        out.write(black_frame)

    out.release()
    print(f"Output video saved as {video_name}")


def process_video(video_path,X,Y,output_folder):
    """Process the video frame by frame and stops after the first black frame is detected."""
    cap = cv2.VideoCapture(video_path)

    # Initialize the FIFO queue
    frame_queue = deque(maxlen=X)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    frame_index = 0
    bad_frame_found = False
    while cap.isOpened() and not bad_frame_found:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the frame
        if analyze_frame(frame):
            bad_frame(list(frame_queue),Y,output_folder)  # Call bad_frame with the last X good frames
            bad_frame_found = True  # Stop processing after the first bad frame is handled
        else:
            # Add current frame to the FIFO queue
            frame_queue.append(frame)

        frame_index += 1