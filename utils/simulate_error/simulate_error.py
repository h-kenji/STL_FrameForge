import cv2
import numpy as np
import os

# Parameters
input_video_path = "input_video.mp4"
output_video_path = "input_video_with_black_frames.mp4"
replace_start_frame = 190  # First frame to replace with black
num_black_frames = 5       # Number of black frames to replace

def replace_with_black_frames(input_path, output_path, start_frame, black_frames_count):
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening input video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)  # Create a black frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Replace specified frames with black frames
        if start_frame <= frame_count < start_frame + black_frames_count:
            out.write(black_frame)
        else:
            out.write(frame)

        frame_count += 1

    # Release resources
    cap.release()
    out.release()
    print(f"Output video saved as {output_path}")

# Run the function
replace_with_black_frames(input_video_path, output_video_path, replace_start_frame, num_black_frames)
