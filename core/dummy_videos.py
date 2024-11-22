import cv2
import numpy as np
import os

def create_dummy_video(input_video_path, dummy_dir, X, Y):
    """
    Create a dummy video with X + Y black frames and save the same video into a folder.

    Parameters:
        input_video_path (str): Path to the input video to extract properties.
        dummy_dir (str): Path to the folder to save the dummy video.
        X (int): Number of good frames.
        Y (int): Number of black frames.
    """
    # Ensure folders exist
    os.makedirs(dummy_dir, exist_ok=True)

    # Open the input video to extract properties
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Extract properties from the input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Read X frames from the input video
    good_frames = []
    for _ in range(X):
        ret, frame = cap.read()
        if not ret:
            print(f"Not enough frames in input video. Only {len(good_frames)} frames read.")
            break
        good_frames.append(frame)

    cap.release()

    # Create the output video paths
    out_video_path = os.path.join(dummy_dir, f"dummy.avi")

    # Define the codec and create the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (frame_width, frame_height))


    # Add X+Y black frames
    black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    for _ in (range(X+Y)):
        out.write(black_frame)

    # Release the VideoWriter
    out.release()


    print(f"Dummy video saved in "+ dummy_dir +" folder.")


    cap.release()