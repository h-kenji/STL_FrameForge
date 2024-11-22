import cv2
import numpy as np
from core.frame_analysis import analyze_frame




def replace_black_frames(input_video_path, preds_video_path, output_video_path):
    """Replace black frames in input_video with corresponding frames from preds.avi."""
    # Open the input video
    input_cap = cv2.VideoCapture(input_video_path)
    if not input_cap.isOpened():
        print("Error: Could not open input video.")
        return

    # Open the predictions video
    preds_cap = cv2.VideoCapture(preds_video_path)
    if not preds_cap.isOpened():
        print("Error: Could not open preds video.")
        return

    # Get video properties
    fps = int(input_cap.get(cv2.CAP_PROP_FPS))
    width = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    # Setup video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    input_frame_index = 0
    preds_frame_index = 0

    while input_cap.isOpened():
        ret, input_frame = input_cap.read()
        if not ret:
            break  # End of input video

        if analyze_frame(input_frame):
            # Replace black frame with frame from preds.avi
            preds_ret, preds_frame = preds_cap.read()
            if preds_ret:
                out.write(preds_frame)
                preds_frame_index += 1
            else:
                # If no more frames in preds.avi, write the original black frame
                out.write(input_frame)
                print("Warning: No more frames in preds.avi, using black frame instead.")
        else:
            # Write original frame if not black
            out.write(input_frame)

        input_frame_index += 1

    # Release resources
    input_cap.release()
    preds_cap.release()
    out.release()
    print(f"Output video saved as {output_video_path}")

# Usage example
input_video_path = "../raw_test/input_video_with_black_frames.mp4"
preds_video_path = "preds.avi"
output_video_path = "output_video.avi"

replace_black_frames(input_video_path, preds_video_path, output_video_path)
