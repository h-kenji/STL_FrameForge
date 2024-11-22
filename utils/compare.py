'''Compares Ground Truth Video (corrupted) with Reconstructed Video (output)'''
import cv2

def combine_videos_with_labels(video1_path, video2_path, label1, label2, output_path):
    # Open the two video files
    video1 = cv2.VideoCapture(video1_path)
    video2 = cv2.VideoCapture(video2_path)

    # Check if the videos opened successfully
    if not video1.isOpened() or not video2.isOpened():
        print("Error: Unable to open one or both video files.")
        return

    # Get the frame width, height, and frame rate of the videos
    width1 = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set the height for both videos to match (resize the taller one if necessary)
    max_height = max(height1, height2)
    
    # Adjust scaling for low resolution videos
    # We use a fixed scale factor for small resolutions to keep the font size and padding reasonable
    scaling_factor = max_height / 128  # Assuming 128px is the largest expected height
    
    # Calculate dynamic font size and padding size
    font_scale = max(0.3, scaling_factor * 0.1)  # Adjust the font size to be smaller
    font_thickness = max(1, int(scaling_factor))  # Font thickness scales slightly for clarity
    label_padding = max(10, int(scaling_factor * 20))  # Small padding for labels at lower resolutions

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'MP4V' for .mp4 format
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width1 + width2, max_height + label_padding))  # Space for label area

    while True:
        # Read frames from both videos
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        # If either video reaches the end, stop
        if not ret1 or not ret2:
            break

        # Resize frames to match the max height while maintaining aspect ratio
        frame1_resized = cv2.resize(frame1, (width1, max_height))
        frame2_resized = cv2.resize(frame2, (width2, max_height))

        # Create a black canvas with space for the labels
        frame1_with_label = frame1_resized.copy()
        frame2_with_label = frame2_resized.copy()

        # Create black frames (for padding) and put labels below
        black_frame1 = cv2.copyMakeBorder(frame1_with_label, 0, label_padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        black_frame2 = cv2.copyMakeBorder(frame2_with_label, 0, label_padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Add text labels below the videos
        cv2.putText(black_frame1, label1, (10, max_height + int(label_padding / 2)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(black_frame2, label2, (10, max_height + int(label_padding / 2)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        # Concatenate the two frames side by side
        combined_frame = cv2.hconcat([black_frame1, black_frame2])

        # Write the combined frame to the output video
        out.write(combined_frame)

    # Release the video objects
    video1.release()
    video2.release()
    out.release()

    print("Video saved successfully:", output_path)

# Usage
video1_path = "../output.avi"
video2_path = "../input.mp4"
label1 = "Rec"
label2 = "GT"
output_path = "combined_output.mp4"
combine_videos_with_labels(video1_path, video2_path, label1, label2, output_path)