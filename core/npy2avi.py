import numpy as np
import cv2 

def generate_video_from_npy(
    npy_file,
    output_video_path,
    index=0,
    fps=24,
    vmax=0.6,
    vmin=0.0,
    cmap='gray',
    use_rgb=True,
):
    """
    Generate an AVI video from a .npy file containing spatio-temporal data.

    Parameters:
        npy_file (str): Path to the .npy file.
        output_video_path (str): Path to save the .avi video.
        index (int): Index of the sequence in the .npy file (if multiple sequences are stored).
        fps (int): Frames per second for the output video.
        vmax (float): Maximum value for normalization (for grayscale).
        vmin (float): Minimum value for normalization (for grayscale).
        cmap (str): Color map for grayscale rendering (ignored if use_rgb=True).
        use_rgb (bool): Whether the frames are RGB or grayscale.
    """
    # Load data from .npy file
    data = np.load(npy_file)
    print(f"Loaded data shape: {data.shape}")

    # Select sequence if multiple are stored
    if len(data.shape) > 4:
        data = data[index]
    print(f"Selected sequence shape: {data.shape}")

    # Ensure data shape is correct
    if use_rgb:
        # RGB frames expected: (frames, height, width, 3)
        data = data.transpose(0, 2, 3, 1)  # Reorder to (frames, height, width, channels)
    else:
        # Grayscale frames expected: (frames, height, width)
        data = data[:, 0, :, :]  # Take the first channel for grayscale

    print(f"Processed data shape: {data.shape}")

    # Validate dimensions
    height, width = data.shape[1], data.shape[2]
    assert height > 2 and width > 2, f"Invalid dimensions: {height}x{width}"

    # Adjust dimensions for video encoding
    if height % 2 != 0 or width % 2 != 0:
        print("Adjusting frame dimensions to be divisible by 2 for video encoding.")
        height, width = (height // 2) * 2, (width // 2) * 2
        data = data[:, :height, :width] if not use_rgb else data[:, :height, :width, :]

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height),
        isColor=use_rgb  # Match the color format
    )

    # Write each frame to the video
    for i, frame in enumerate(data):
        if use_rgb:
            # Ensure the frame is normalized to [0, 255] and converted to uint8
            #frame_normalized = (np.clip(frame, 0, 1) * 255).astype(np.uint8) if frame.dtype != np.uint8 else frame
            frame_normalized = (np.clip((frame - vmin) / (vmax - vmin), 0, 1) * 255).astype(np.uint8)
            # Convert to BGR for OpenCV
            # frame_bgr = cv2.cvtColor(frame_normalized, cv2.COLOR_RGB2BGR) # Uncomment for BGR output
            # frame_bgr = cv2.cvtColor(frame_normalized, cv2.COLOR_BGR2RGB)
            frame_bgr = frame_normalized
        else:
            # Normalize grayscale frame to [0, 255] and convert to uint8
            normalized_frame = (np.clip((frame - vmin) / (vmax - vmin), 0, 1) * 255).astype(np.uint8)
            # Convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(normalized_frame, cv2.COLOR_GRAY2BGR)

        # Ensure frame dimensions match
        assert frame_bgr.shape[:2] == (height, width), f"Frame {i} shape mismatch: {frame_bgr.shape[:2]} != {(height, width)}"
        video_writer.write(frame_bgr)

    # Release video writer
    video_writer.release()
    print(f"Video saved to {output_video_path}")


""" # Usage example:
generate_video_from_npy(
    npy_file='../work_dirs/custom_exp/saved/preds.npy',
    output_video_path='preds.avi',
    index=0,
    fps=24,
    vmax=1,
    vmin=0.0,
    cmap='gray',
    use_rgb=True  # Set to False if data is grayscale
)
"""