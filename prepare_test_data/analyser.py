import cv2
import numpy as np
import os
from collections import deque
from core.frame_analysis import analyze_frame
from core.dummy_videos import create_dummy_video
from core.video_analyser import bad_frame, process_video

# Define parameters
X = 20  # Number of previous good frames to store
Y = 10  # Number of black frames to add after detecting a bad frame


# Define paths
train_dummy = "../auxiliary_dirs/train_dummy"
val_dummy = "../auxiliary_dirs/val_dummy"
input_video_path = "input_video_with_black_frames.mp4"
output_folder = "../test"  # For practical purposes, output should preferably go straight to the model's test directory
os.makedirs(output_folder, exist_ok=True)

# Input video analysis
process_video(input_video_path,X,Y,output_folder)

# Dummy videos creation
create_dummy_video(input_video_path, val_dummy, X, Y)
create_dummy_video(input_video_path, train_dummy,  X, Y)

