# This script works for splitting up a dataset into training, testing, and validation sets. In this particular case, we're going to comment out the test set 
# and only split the dataset into training and validation sets. As we are working with video prediction, the test split will be a separeted mannualy by taking
# out a continuous part of the end of the full video.

import os
import random
import shutil

# Define paths for the output directories
train = '../../../train'
#test = 
val = '../../../val'

# Define video extensions
ext = ".avi"


# Set the path to your source directory containing the .avi files
source_dir = './' # Script is in the same directory as the videos
# Define paths for the output directories
train_dir = os.path.join(source_dir, train)
#test_dir = os.path.join(source_dir, test)
val_dir = os.path.join(source_dir, val)

# Create output directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
#os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# List all video files in the source directory
all_files = [f for f in os.listdir(source_dir) if f.endswith(ext)]
# Shuffle files randomly
random.shuffle(all_files)

# Calculate split sizes
train_size = int(0.8 * len(all_files))
#test_size = int(0.1 * len(all_files))
val_size = len(all_files) - train_size #- test_size

# Split files into train, test, and val sets
train_files = all_files[:train_size]
#test_files = all_files[train_size:train_size + test_size]
#val_files = all_files[train_size + test_size:]
val_files = all_files[train_size:]

# Move files to their respective directories
for f in train_files:
    shutil.move(os.path.join(source_dir, f), os.path.join(train_dir, f))

#for f in test_files:
#    shutil.move(os.path.join(source_dir, f), os.path.join(test_dir, f))

for f in val_files:
    shutil.move(os.path.join(source_dir, f), os.path.join(val_dir, f))

#print("Files have been split into train, test and val folders.")
print("Files have been split into train and val folders.")

