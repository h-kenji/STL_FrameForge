# STL Frame Forge  

## Overview  
**STL Frame Forge** is a project aimed at reconstructing video signals using the OpenSTL library and additional Python scripts. The project provides two main scripts:  
1. **`train.py`**: For training the model.  
2. **`reconstruct.py`**: For performing complete signal reconstruction.  

Since the project is built on **OpenSTL**, its setup process aligns with OpenSTL's environment configuration using **Conda**.

## Installation  

To set up the project environment, follow these steps:

### Step 1: Install OpenSTL  
Use the installation instructions provided in the [OpenSTL GitHub repository](https://github.com/chengtan9907/OpenSTL) to configure the base environment:  

```
git clone https://github.com/chengtan9907/OpenSTL
cd OpenSTL
conda env create -f environment.yml
conda activate OpenSTL
python setup.py develop
```
### Step 2: Set Up STL Frame Forge
Clone the STL Frame Forge repository into the OpenSTL directory:

```
cd OpenSTL
git clone https://github.com/h-kenji/STL_FrameForge
```
Ensure the environment is properly set up, as all dependencies required by STL Frame Forge are included in the Conda environment defined by OpenSTL.

Certify that your workspace follows the directory structure outlined above so the scripts can run properly as it is.

```
OpenSTL/
├── STL_FrameForge/                     # Main directory for STL Frame Forge
│   ├── README.md                       # This file
│   ├── train.py                        # Script for training the model
│   ├── reconstruct.py                  # Script for video signal reconstruction
│   ├── train.ipynb                     # Notebook for running the train script on a Jupyter Kernel
│   ├── specs.txt                       # Specifications for training and reconstruct scripts (currently not in use)
│   ├── core/                           # Core modules for main scripts functionality
│   │   ├── __init__.py                 # Initializes the `core` module as a package
│   │   ├── dummy_videos.py             # Dummy video generator
│   │   ├── frame_analysis.py           # Frame analysis 
│   │   ├── npy2avi.py                  # Converts .npy files to .avi format
│   │   ├── out_rebuilder.py            # Rebuilds output files for post-processing
│   │   ├── read_spec_file.py           # Reads specification files (currently not in use)
│   │   └── video_analyser.py           # Performs frame by frame video analysis
│   ├── prepare_test_data/              # Prepares data for testing
│   │   └── analyser.py                 # Analyzes test data before reconstruction or analysis
│   └── utils/                          # Utility scripts for preparing train data, comparing reconstruct video and simulating error in video
│       ├── compare.py                  # Compares reconstruct video with corrupted one
│       ├── prepare_train_data/         # Training data preparation
│       │   ├── segments/               # Contains tools for segmenting training data
│       │   │   └── data_split.py       # Splits training data and dumps it into train and val dirs
│       │   └── slice_n_frames.py       # Slices training input in slices with N frames and dumps into segments dir
│       └── simulate_error/             # Simulates errors for testing
│           └── simulate_error.py       # Substitutes specified normal frames with black frames in a video
...
```
 
## Usage
### Training
Before training the model with **`train.py`**, you need to prepare the training data.
1. 
### Reconstruction: Use reconstruct.py for complete signal reconstruction.
Ensure that the environment is activated before executing any scripts:

```bash
conda activate OpenSTL
```
