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

Certify that your workspace follows the directory structure outlined above so the scripts can run properly.

```
OpenSTL/
├── STL_FrameForge/                     # Main directory for STL Frame Forge
│   ├── train.py                        # Script for training the model
│   ├── reconstruct.py                  # Script for video signal reconstruction
│   ├── core/                           # Configuration files for the project
│   │   ├── __init__.py                 # Pretrained models or model definitions
│   │   ├── dummy_videos.py             # Directory to store datasets
│   │   ├── frame_analysis.py           # Helper functions and utilities
│   │   ├── npy2avi.py                  # Helper functions and utilities
│   │   ├── out_rebuilder.py            # Helper functions and utilities
│   │   ├── read_spec_file.py           # Helper functions and utilities
│   │   └── video_analyser.py           # Outputs of training and reconstruction
│   ├── prepare_test_data/              # Configuration files for the project
│   │   └── analyser.py                 # Outputs of training and reconstruction
│   └── utils/                          # Configuration files for the project
│       ├── compare.py                  # Helper functions and utilities
│       ├── prepare_train_data/         # Helper functions and utilities
│       │   ├── segments                # Helper functions and utilities
│       │   |   └── data_split.py       # Helper functions and utilities
│       |   └── slice_n_frames.py       # Helper functions and utilities
│       └── simulate_error/             # Helper functions and utilities
│           ├── simulate_error.py       # Helper functions and utilities
│           └── simulate_error.py       # Helper functions and utilities
...
```
 
## Usage
### Training
Run **`train.py`** to train the model.
### Reconstruction: Use reconstruct.py for complete signal reconstruction.
Ensure that the environment is activated before executing any scripts:

```bash
conda activate OpenSTL
```
