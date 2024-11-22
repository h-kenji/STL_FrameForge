import time
start_time = time.time()

import argparse
import pickle
from core.video_analyser import process_video, bad_frame, sample_frames, process_folder
from core.dummy_videos import create_dummy_video
from core.npy2avi import generate_video_from_npy
from core.out_rebuilder import replace_black_frames
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset

#from core.read_spec_file import read_specs
'''parser = argparse.ArgumentParser(description="Process input via arguments and a specs file.")
parser.add_argument("--specs", type=str, help="Path to the specs file")
parser.add_argument("--input", type=str, required=True, help="Path to the input file")
parser.add_argument("--model", type=str, help="Model file (overrides specs file, default: 'best.ckpt')")
parser.add_argument("--pre_seq_lenght", type=int, help="Pre-sequence length (overrides specs file)")
parser.add_argument("--aft_seq_lenght", type=int, help="After-sequence length (overrides specs file)")
args = parser.parse_args()

specs = {}
if args.specs:
    specs = read_specs(args.specs)

input_file = args.input or specs.get("input")
model = args.model or specs.get("model", "best.ckpt")
X = args.pre_seq_lenght or int(specs.get("pre_seq_lenght", 0))
Y = args.aft_seq_lenght or int(specs.get("aft_seq_lenght", 0))
'''
input_file='input.mp4'
X=20
Y=10
model='best.ckpt'

print("=" * 50)
print(" STL FrameForge - Video Reconstructor")
print("=" * 50)
print(f"Input file      : {input_file}")
print(f"Model           : {model}")
print(f"Pre-seq length  : {X}")
print(f"Aft-seq length  : {Y}")
print("=" * 50)

train_dummy = "auxiliary_dirs/train_dummy"
val_dummy = "auxiliary_dirs/val_dummy"

# Input video analysis
process_video(input_file,X,Y,'test')

corrupted_vid=f'test/corrupted_vid.avi'

# Dummy videos creation
create_dummy_video(corrupted_vid, val_dummy, X, Y)
create_dummy_video(corrupted_vid, train_dummy,  X, Y)

# Set the precision for matrix multiplications to utilize Tensor Cores
torch.set_float32_matmul_precision('high')

pre_seq_length = X
aft_seq_length = Y

train_dir = train_dummy
val_dir = val_dummy
test_dir = 'test'

class CustomDataset(Dataset):
    def __init__(self, X, Y, normalize=False, data_name='custom'):
        super(CustomDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.mean = None
        self.std = None
        self.data_name = data_name

        if normalize:
            # get the mean/std values along the channel dimension
            mean = data.mean(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            std = data.std(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            data = (data - mean) / std
            self.mean = mean
            self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index]).float()
        labels = torch.tensor(self.Y[index]).float()
        return data, labels

dataset = {}
folders = [train_dir, val_dir, test_dir]
for folder in folders:
    data_x, data_y = process_folder(folder, pre_slen=pre_seq_length, aft_slen=aft_seq_length, suffix='.avi')
    dataset['X_' + folder], dataset['Y_' + folder] = data_x, data_y

# save as a pkl file
with open('dataset_test.pkl', 'wb') as f:
    pickle.dump(dataset, f)

# load the dataset
with open('dataset_test.pkl', 'rb') as f:
    dataset = pickle.load(f)

test_x, test_y = dataset['X_test'], dataset['Y_test']
print(test_x.shape)
# the shape is B x T x C x H x W
# B: the number of samples
# T: the number of frames in each sample
# C, H, W: the channel, height, width of each frame

X_train, X_val, X_test, Y_train, Y_val, Y_test = dataset[f'X_{train_dir}'], dataset[
    f'X_{val_dir}'], dataset[f'X_{test_dir}'], dataset[f'Y_{train_dir}'], dataset[f'Y_{val_dir}'], dataset[f'Y_{test_dir}']

train_set = CustomDataset(X=X_train, Y=Y_train)
val_set = CustomDataset(X=X_val, Y=Y_val)
test_set = CustomDataset(X=X_test, Y=Y_test)

dataloader_train = torch.utils.data.DataLoader(
    train_set, batch_size=1, shuffle=True, pin_memory=True)
dataloader_val = torch.utils.data.DataLoader(
    val_set, batch_size=1, shuffle=True, pin_memory=True)
dataloader_test = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=True, pin_memory=True)

custom_training_config = {
    'pre_seq_length': pre_seq_length,
    'aft_seq_length': aft_seq_length,
    'total_length': pre_seq_length + aft_seq_length,
    'batch_size': 1,
    'val_batch_size': 1,
    'epoch': 1,
    'lr': 0.001,   
    'metrics': ['mse', 'mae'],
    'test': True,
    'ex_name': 'custom_exp',
    'dataname': 'custom',
    'in_shape': [pre_seq_length, test_x.shape[2], test_x.shape[3], test_x.shape[4]], #pre_seq_length, channels, height, widht
}

custom_model_config = {
    # For MetaVP models, the most important hyperparameters are: 
    # N_S, N_T, hid_S, hid_T, model_type
    'method': 'SimVP',
    # Users can either using a config file or directly set these hyperparameters 
    # 'config_file': 'configs/custom/example_model.py',
    
    # Here, we directly set these parameters
    'model_type': 'gSTA',
    'N_S': 4,
    'N_T': 8,
    'hid_S': 64,
    'hid_T': 256
}

from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser

args = create_parser().parse_args([])
config = args.__dict__

# update the training config
config.update(custom_training_config)
# update the model config
config.update(custom_model_config)
# fulfill with default values
default_values = default_parser()
for attribute in default_values.keys():
    if config[attribute] is None:
        config[attribute] = default_values[attribute]

exp = BaseExperiment(args, dataloaders=(dataloader_train, dataloader_val, dataloader_test), strategy='auto')

print('>'*35 + ' testing  ' + '<'*35)
exp.test()

generate_video_from_npy(
    npy_file='work_dirs/custom_exp/saved/preds.npy',
    output_video_path='utils/preds.avi',
    index=0,
    fps=24,
    vmax=1,
    vmin=0.0,
    cmap='gray',
    use_rgb=True  # Set to False if data is grayscale
)

replace_black_frames(input_file, 'utils/preds.avi', 'output.avi')


print(f"Execution time: {time.time() - start_time:.2f} seconds")