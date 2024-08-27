import pdb
import torch as th
import math
import numpy as np
import torch
from run_on_video.video_loader import VideoLoader
from torch.utils.data import DataLoader
import argparse
from run_on_video.preprocessing import Preprocessing
import torch.nn.functional as F
from tqdm import tqdm
import os
import sys
from run_on_video import clip
import argparse
import ipdb

dataset = VideoLoader(
    vid_path,
    framerate=1/clip_len,
    size=224,
    centercrop=True,
    overwrite=overwrite,
    model_version=model_version
)
n_dataset = len(dataset)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=num_decoding_thread,
    sampler=None,
)