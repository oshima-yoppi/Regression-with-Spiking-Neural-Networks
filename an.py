import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
# from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset
import os 
from tqdm import tqdm
# from rectangle_builder import rectangle,test_img
import sys
# sys.path.append(r"C:\Users\aki\Documents\GitHub\deep\pytorch_test\snu")
from model import snu_layer
from model import network
from tqdm import tqdm
#from mp4_rec import record, rectangle_record
import pandas as pd
# import scipy.io
# from torchsummary import summary
import argparse

class LoadDataset(Dataset):
    dir = 'C:/Users/oosim/Desktop/snn/v2e/output/'
    event_list = []
    for _, _, files in os.walk(dir):
        for file in files:
            if file.endswith('.h5')
                event_list.append(os.path.join(dir, file))
    
    def __init__(self, dir):