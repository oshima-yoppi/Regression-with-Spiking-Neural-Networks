import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
# from rectangle_builder import rectangle,test_img
import sys
sys.path.append(r"C:\Users\aki\Documents\GitHub\deep\pytorch_test\snu")
from model import snu_layer
from model import network
from tqdm import tqdm
# from mp4_rec import record, rectangle_record
from PIL import Image
import scipy.io
# from torchsummary import summary

class TrainLoadDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
       
        self.df = pd.read_csv(csv_file)
        #self.data_transform = data_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        file = self.df['id'][i]
        label = np.array(self.df['label'][i])
        image = scipy.io.loadmat(file)
        #print("label : ",label)
        #print("image : ",image['time_data'].shape)

        #label = torch.tensor(label, dtype=torch.float32)
        image = image['time_data']
        #image = image.reshape(4096,21) # flash 仕様
        image = image.reshape(1024,11264) # scan LiDAR　仕様
        #print("image : ",image.shape)
        image = image.astype(np.float32)
        label = label.astype(np.int64)
        label = torch.tensor(label,dtype =torch.int64 )
        #label = F.one_hot(label,num_classes=2)
        return image, label
     