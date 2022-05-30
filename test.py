import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from data import LoadDataset
import os 
from tqdm import tqdm
import datetime
# from rectangle_builder import rectangle,test_img
import traceback

from model import snu_layer
from model import network
from model import loss
from tqdm import tqdm
# from torchsummary import summary
import argparse
import time

start_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--batch', '-b', type=int, default=7)
parser.add_argument('--epoch', '-e', type=int, default=10)##英さんはepoc100だった
parser.add_argument('--time', '-t', type=int, default=100,
                        help='Total simulation time steps.')
parser.add_argument('--rec', '-r', action='store_true' ,default=False)  # -r付けるとTrue                  
parser.add_argument('--forget', '-f', action='store_true' ,default=False) 
parser.add_argument('--dual', '-d', action='store_true' ,default=False)
args = parser.parse_args()


print("***************************")
train_dataset = LoadDataset(dir = 'C:/Users/oosim/Desktop/snn/v2e/output/', which = "train" ,time = args.time)
test_dataset = LoadDataset(dir = 'C:/Users/oosim/Desktop/snn/v2e/output/', which = "test" ,time = args.time)
data_id = 2
# print(train_dataset[data_id][0]) #(784, 100) 
train_iter = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=args.batch, shuffle=True)
# print(train_iter.shape)
# ネットワーク設計
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 畳み込みオートエンコーダー　リカレントSNN　
# model = network.SNU_Regression(num_time=args.time,l_tau=0.8, soft =False, rec=args.rec, forget=args.forget, dual=args.dual, gpu=True, batch_size=args.batch)
model = network.Conv4Regression(num_time=args.time,l_tau=0.8, soft =False, rec=args.rec, forget=args.forget, dual=args.dual, gpu=True, batch_size=args.batch)
model_path = 'models/2.pth'
model.load_state_dict(torch.load(model_path))


model = model.to(device)
print("building model")
print(model.state_dict().keys())
epochs = args.epoch
before_loss = None
loss_hist = []
test_hist = []
test_loss = []
over = {}
th = 5
for i in range(300 // th ):
    over[i] = []
    over[-i] = []

try:    
    with torch.no_grad():
        for i,(inputs, labels) in tqdm(enumerate(test_iter, 0), total=len(test_dataset)):
            # if i == 2:
            #     break
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs, labels)
            los = loss.compute_loss(output, labels)
            test_loss.append(los.item())
            # if labels[:,0].item() // th == -6:
                # print(labels[:,0].item())
            over[int(labels[:,0].item() / th)].append(los.item())
    

except:
    traceback.print_exc()
    pass



for key in over.keys():
    over[key] = np.mean(over[key])
print(over)





###ログのグラフ

def sqrt_(n):
    return n ** 0.5
def double(n):
    return n * th

over = over.items()
over = sorted(over)
x,y = zip(*over)
x = list(map(double, x))
y = list(map(sqrt_, y))
plt.plot(x, y)
plt.xlabel('angular velocity')
plt.ylabel('loss **1')
plt.title('model:' + model_path)
plt.show()
    
 



