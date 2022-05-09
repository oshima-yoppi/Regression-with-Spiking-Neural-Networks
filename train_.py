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
# from rectangle_builder import rectangle,test_img
import sys

sys.path.append(r"C:\Users\aki\Documents\GitHub\deep\pytorch_test\snu")
from model import snu_layer
from model import network
from model import loss
from tqdm import tqdm
#from mp4_rec import record, rectangle_record
import pandas as pd
# import scipy.io
# from torchsummary import summary
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--batch', '-b', type=int, default=32)
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
train_iter = DataLoader(train_dataset, batch_size=args.batch, shuffle=False)
test_iter = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
# print(train_iter.shape)
# ネットワーク設計
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 畳み込みオートエンコーダー　リカレントSNN　
model = network.SNU_Regression(num_time=args.time,l_tau=0.8,rec=args.rec, forget=args.forget, dual=args.dual, gpu=True, batch_size=args.batch)



model = model.to(device)
print("building model")
print(model.state_dict().keys())
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = args.epoch

loss_hist = []
for epoch in range(epochs):
    running_loss = 0.0
    local_loss = []
    print("EPOCH",epoch)
    # モデル保存
    if epoch == 0 :
        torch.save(model.state_dict(), "models/models_state_dict_"+str(epoch)+"epochs.pth")
        print("success model saving")
        print(model)
    # with tqdm(total=len(train_dataset),desc=f'Epoch{epoch+1}/{epochs}',unit='img')as pbar:
        # for i,(inputs, labels, name) in enumerate(train_iter, 0):
    for i ,(inputs, labels) in tqdm(enumerate(train_iter, 0)):
        

        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        torch.cuda.memory_summary(device=None, abbreviated=False)
        # loss, pred, _, iou, cnt = model(inputs, labels)
        output= model(inputs, labels)
        los = loss.compute_loss(output, labels)
        #iou = 各発火閾値ごとに連なり[??(i=1),??(i=2),,,,]

        torch.autograd.set_detect_anomaly(True)
        los.backward(retain_graph=True)
        running_loss += los.item()
        local_loss.append(los.item())
        del los
        optimizer.step()

        # print statistics
            

    
    with torch.no_grad():
        for i,(inputs, labels) in enumerate(test_iter, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs, labels)
            los = loss.compute_loss(output, labels)
            
            
    

  
    
    
    
    mean_loss = np.mean(local_loss) 
    print("mean loss",mean_loss)
    loss_hist.append(mean_loss)

# ログファイル二セーブ
path_w = 'loss_hist.txt'
with open(path_w, mode='w') as f:
    for i , x in enumerate(loss_hist):
        f.write(f"{i}: {x}\n")


torch.save(model.state_dict(), "models/models_state_dict_end.pth")
 # モデル読み込み
print("success model saving")