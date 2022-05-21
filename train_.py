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
import time


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
model = network.SNU_Regression(num_time=args.time,l_tau=0.8, soft =False, rec=args.rec, forget=args.forget, dual=args.dual, gpu=True, batch_size=args.batch)



model = model.to(device)
print("building model")
print(model.state_dict().keys())
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = args.epoch
before_loss = None
loss_hist = []
test_hist = []
try:

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        local_loss = []
        test_loss = []
        print("EPOCH",epoch)
        # モデル保存
        if epoch == 0 :
            torch.save(model.state_dict(), "models/models_state_dict_"+str(epoch)+"epochs.pth")
            print("success model saving")
            print(model)
        # with tqdm(total=len(train_dataset),desc=f'Epoch{epoch+1}/{epochs}',unit='img')as pbar:
            # for i,(inputs, labels, name) in enumerate(train_iter, 0):
        for i ,(inputs, labels) in tqdm(enumerate(train_iter, 0)):
            # if i < 74:
            #     continue
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            torch.cuda.memory_summary(device=None, abbreviated=False)
            # loss, pred, _, iou, cnt = model(inputs, labels)
            output= model(inputs, labels)

            # print(f"output.shape:{output.shape}")###torch.Size([32, 100])
            # kazu = torch.count_nonzero(inputs[0,1,:, :,:,] == 1.)
            # print(f"sssssssssssssssssssss{kazu}")
            print(output)
            los = loss.compute_loss(output, labels)
            print(f'label:{labels[0,0]}, {labels[1,0]}')
            print(f'epoch:{epoch+1}  loss:{los}') # 
            print(f'before_loss:{before_loss}') ## 一個前のepoch loss 
            torch.autograd.set_detect_anomaly(True)
            los.backward(retain_graph=True)
            running_loss += los.item()
            local_loss.append(los.item())
            del los
            optimizer.step()
            

            # print statistics
                

        
        with torch.no_grad():
            for i,(inputs, labels) in enumerate(test_iter, 0):
                # if i == 2:
                #     break
                inputs = inputs.to(device)
                labels = labels.to(device)
                output = model(inputs, labels)
                los = loss.compute_loss(output, labels)
                test_loss.append(los.item())
                
                
        

    
        
        
        
        mean_loss = np.mean(local_loss) 
        before_loss = mean_loss
        print("mean loss",mean_loss)
        loss_hist.append(mean_loss)

        test_mean_loss = np.mean(test_loss) 
        test_hist.append(test_mean_loss)
except:
    traceback.print_exc()
    pass
    
# ログファイル二セーブ
path_w = 'loss_hist.txt'
with open(path_w, mode='w') as f:
    now = datetime.datetime.now()
    f.write(f'{now}\n')
    for i , x in enumerate(loss_hist):
        f.write(f"{i}: {x}\n")

##　最後の出力結果の確認用
print(output)




###ログのグラフ
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(loss_hist)
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss_hist')
ax2.plot(test_hist)
ax2.set_xlabel('epoch')
ax2.set_ylabel('test_hist')
plt.show()




torch.save(model.state_dict(), "models/models_state_dict_end.pth")
 # モデル読み込み
print("success model saving")