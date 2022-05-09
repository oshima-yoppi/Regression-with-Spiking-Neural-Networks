import numpy as np
import torch
def cross_entropy_error(y, t):
    """
    交差エントロピー誤差
    :param y: output
    :param t: label
    """
    # log 0回避用の微小な値を作成
    delta = 1e-7
    
    # 交差エントロピー誤差を計算:式(4.2)
    return - torch.sum(t * torch.log(y + delta))
def compute_loss(input: torch.Tensor, label):
    """
    :param input: tensor of shape (batch, 3, time)
    :param target: tensor of shape (batch, 3, time)
    :param time_start_idx: Time-index from which to start computing the loss
    :return: loss
    """
    # assert len(input_.shape) == 3
    # assert len(target.shape) == 3
    # assert input_.shape == target.shape
    # assert input_.shape[1] == 3
    label = 



    # if time_start_idx:
    #     input_ = input_[..., time_start_idx:]
    #     target = target[..., time_start_idx:]
    return cross_entropy_error(input, torch.tensor([t, 1-t]))
    # return torch.mean(torch.sqrt(torch.sum((input - label) ** 2, dim = 1)))

if __name__ == "__main__":
    a = torch.tensor([0.2, 0.8])
    b = torch.tensor([0.2,0.8])
    c = compute_loss(a,b)
    print(c)