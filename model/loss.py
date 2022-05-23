
import torch
def compute_loss(input: torch.Tensor, label:torch.Tensor):
    """
    :param input: tensor of shape (batch, time , 1)
    :param target: tensor of shape (batch, 3)
    :param time_start_idx: Time-index from which to start computing the loss
    :return: loss
    """
    # assert len(input_.shape) == 3
    # assert len(target.shape) == 3
    # assert input_.shape == target.shape
    # assert input_.shape[1] == 3
    # print(label.shape)
    # print(label)
    # print(label[0,0].item())
    # s =(input - label[0]) ** 2
    # print("3333333333333333")
    # print(s)
    # print(input.shape)#torch.Size([batch size, 100])
    # print(len(input[-1])) # 100
    ### ３３％超えてからロスの平均計算するようにした。
    # input = input[:, len(input[-1]) // 3:]
    input = input[:, 8:]
    input = torch.mean(input, dim = 1)
    # print(f'平均だよ {input}')
    label = label[:,0]
    # print(f'labelだよ{label}')
    # print('input.shape', input.shape)
    # print('label.shape', label.shape)
    # print(input-label)
    # loss = torch.mean(torch.sqrt(torch.sum((input - label) ** 2)))
    # print('bbbbbbbbbbbbbbbbbbbbbbbbb')
    # print(loss)
    loss = torch.mean(torch.sqrt((input - label) ** 2))
    # print('aaaaaaaaaaaaaaaaaaaaaaa')
    # print(loss)
    return loss
if __name__ == "__main__":
    a = torch.zeros(3,5
    )
    print(len(a[0]))