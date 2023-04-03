import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, dim, transform=None):
        super(MyDataset, self).__init__()
        filename = r'C:\Users\WJN\Desktop\nk_1101'
        x_data = np.random.normal(0, 1, (1, dim))
        # y_data = np.genfromtxt(filename + '/1.csv', delimiter=',').reshape(1, -1)         #csv格式(1，N)
        y_data = np.genfromtxt(filename + '/500to800.csv', delimiter=',', encoding='utf-8')[:, 1].reshape(1,
                                                                                                          -1)  # txt格式(N，2)
        self.data = torch.from_numpy(x_data).float()
        self.label = torch.from_numpy(y_data).float()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = self.data[item]
        label = self.label[item]
        if self.transform is not None:
            x = self.transform(x)
        return x, label


class CustomLoss(nn.Module):
    '''自定义损失函数'''

    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, x, y):
        # mse_loss = torch.mean(torch.exp(torch.pow((x - y), 2)/10))
        mse_loss = self.mse(x, y)
        return mse_loss


def count_and_plot(data, sort=False):
    for i in range(len(data)):
        data[i] = data[i].detach().cpu().numpy()
    data_min = min(data)
    dara_max = max(data)
    bins = np.linspace(data_min, dara_max, 6)
    cats = pd.cut(data, bins)
    num = pd.value_counts(cats, sort=sort)
    print(num)
    # num.plot(kind='bar')
    # plt.tick_params(labelsize=10)
    # plt.xticks(rotation=0)
    # plt.show()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0.0000001):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss


def I_bb(lamda, T):
    C1 = 3.741 * 10 ** -16  # W.m ^ 2 / sr
    C2 = 0.01438769  # m.K
    result = C1 / ((lamda ** 5) * (torch.exp(C2 / (lamda * T)) - 1))
    return result


def cal_emittance_rad(lamda, input_tensor, T_sur):
    BB_Tsur_l = I_bb(lamda, T_sur).reshape(1, -1).to(device)
    abs_lamda = F.interpolate(input_tensor, size=[2, 101], mode="bilinear", align_corners=False)
    ynew = abs_lamda.view(2,-1)
    ynew = ynew[1:]
    # ynew = input_tensor
    emittance = ynew
    emittance_rad = torch.trapz(BB_Tsur_l * emittance, lamda) / torch.trapz(BB_Tsur_l, lamda)
    return emittance_rad

def AM15(lamda, input_tensor):
    lamda = lamda.to(device)
    filename = r'C:\Users\WJN\Desktop\nk\AM.csv'
    AM15cir = np.genfromtxt(filename, delimiter=',', encoding='utf-8')[:, 1]
    AM15cir = AM15cir.reshape(1, -1)
    AM15cir = torch.tensor(AM15cir).to(device)
    A = input_tensor
    alpha = torch.trapz(AM15cir * A, lamda) / torch.trapz(AM15cir, lamda)
    return alpha
