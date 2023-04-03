import torch
from torch.utils.data import Dataset,DataLoader
import torchvision
from torch import nn
import torch.nn.functional as F
import tmm_fast_main.vectorized_tmm_dispersive_multistack as tmm
from torchvision import transforms
import numpy as np
class MyDataset(Dataset):
    def __init__(self,transform=None):
        super(MyDataset,self).__init__()
        filename = r'E:\matlab_work\multilayer\数据集\layer3'
        x_data = np.genfromtxt(filename + '/materials_index.csv', delimiter=',')
        t_data = (np.genfromtxt(filename + '/t_layers.csv', delimiter=',')/0.2)[:,0].reshape(-1,1)
        y_data = np.genfromtxt(filename + '/r.csv', delimiter=',')
        self.data=torch.from_numpy(x_data).long()
        self.data_t = torch.from_numpy(t_data).float()
        self.label=torch.from_numpy(y_data).float()
        self.transform=transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x=self.data[item]
        t = self.data_t[item]
        # m_t=torch.cat([x,t],-1)
        label=self.label[item]
        if self.transform is not None:
            label=self.transform(x)
        return x,t,label