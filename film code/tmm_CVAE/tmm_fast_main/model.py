import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from vectorized_tmm_dispersive_multistack import coh_vec_tmm_disp_mstack as tmm
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
# 1.4 定义条件变分自编码神经网络模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
z_dim=10
wl = np.linspace(300, 2500, 1101) * (10**(-9))
theta = np.linspace(0, 2 ,2) * (np.pi/180)
wl=torch.from_numpy(wl)
theta=torch.from_numpy(theta)
mode = 'T'
class CondVAE(nn.Module):  # 继承VAE类，实现条件变分自编码神经网络模型的正向结构。
    def __init__(self,z_dim=10):
        super(CondVAE, self).__init__()
        self.rnn=nn.Sequential(
                                nn.Embedding(num_embeddings=11, embedding_dim=10),
                                nn.LSTM(input_size=10, hidden_size=64,batch_first=True)
                                )
        self.fc_n = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1101)
        )

        self.fc_k = nn.Sequential(
                                nn.Linear(64, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1101)
                            )

        self.encode_fc1 = nn.Linear(1101, 256)
        self.encode_fc2 = nn.Linear(1101, 256)
        self.encoder = nn.Sequential(
                                nn.Linear(1613, 512),
                                nn.ReLU(),
                                nn.Linear(512, 2*z_dim)
                            )

        self.decoder_fc1 = nn.Linear(1101, 256)
        self.decoder = nn.Sequential(
            nn.Linear(266, 128),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def output_n_k_t(self,m):
        h1, _ = self.rnn(m)
        n = self.fc_n(h1[:, -1, :])
        k = self.fc_k(h1[:, -1, :])
        return n, k

    def TMM(self,n,k,t):
        T = F.pad(t, (1,1,0,0), mode='constant', value=np.inf) + 0j

        out_n = F.pad(n.reshape(n.shape[0], 1, 1101),(0,0,1,1,0,0),mode='constant',value=1)
        out_k = F.pad(k.reshape(k.shape[0], 1, 1101),(0,0,1,1,0,0),mode='constant',value=0)
        # out_n = out_n.repeat(1, 1, 1101)
        # out_k = out_k.repeat(1, 1, 1101)
        M = out_n + out_k * 1j
        O = tmm('s', M, T, theta, wl, device='cpu')['T']
        return O[:,0,:]

    def encode(self, n, k, label):
        n=F.relu(self.encode_fc1(n))
        k = F.relu(self.encode_fc1(k))
        h1 = torch.cat([n, k,label], dim=-1)
        h1 = self.encoder(h1)
        return h1[:5,:], h1[5:,:]

    def reparametrize(self, mean, lg_var):  # 采样器方法：对方差(lg_var)进行还原，并从高斯分布中采样，将采样数值映射到编码器输出的数据分布中。
        std = lg_var.exp().sqrt()
        # torch.FloatTensor(std.size())的作用是，生成一个与std形状一样的张量。然后，调用该张量的normal_()方法，系统会对该张量中的每个元素在标准高斯空间（均值为0、方差为1）中进行采样。
        eps = torch.FloatTensor(std.size()).normal_().to(device)  # 随机张量方法normal_()，完成高斯空间的采样过程。
        return eps.mul(std).add_(mean)


    def decode(self, z, label):
        h3 = F.relu(self.decoder_fc1(label))
        h3 = torch.cat([z, h3], dim=-1)
        h3 = self.decoder(h3)
        return h3


    def forward(self, m,t,label):
        n,k=self.output_n_k_t(m)
        O=self.TMM(n,k,t)
        mean, lg_var=self.encoder(n,k,O)
        z = self.reparametrize(mean, lg_var)
        return self.decode(z, label), mean, lg_var,O