import torch
import torchvision
from torch import nn
import numpy as np
from get_data import MyDataset
from model import CondVAE
from torch.utils.data import Dataset,DataLoader
import tmm_fast_main.vectorized_tmm_dispersive_multistack as tmm
# 1.3 完成损失函数和训练函数
reconstruction_function = nn.MSELoss()
samples=MyDataset()
train_size = int(len(samples) * 0.7)
test_size = len(samples) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(samples, [train_size, test_size])
train_loader = DataLoader(train_dataset,batch_size=128,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=128,shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loss_function(recon_x, x, mean, lg_var):  # 损失函数：将MSE的损失缩小到一半，再与KL散度相加，目的在于使得输出的模拟样本可以有更灵活的变化空间。
    MSEloss = reconstruction_function(recon_x, x)  # MSE损失
    KLD = -0.5 * torch.sum(1 + lg_var - mean.pow(2) - lg_var.exp())
    return KLD,MSEloss,0.5 * MSEloss + KLD

def train(model, num_epochs):  # 训练函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            x, t, label = data
            x=x.to(device)
            t = t.to(device)
            label=label.to(device)
            optimizer.zero_grad()
            recon_batch, mean, lg_var,O = model(x,t,label)
            Kl,ELBO,loss1 = loss_function(recon_batch, torch.cat([x,t],-1), mean, lg_var)
            loss2=reconstruction_function(label,O)
            loss=loss1+loss2
            loss.backward()
            train_loss += loss.data
            optimizer.step()
        if epoch % 10 == 0:
            total_test_loss = 0
            for batch_idx, data in enumerate(test_loader):
                with torch.no_grad():
                    x, t, label = data
                    x = x.to(device)
                    t = t.to(device)
                    label = label.to(device)
                    recon_batch, mean, lg_var = model(x, t, label)
                    test_Kl, test_ELBO, test_loss = loss_function(recon_batch, torch.cat([x, t], -1), mean, lg_var)
                    total_test_loss =total_test_loss+test_loss.data
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(loss.data),"Kl=", "{:.9f}".format(Kl.data))
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(total_test_loss/(batch_idx+1)), "Kl=", "{:.9f}".format(test_Kl.data))
    print("完成训练 cost=", loss.data)
    torch.save(model, 'CVAE.pth')

# 1.5 训练模型并输出可视化结果
if __name__ == '__main__':
    model = CondVAE().to(device)  # 实例化模型
    train(model, 1000)
    sample = iter(train_loader)  # 取出10个样本，用于测试
    x,t, labels = next(sample)
    x=x[0:2,:].to(device)
    t = t[0:2, :].to(device)
    labels = labels[0:2, :].to(device)
    with torch.no_grad():
        pred, mean, lg_var = model(x,t, labels)
    print(pred)
    print(x,t)