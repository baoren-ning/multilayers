#加ag衬底算红外光谱
import matplotlib.pyplot as plt
import numpy as np
import torch
from tmm_fast_main.vectorized_tmm_dispersive_multistack import coh_vec_tmm_disp_mstack as tmm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from get_data import *
from torch.utils.data import DataLoader
from model import Generator, Generator1
from material import get_ma, get_index
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
filename = r'C:\Users\WJN\Desktop\nk_1101'
vo2_l_path = filename + '/vo2_l.csv'
nk_vo2_l = np.genfromtxt(vo2_l_path, delimiter=',')[54:105]#[0:351]
refractive_index_vo2_l = (nk_vo2_l[:, 1] + nk_vo2_l[:, 2] * 1j).reshape(1, -1)
nk_material = torch.from_numpy(refractive_index_vo2_l)

ag_path = filename + '/ag.csv'
nk_ag = np.genfromtxt(ag_path, delimiter=',')[54:105]#[0:351]
refractive_index_ag = (nk_ag[:, 1] + nk_ag[:, 2] * 1j).reshape(1, -1)

nk_material_ag = torch.from_numpy(refractive_index_ag)
lamda = (torch.arange(8000, 13001, 50) * 10 ** -9).to(device)#400, 1101, 2
'''一些可能需要修改的全局变量'''
continuous_nk = 1  # 1是不连续，0是连续
nk_dim = 51  # 波长范围离散为nk_dim个点    351
num_of_layer = 10  # 膜层数
num_of_material = 12  # 材料库里材料的数量
dim = 10  # 随机正态分布的维度
wl = np.linspace(8000, 13000, nk_dim) * (10 ** (-9))  # 波长范围400, 1100
theta = np.linspace(0, 2, 2) * (np.pi / 180)  # 入射角，一般不改就是正入射
theta = torch.tensor(theta, requires_grad=False)  # 不记录入射角的梯度
wl = torch.tensor(wl, requires_grad=False)  # 不记录波长的梯度
mode = 'R'  # R，T，A光谱
myloss = CustomLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
samples = MyDataset(dim=dim)
material_library = get_ma()
train_loader = DataLoader(samples, batch_size=1, shuffle=False)


def M_T(out_t, out_nk, nk_dim, continuous_nk):
    ''''
    out_t：神经网络输出的膜层厚度，需要调整为两边是无限厚的空气层
    out_nk：n+jk形式，需要调整为两边为空气时的折射[1,1+1j,2+0.4j........,1]
    '''
    T = F.pad(out_t, (1, 1), mode='constant', value=np.inf) + 0j
    if continuous_nk:
        out_nk = F.pad(out_nk.reshape(1, num_of_layer+1, nk_dim), (0, 0, 1, 1), mode='constant', value=1)
    else:
        out_nk = F.pad(out_nk.reshape(1, num_of_layer+1, 1), (0, 0, 1, 1), mode='constant', value=1)
        out_nk = torch.tile(out_nk, (1, 1, nk_dim))
    M = out_nk
    return T, M


def get_best_value(error, best_mse, T, best_T, M, best_M, material_id, best_material_id, p_value, best_p_value):
    '''筛选每次优化中MSE最小的优化结果'''
    if error < best_mse:
        best_mse = error
        best_T = T
        best_M = M
        if continuous_nk:
            best_material_id = material_id
            best_p_value = p_value
    return best_mse, best_T, best_M, best_material_id, best_p_value


def train(model, epoch, continuous_nk):
    '''单次优化'''
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=False,
                                  min_lr=1e-6)  # verbose控制是否显示具体学习率信息
    early_stopping = EarlyStopping(patience=20, verbose=False)
    # mse = torch.nn.MSELoss()
    mse = CustomLoss()
    best_mse = 999  # 设置一个无穷大的初始值，这里认为999是一个足够大的值
    best_T = None  # 每次优化最好的厚度
    best_M = None  # 每次优化最好的折射率矩阵
    best_material_id = None
    best_p_value = None
    material_id = None
    p_value = None
    for i in range(epoch):
        model.train()
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            x = x.reshape(1, -1)
            optimizer.zero_grad()
            if continuous_nk:
                alpha = 1 + (i * 0.95) ** 2  # 可以进行修改
                out_t, probability_matrix = model(x)
                assert not torch.any(torch.isnan(probability_matrix))
                probability_matrix = F.softmax(alpha * probability_matrix, dim=1)
                M_matrix, material_id, p_value = get_index(probability_matrix, material_library)  # p_value 材料种类的概率值
                t_ag = torch.tensor(5 * 10 ** -9).view(1, -1).to(device)
                out_t = torch.cat([out_t, t_ag], 1)
                M_matrix = torch.cat([M_matrix,nk_material_ag], 0)
                T, M = M_T(out_t, M_matrix, nk_dim, continuous_nk)
                c_id = torch.where(material_id == 2)
                M_matrix[c_id] = nk_material
                T_l, M_l = M_T(out_t, M_matrix, nk_dim, continuous_nk)
            else:
                out_t, out_nk = model(x)
                T, M = M_T(out_t, out_nk, nk_dim, continuous_nk)
            O_R = tmm('s', M, T, theta, wl, device='cuda')[mode]
            O_T = tmm('s', M, T, theta, wl, device='cuda')['T']
            O_A = 1-O_R-O_T
            O_l_R = tmm('s', M_l, T_l, theta, wl, device='cuda')[mode]
            O_l_T = tmm('s', M_l, T_l, theta, wl, device='cuda')['T']
            O_l_A = 1 - O_l_R - O_l_T
            input_h = torch.cat([wl.reshape(1, -1).to(device), O_A[0:1, :]], dim=0).view(1, 1, 2, nk_dim)
            input_l = torch.cat([wl.reshape(1, -1).to(device), O_l_A[0:1, :]], dim=0).view(1, 1, 2, nk_dim)
            em_h = cal_emittance_rad(lamda, input_h, 373)
            em_l = cal_emittance_rad(lamda, input_l, 300)
            deta = em_h - em_l
            error = 1 - deta
            best_mse, best_T, best_M, best_material_id, best_p_value = get_best_value(error, best_mse, T, best_T, M,
                                                                                      best_M, material_id,
                                                                                      best_material_id, p_value,
                                                                                      best_p_value)
            error.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            # print(model.nk_net[-2].weight.grad) #seq层可以这样看梯度
            optimizer.step()
            scheduler.step(error)
        early_stopping(error, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练
        # if i % 10 == 0:
        #     print('第{}轮的loss：{}'.format(i,error))
    if continuous_nk:
        return best_T, best_M, best_material_id, best_p_value, best_mse
    else:
        return best_T, best_M, best_mse


def model_train(iteration, continuous_nk):
    '''
        功能：用于反复优化iteration次，避免单次陷入局部最小值
        iteration次迭代寻找MSE最小的结构
        model是预测不连续n，k值的模型
        model1是预测连续n，k值的模型
    '''
    T_film = []  # 用来存储每次iteration中的膜层厚度值
    MSE = []  # 用来存储每次iteration结束时MSE的值
    material = []  # 用来存储每次iteration结束时MSE的值
    M_matrix = []
    p_values = []
    model = Generator(input_dim=dim
                      , t_lim=(15 * 10 ** -9, 250 * 10 ** -9)  # 修改每层的厚度范围
                      , num_of_layer=num_of_layer
                      , num_of_material=num_of_material
                      , middle=128, output=256).to(device)
    model1 = Generator1(input_dim=dim
                        , t_lim=(20 * 10 ** -9, 200 * 10 ** -9)  # 修改每层的厚度范围
                        , num_of_layer=num_of_layer
                        , num_of_material=num_of_material
                        , middle=128, output=256).to(device)
    model = model if continuous_nk == 1 else model1
    for i in range(iteration):
        if continuous_nk:
            T, M, material_id, p_value, mse = train(model, 400, continuous_nk)
            T_film.append(T)
            MSE.append(mse)
            M_matrix.append(M)
            material.append(material_id)
            p_values.append(p_value)
        else:
            T, M, mse = train(model, 200, continuous_nk)
            T_film.append(T)
            MSE.append(mse)
            M_matrix.append(M)
        if (i + 1) % 10 == 0:
            print('第{}次优化的损失是:{}'.format(i + 1, mse))
    return T_film, MSE, M_matrix, material, p_values


if __name__ == '__main__':
    start = time.time()
    T_film, MSE, M_matrix, material, p_values = model_train(1, continuous_nk)
    id = torch.argmin(torch.tensor(MSE))
    O_R = tmm('s', M_matrix[id], T_film[id], theta, wl, device='cuda')[mode]
    O_T = tmm('s', M_matrix[id], T_film[id], theta, wl, device='cuda')['T']
    O = 1-O_R-O_T
    vo2_id = torch.where(material[id] == 2)
    vo2_id = [i + 1 for i in list(vo2_id)]
    print(vo2_id)
    M_matrix_l=M_matrix[id].reshape(-1,nk_dim)

    M_matrix_l[vo2_id] = nk_material
    O_R_l = tmm('s', M_matrix_l.reshape(1,-1,nk_dim), T_film[id], theta, wl, device='cuda')[mode]
    O_T_l = tmm('s', M_matrix_l.reshape(1,-1,nk_dim), T_film[id], theta, wl, device='cuda')['T']
    O_l = 1 - O_R_l - O_T_l

    mse1 = torch.nn.MSELoss()
    end = time.time()
    runTime = end - start
    print("运行时间：", runTime)
    plt.plot(wl * 10 ** 9, O[0, :].detach().cpu().numpy(), color='red', label='H_vo2')
    plt.plot(wl * 10 ** 9, O_l[0, :].detach().cpu().numpy(), color='blue', label='L_vo2')
    plt.ylim((0, 1))
    plt.legend()
    plt.show()
    # count_and_plot(MSE, sort=False)
    print('1', MSE[id])
    print('2',id)
    print(p_values[id])
    print(T_film[id])