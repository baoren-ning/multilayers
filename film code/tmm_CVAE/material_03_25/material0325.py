import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_ma():
    '''载入材料库'''
    filename= r'..\nk'
    si_path = filename+'/si.csv'
    # si_path = filename+'/si_modify.csv'
    al2o3_path = filename+'/al2o3.csv'
    sio2_path = filename+'/sio2.csv'
    vo2_h_path = filename + '/vo2_h.csv'
    hfo2_path = filename+'/hfo2.csv'
    mgf2_path = filename+'/mgf2.csv'
    ag_path = filename + '/ag.csv'
    zno_path = filename+'/zno.csv'
    zns_path = filename+'/zns.csv'
    si3n4_path = filename + '/si3n4.csv'
    # sic_path = filename + '/sic1.csv'
    sic_path = filename + '/sic.csv'
    tio2_path = filename + '/tio2.csv'
    # znse_path = filename + '/znse.csv'
    # znse_path = filename + '/znse-1.csv'
    znse_path = filename + '/znse.csv'
    nk_si = np.genfromtxt(si_path, delimiter=',')[1:,:]
    nk_sio2 = np.genfromtxt(sio2_path, delimiter=',')[1:,:]
    nk_vo2_h = np.genfromtxt(vo2_h_path, delimiter=',')[1:,:]
    nk_al2o3 = np.genfromtxt(al2o3_path, delimiter=',')[1:,:]
    nk_hfo2 = np.genfromtxt(hfo2_path, delimiter=',')[1:,:]
    nk_mgf2 = np.genfromtxt(mgf2_path, delimiter=',')[1:,:]
    nk_zno = np.genfromtxt(zno_path, delimiter=',')[1:,:]
    nk_zns = np.genfromtxt(zns_path, delimiter=',')[1:,:]
    nk_si3n4 = np.genfromtxt(si3n4_path, delimiter=',')[1:,:]
    nk_sic = np.genfromtxt(sic_path, delimiter=',')[1:,:]
    nk_tio2 = np.genfromtxt(tio2_path, delimiter=',')[1:,:] #51:402
    nk_znse = np.genfromtxt(znse_path, delimiter=',')[1:,:]
    refractive_index_si = (nk_si[:, 1] + nk_si[:, 2] * 1j).reshape(-1, 1)
    refractive_index_sio2 = (nk_sio2[:, 1] + nk_sio2[:, 2] * 1j).reshape(-1, 1)
    refractive_index_vo2_h = (nk_vo2_h[:, 1] + nk_vo2_h[:, 2] * 1j).reshape(-1, 1)
    refractive_index_al2o3 = (nk_al2o3[:, 1] + nk_al2o3[:, 2] * 1j).reshape(-1, 1)
    refractive_index_hfo2 = (nk_hfo2[:, 1] + nk_hfo2[:, 2] * 1j).reshape(-1, 1)
    refractive_index_mgf2 = (nk_mgf2[:, 1] + nk_mgf2[:, 2] * 1j).reshape(-1, 1)
    refractive_index_zno = (nk_zno[:, 1] + nk_zno[:, 2] * 1j).reshape(-1, 1)
    refractive_index_zns = (nk_zns[:, 1] + nk_zns[:, 2] * 1j).reshape(-1, 1)
    refractive_index_si3n4 = (nk_si3n4[:, 1] + nk_si3n4[:, 2] * 1j).reshape(-1, 1)
    refractive_index_sic = (nk_sic[:, 1] + nk_sic[:, 2] * 1j).reshape(-1, 1)
    refractive_index_tio2 = (nk_tio2[:, 1] + nk_tio2[:, 2] * 1j).reshape(-1, 1)
    refractive_index_znse = (nk_znse[:, 1] + nk_znse[:, 2] * 1j).reshape(-1, 1)
    # refractive_index_ag = (nk_ag[:, 1] + nk_ag[:, 2] * 1j).reshape(-1, 1)
    material_matrix = np.concatenate(
        [
            refractive_index_si         #0 前39个折射率有问题
            ,refractive_index_sio2          #1
            ,refractive_index_vo2_h         #2
            ,refractive_index_al2o3         #3
            ,refractive_index_hfo2          #4
            ,refractive_index_mgf2          #5
            , refractive_index_zno          #6
             ,refractive_index_zns          #7
            , refractive_index_si3n4        #8
            , refractive_index_sic          #9
            ,refractive_index_tio2          #10
            ,refractive_index_znse    #11
         ]
        , axis=1)
    # material_matrix=pd.DataFrame(material_matrix,columns=['si','sio2'])
    return torch.from_numpy(material_matrix).to(device)


def get_tensor_from_pd(dataframe_series):
    return torch.tensor(data=dataframe_series.values)


# material_matrix=get_tensor_from_pd(material_matrix)
def get_index(probability_matrix, material_library):
    assert probability_matrix.shape[1] == material_library.shape[1]
    M_matrix = torch.empty(probability_matrix.shape[0], material_library.shape[0], dtype=torch.complex128)  # TMM中的M矩阵
    material_id = torch.argmax(probability_matrix, dim=1)  # 概率矩阵中每行最大值的位置
    probability_value = torch.max(probability_matrix, dim=1)
    for i in range(probability_matrix.shape[0]):
        index_i = torch.sum(probability_matrix[i, :] * material_library, dim=1)
        M_matrix[i, :] = index_i
    return M_matrix, material_id, probability_value
