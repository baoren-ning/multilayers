import numpy as np
import torch
from torch.nn import functional as F
from scipy import interpolate
import matplotlib.pyplot as plt
import torch.nn as nn
class CustomLoss(nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()


    def Piecewise_function(self, x):
        if x > 0:
            return 1 / x
        else:
            return (1000 * (-x)) ** 2

    def forward(self, x, y):
        mse_loss = 2**(5*x)+Piecewise_function(y)
        return mse_loss
myloss=CustomLoss()
error1 = myloss(0.0503, -0.0038)
print(error1)



