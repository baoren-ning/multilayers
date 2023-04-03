import torch
from torch import nn, optim
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Act_fun(nn.Module):
    def __init__(self, target_min, target_max):
        super(Act_fun, self).__init__()
        self.target_min = target_min
        self.target_max = target_max

    def forward(self, x):
        x02 = torch.tanh(x) + 1
        scale = (self.target_max - self.target_min) / 2.
        return x02 * scale + self.target_min


class Residual(nn.Module):
    def __init__(self, input, middle, output, use_fc=True):
        super(Residual, self).__init__()
        self.fc1 = nn.Linear(input, middle)
        self.fc2 = nn.Linear(middle, output)
        if use_fc:
            self.fc3 = nn.Linear(input, output)
        else:
            self.fc3 = None
        self.bn1 = nn.BatchNorm1d(middle)
        self.bn2 = nn.BatchNorm1d(output)

    def forward(self, X):
        Y = F.relu((self.fc1(X)))
        Y = (self.fc2(Y))
        if self.fc3:
            X = self.fc3(X)
        return F.relu(Y + X)


class Generator(nn.Module):
    def __init__(self, input_dim, t_lim, num_of_layer, num_of_material, middle=256, output=128):
        super(Generator, self).__init__()
        self.t_lim = t_lim
        self.rs1 = Residual(input_dim, middle, output, True)
        self.rs2 = Residual(output, middle, output, False)
        self.rs3 = Residual(output, middle, output, False)
        self.rs4 = Residual(output, middle, output, False)

        self.t_net = nn.Sequential(
            Residual(output, 256, 256, True),#256  256
            nn.ReLU(),
            Residual(256, 128, 512, True),#256  128   512
            nn.ReLU(),
            nn.Linear(512, 256),#512, 256
            nn.ReLU(),
            nn.Linear(256, num_of_layer),#256, num_of_layer
            # nn.Linear(output, 128),
        )
        self.nk_net = nn.Sequential(
            Residual(output, 256, 512, True), #256, 512
            nn.ReLU(),
            Residual(512, 256, 512, True),#512, 256, 512
            nn.ReLU(),
            Residual(512, 128, 256, True),#512, 128, 256
            nn.ReLU(),
            nn.Linear(256, 512),#256, 512
            nn.ReLU(),
            nn.Linear(512, 128),#512, 128
            nn.ReLU(),
            # nn.LeakyReLU(),
            nn.Linear(128, (num_of_layer-2) * num_of_material),#128, num_of_layer * num_of_material
            Reshape(num_of_layer-2, num_of_material),
        )

    def forward(self, x):
        out = F.leaky_relu(self.rs1(x))
        out = F.relu(self.rs2(out))
        out = F.relu(self.rs3(out))
        out = F.relu(self.rs4(out))
        # out = F.leaky_relu(self.rs1(x))
        # out = F.leaky_relu(self.rs2(out))
        # out = F.leaky_relu(self.rs3(out))
        # out = F.leaky_relu(self.rs4(out))
        t = Act_fun(self.t_lim[0], self.t_lim[1])(self.t_net(out))
        probability_matrix = self.nk_net(out)
        return t, probability_matrix

class Generator1(Generator):
    '''用于折射率连续任务的预测网络'''
    def __init__(self,input_dim, t_lim, num_of_layer, num_of_material, middle=256, output=128):
        super(Generator1, self).__init__(input_dim, t_lim, num_of_layer, num_of_material, middle, output)
        self.n_net = nn.Sequential(
            Residual(output, 256, 512, True),
            nn.ReLU(),
            Residual(512, 256, 512, False),
            nn.ReLU(),
            Residual(512, 128, 256, True),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            # nn.LeakyReLU(),
            nn.Linear(128, num_of_layer),

        )

        self.k_net = nn.Sequential(
            Residual(output, 256, 512, True),
            nn.ReLU(),
            Residual(512, 256, 512, False),
            nn.ReLU(),
            Residual(512, 128, 256, True),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            # nn.LeakyReLU(),
            nn.Linear(128, num_of_layer),

        )
    def forward(self, x):
        out = F.leaky_relu(self.rs1(x))
        out = F.relu(self.rs2(out))
        out = F.relu(self.rs3(out))
        out = F.relu(self.rs4(out))
        t = Act_fun(self.t_lim[0], self.t_lim[1])(self.t_net(out))
        n = Act_fun(1, 4)(self.n_net(out))
        k = Act_fun(0, 0.5)(self.n_net(out))
        out_nk = n + 1j * k
        return t, out_nk