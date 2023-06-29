from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np



class TaskSelector(nn.Module):
    def __init__(self, input_size, hidden_size, t=1, adjust_step=500):
        super(TaskSelector, self).__init__()
        self.z1 = nn.Sequential(nn.Linear(input_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, 1),
                                nn.Tanh())
        self.z2 = nn.Sequential(nn.Linear(input_size, input_size),
                                nn.ReLU(),
                                nn.Linear(input_size, input_size))
        self.t = t
        self.adjust_step = adjust_step

    def forward(self, x, iter_num):
        """
        :param x: [n_pool, d]
        :return: [n_pool]
        """
        t = ((iter_num // self.adjust_step) + 1) * self.t
        comp_x = (x.sum(0).unsqueeze(0) - x) / (len(x) - 1)
        comp_x = self.z2(comp_x)
        output = self.z1(x + comp_x)
        output = output.view(-1)
        prob = torch.softmax(output / t, dim=0)
        return prob

    def sample(self, prob, n):
        # prob: List
        prob = np.array(prob)
        prob /= prob.sum()
        res = np.random.choice(len(prob), n, replace=False, p=prob).tolist()
        return res
