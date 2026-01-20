import sys

sys.path.append("./")

import math

import torch
import torch.nn.functional as F
from torch import einsum, nn
import torch.nn.init as init

import config.Constants as Constants
from config.Constants import DEVICE


class TimeEncoder(nn.Module):
    def __init__(self,opt):
        super(TimeEncoder, self).__init__()
        self.encoder_type = opt.time_encoder_type
        if opt.time_encoder_type == "Interval":
            self.time_encoder= IntervalTimeEncoder(opt)
        else:
            raise  Exception("Undefined time encoder") 

        self.output_dim = self.time_encoder.output_dim

    def forward(self, inputs, timestamp, train):
        return self.time_encoder(inputs, timestamp, train)

class IntervalTimeEncoder(nn.Module):
    def __init__(self,opt):
        super(IntervalTimeEncoder, self).__init__()

        data_name="./data/"+opt.data
        self.pass_time=opt.pass_time
        self.n_time_interval = opt.time_interval
        self.output_dim=opt.time_dim
        self.linear_1= nn.Linear(self.n_time_interval+1, self.output_dim, bias=True).to(DEVICE)
        init.xavier_normal_(self.linear_1.weight)
        self.relu=nn.ReLU()


    def forward(self,inputs,timestamp,train):
        batch_size,max_len=inputs.size()

        pass_time=timestamp[:,1:]-timestamp[:,:-1]
        pass_time=F.relu(((pass_time / self.pass_time) * self.n_time_interval).long())
        pass_time=pass_time.view(batch_size*max_len,1).to(DEVICE)

        time_embedding_one_hot=torch.zeros(batch_size*max_len, self.n_time_interval+1).to(DEVICE)
        time_embedding_one_hot=time_embedding_one_hot.scatter_(1, pass_time, 1).to(DEVICE)
        time_embedding = self.linear_1(time_embedding_one_hot)  # [batch_size, max_len, output_dim]
        time_embedding=time_embedding.view(batch_size, max_len, self.output_dim).to(DEVICE)

        return time_embedding.to(DEVICE), timestamp[:, :-1]

