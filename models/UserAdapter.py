import numpy as np
import torch
import torch as th 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from models.hypercomplex_func import hypercomplex_Linear

from config.Constants import DEVICE


class AttentionMerger_hypercomplex(nn.Module):
    def __init__(self, opt, input_size, hyper_dim):
        super(AttentionMerger_hypercomplex, self).__init__()
        self.input_size = input_size
        self.hyper_dim = hyper_dim
        self.input_dim = self.hyper_dim * self.input_size
        self.output_size = input_size
        self.max_len = opt.max_len

        self.Lin_Q = hypercomplex_Linear(self.input_size, self.hyper_dim, self.input_size, concat=False, dropout_ratio=0.1)
        self.Lin_K = hypercomplex_Linear(self.input_size, self.hyper_dim, self.input_size, concat=False, dropout_ratio=0.1)

    def Rotation_func(self, Q, Q_):
        assert Q.shape[-2] == self.hyper_dim, 'Rotation_func: Q dim is not matched!'
        assert Q_.shape[-2] == self.hyper_dim, 'Rotation_func: euler dim is not matched!'
        Q = Q.permute(0, 2, 1, 3)
        Q_ = Q_.permute(0, 2, 1, 3)
        result = torch.einsum("bhqd,bhkd->bhqk", Q, Q_)
        return result
    
    def forward(self, user_embedding):
        batch_size, max_len, input_dim = user_embedding.size()
        user_embedding = user_embedding.reshape(batch_size, max_len, self.hyper_dim, -1)
        Q = F.tanh(self.Lin_Q(user_embedding))
        K = F.tanh(self.Lin_K(user_embedding))
 
        Q = torch.flatten(Q, start_dim=-2, end_dim=-1)
        K = torch.flatten(K, start_dim=-2, end_dim=-1)
        Q_K = torch.einsum("bmk, bqk->bmq", Q, K)

        temperature = self.input_dim ** 0.5
        episilon = 1e-6
        Q_K = Q_K / (temperature + episilon)
        mask = torch.zeros([max_len, max_len]).to(DEVICE)
        mask += -2**32+1
        mask = torch.triu(mask, diagonal=1).to(DEVICE)

        b_mask = torch.zeros_like(Q_K).to(DEVICE)
        b_mask[:, :, :] = mask[:, :]

        Q_K += b_mask
        score = F.softmax(Q_K, dim=-1).to(DEVICE)
        user_embedding = user_embedding.reshape(batch_size, max_len, -1)
        output = torch.einsum("bll,bmd->bld", score, user_embedding)

        return output


