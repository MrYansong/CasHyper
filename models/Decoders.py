from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from models.hypercomplex_func import Quaternion, hypercomplex_Linear, Octonion, Complex, Sedenion

import config.Constants as Constants
from config.Constants import DEVICE

from models.UserAdapter import AttentionMerger_hypercomplex

class FusionDecoder(nn.Module):
    def __init__(self, opt):
        super(FusionDecoder, self).__init__()
    
        # hypers for the model
        self.hyper_dim = opt.hyper_dim
        self.input_size = opt.d_word_vec // self.hyper_dim
        self.input_dim = opt.d_word_vec
        self.user_size = opt.user_size
        self.d_model = opt.d_word_vec
        self.transformer_dim = opt.d_word_vec
        self.user_lt_attention= AttentionMerger_hypercomplex(opt, input_size = self.input_size, hyper_dim=self.hyper_dim)
        self.user_lt_merger = Merger_Users(opt, self.input_dim, self.input_dim)

        self.decoder = HyperTransformerBlock(input_size = self.transformer_dim // self.hyper_dim, max_len=opt.max_len, hyper_dim=self.hyper_dim, n_heads=opt.heads)

        self.f_linear = nn.Linear(self.input_size + opt.time_dim, 1, bias=True)

        self.user_predictor_linear = nn.Linear(self.d_model + opt.time_dim, self.user_size)
        self.pred_with_weights = AttentionWeightedModel(self.user_size)
        self.dropout = nn.Dropout(opt.dropout_rate)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_embeddings, all_user_embeddings, time_embedding, inputs, input_timestamp, input_id, epoch, train=True):
        batch_size, max_len, dim = input_embeddings.size()
        mask = (inputs == Constants.PAD)

        user_att_embedding  = self.user_lt_attention(input_embeddings)
        user_att_embedding = self.user_lt_merger(input_embeddings, user_att_embedding)
        user_att_embedding = self.dropout(user_att_embedding)
        user_embedding = user_att_embedding

        final_input = user_embedding.view(batch_size, max_len, self.hyper_dim, -1)
        att_out, mseloss = self.decoder(final_input.to(DEVICE), mask=mask.to(DEVICE))
        final_out = self.dropout(att_out)

        all_user_embeddings = all_user_embeddings.view(self.user_size, self.hyper_dim, self.input_size)
        pred_inter = self.predictor(final_out.to(DEVICE), all_user_embeddings.to(DEVICE))
        pred_lin = self.predictor_simply(final_out.to(DEVICE), time_embedding.to(DEVICE))
        pred = self.pred_with_weights(pred_inter, pred_lin)
        mask = self.get_previous_user_mask(inputs.to(DEVICE), self.user_size)
        output = pred.to(DEVICE) + mask.to(DEVICE)
        if train:
            return output, mseloss
        else:
            return output
    def get_previous_user_mask(self, seq, user_size):
        assert seq.dim() == 2
        prev_shape = (seq.size(0), seq.size(1), seq.size(1))
        seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
        previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
        previous_mask = torch.from_numpy(previous_mask)
        if seq.is_cuda:
            previous_mask = previous_mask.to(DEVICE)

        masked_seq = previous_mask * seqs.data.float()

        PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
        if seq.is_cuda:
            PAD_tmp = PAD_tmp.to(DEVICE)
        masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
        ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
        if seq.is_cuda:
            ans_tmp = ans_tmp.to(DEVICE)
        masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
        masked_seq = Variable(masked_seq, requires_grad=False)
        return masked_seq
    
    def predictor(self, cascade_user_emb, all_user_emb):

        all_user_emb = all_user_emb.view(self.user_size, -1).transpose(0, 1)
        cascade_user_emb = cascade_user_emb.unsqueeze(2)
        score = torch.matmul(cascade_user_emb, all_user_emb)
        scores = score.squeeze(dim=2)

        return scores
    
    def predictor_simply(self, cascade_user_emb, time_embedding):
        scores = torch.cat([cascade_user_emb, time_embedding], dim=-1)
        scores = self.user_predictor_linear(scores)
        return scores

    
class AttentionWeightedModel(nn.Module):
    def __init__(self, input_dim):
        super(AttentionWeightedModel, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, pred1, pred2):
        weight = self.attention(pred1)
        pred = weight * pred1 + (1 - weight) * pred2
        return pred

class HyperTransformerBlock(nn.Module):
    def __init__(self, input_size, max_len, hyper_dim, d_k=64, d_v=64, n_heads=2, is_layer_norm=True, attn_dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.input_size = input_size
        self.hyper_dim = hyper_dim
        self.input_dim = input_size * self.hyper_dim
        self.max_len = max_len
        self.pi = 3.1415
        self.ReLU = nn.ReLU()
        self.tanh = nn.Tanh()

        self.input_Lin_Q = hypercomplex_Linear(self.input_size, self.hyper_dim, self.n_heads*self.input_size, concat=False, dropout_ratio=0.1)
        self.input_Lin_K = hypercomplex_Linear(self.input_size, self.hyper_dim, self.n_heads*self.input_size, concat=False,dropout_ratio=0.1)
        self.input_Lin_V = hypercomplex_Linear(self.input_size, self.hyper_dim, self.n_heads*self.input_size, concat=False,dropout_ratio=0.1)

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=self.input_dim * self.n_heads)

        self.freq = (2 * self.pi * torch.arange(1, self.max_len + 1, dtype=torch.float32) / self.max_len).to(DEVICE)

        self.W_omega = nn.Parameter(torch.FloatTensor(self.input_dim, self.input_dim))
        self.W_theta = nn.Parameter(torch.FloatTensor(self.input_dim, self.input_dim))
        nn.init.xavier_uniform_(self.W_omega)
        nn.init.xavier_uniform_(self.W_theta)

        self.linear1 = nn.Linear(self.input_dim * self.n_heads, self.input_dim)
        self.linear2 = nn.Linear(self.input_dim * self.n_heads, self.input_dim)
        self.dropout = nn.Dropout(attn_dropout)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output
    
    def MSELoss(self):
        return F.mse_loss(self.W_omega, self.W_theta)

    def Euler_func(self, Q):
        if len(Q.shape) == 3:
            assert Q.shape[-1] == self.input_dim, 'Euler_func: Q dim[-1] is not matched!'
        elif len(Q.shape) == 4:
            assert Q.shape[-2] == self.hyper_dim, 'Euler_func: Q dim[-2] is not matched!'
            bsz, max_len, hyperdim, dim = Q.shape
            Q = Q.view(bsz, max_len, -1)
        else:
            assert 'Euler_func: Q dim error!'
        assert Q.shape[-1] == self.input_dim, 'Euler_func: Q input_dim is not matched!'
        theta = self.tanh(torch.matmul(Q, self.W_theta))
        omega = self.ReLU(torch.matmul(Q, self.W_omega))
        freq = (self.freq.unsqueeze(0).unsqueeze(-1)).repeat(omega.shape[0], 1, omega.shape[-1]) * omega + theta
        result = torch.cos(freq) + torch.sin(freq) * Q
        chunks = torch.chunk(result, self.hyper_dim, dim=-1)
        return torch.stack(chunks, dim=-2)

    def Rotation_func(self, Q, euler):
        assert Q.shape[-2] == self.hyper_dim, 'Rotation_func: Q dim is not matched!'
        assert euler.shape[-2] == self.hyper_dim, 'Rotation_func: euler dim is not matched!'
        Q = Q.permute(0, 2, 1, 3)
        euler = euler.permute(0, 2, 1, 3)
        result = torch.einsum("bhqd,bhkd->bhqk", Q, euler)
        return result

    def forward(self, input, mask=None):
        bsz, max_len, hyper_dim, input_size = input.shape
        Q_hat = self.input_Lin_Q(input).view(bsz, max_len, self.hyper_dim, self.n_heads, self.input_size)
        K_hat = self.input_Lin_K(input).view(bsz, max_len, self.hyper_dim, self.n_heads, self.input_size)
        V = self.input_Lin_V(input).view(bsz, max_len, self.hyper_dim, self.n_heads, self.input_size)

        Q_hat = Q_hat.permute(0,3,1,2,4).view(bsz*self.n_heads, max_len, self.hyper_dim, self.input_size)
        K_hat = K_hat.permute(0,3,1,2,4).view(bsz*self.n_heads, max_len, self.hyper_dim, self.input_size)
        V = V.permute(0,3,1,2,4).view(bsz*self.n_heads, max_len, self.hyper_dim, self.input_size)

        Q_euler = self.Euler_func(Q_hat)
        K_euler = self.Euler_func(K_hat)

        Q_phi = self.Rotation_func(Q_hat, Q_euler)
        K_psi = self.Rotation_func(K_hat, K_euler)
        original_Q_phi_shape = Q_phi.shape
        Q_phi = torch.flatten(Q_phi, start_dim=-2, end_dim=-1)
        K_psi = torch.flatten(K_psi, start_dim=-2, end_dim=-1)

        if self.hyper_dim == 4:
            Score = (Quaternion(Q_phi).multiply(Quaternion(K_psi))).tensor[:, 0, :] / (self.max_len * (self.input_dim**0.5))
        elif self.hyper_dim == 8:
            Score = (Octonion(Q_phi).multiply(Octonion(K_psi))).tensor[:, 0, :] / (self.max_len * (self.input_dim**0.5))
        elif self.hyper_dim == 2:
            Score = (Complex(Q_phi).multiply(Complex(K_psi))).tensor[:, 0, :] / (self.max_len * (self.input_dim**0.5))
        elif self.hyper_dim == 16:
            Score = (Sedenion(Q_phi).multiply(Sedenion(K_psi))).tensor[:, 0, :] / (self.max_len * (self.input_dim**0.5))
        else:
            print('hyper_dim ', self.hyper_dim, ' illegal.')
        Score = Score.view(original_Q_phi_shape[0], original_Q_phi_shape[-2], original_Q_phi_shape[-1])

        if mask is not None:
            mask = mask.unsqueeze(dim=1).expand(-1, self.n_heads, -1)
            mask = mask.view(-1, mask.size(-1))
            pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, Q_hat.size(1))
            mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool().to(DEVICE)
            mask_ = mask + pad_mask
            Score = Score.masked_fill(mask_, -2**32+1)

        Score = torch.softmax(Score, dim=-1)

        V_permuted = V.permute(0, 2, 3, 1)
        Att_val = torch.einsum("bmm, bhkm->bhkm", Score, V_permuted).view(bsz, self.n_heads, self.hyper_dim, self.input_size, max_len)
        Att_val = Att_val.permute(0, 4, 2, 1, 3).contiguous().view(bsz, max_len, self.hyper_dim, self.n_heads*self.input_size)

        bsz, max_len, hyperdim, dim = Q_hat.shape
        input_flat = input.unsqueeze(2).expand(-1, -1, self.n_heads, -1, -1).view(bsz, max_len, -1)
        Att_val_flat = Att_val.view(bsz, max_len, -1)
        mseloss = self.MSELoss()
        if self.is_layer_norm:
            X = self.layer_morm(input_flat + Att_val_flat)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = input_flat + Att_val_flat
            output = self.FFN(X) + X
        return output, mseloss


class Merger_Users(nn.Module):
    def __init__(self, opt, input_size=None, output_size=None, dropout_rate=0.1):
        super(Merger_Users, self).__init__()

        self.input_size = input_size or opt.d_word_vec
        self.output_size = output_size or opt.d_word_vec

        self.LN1 = nn.LayerNorm(normalized_shape=self.input_size)
        self.LN2 = nn.LayerNorm(normalized_shape=self.input_size)

    def forward(self, input_x, input_Fx):
        return self.LN2(input_x + self.LN1(input_x + input_Fx)).to(DEVICE)