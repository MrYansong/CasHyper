import os, sys
from models.ODEGraphEncoder import GraphNN
from models.ODEEncoder import DynamicGraphODE
from models.SimpleModules import SelfGating
from models.GraphEncoder import  NormalGraphNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from config.Constants import DEVICE

sys.path.append("./")

class GraphEncoder(nn.Module):
    def __init__(self, opt):
        super(GraphEncoder, self).__init__()

        # hypers
        self.hyper_dim = opt.hyper_dim
        self.ntoken = opt.token_size
        self.user_size = opt.user_size
        self.d_model = opt.d_model  
        self.time_step = opt.time_step

        self.node_embedding = nn.Embedding(self.ntoken, self.d_model, padding_idx=0)
        nn.init.xavier_uniform_(self.node_embedding.weight)

        self.with_st_graph = opt.with_st_graph
        self.with_normal_gnn = opt.with_normal_gnn

        self.st_input_gating = SelfGating(self.d_model,self.d_model)
        self.static_graph_encoder = GraphNN(self.ntoken, self.d_model, dropout_rate=opt.dropout_rate, self_embedding=False)
        if self.with_normal_gnn: 
            self.static_graph_encoder = NormalGraphNN(self.ntoken, self.d_model, self.hyper_dim, dropout_rate=opt.dropout_rate, self_embedding=False)

        self.with_dy_graph = opt.with_dy_graph 
        self.with_edge_recomputing = opt.with_edge_recomputing
        if self.with_edge_recomputing: assert self.with_dy_graph == True
        self.dy_merger = opt.dy_merger                                      

        self.dy_input_gating = SelfGating(self.d_model,self.d_model)
        self.dy_graph_encoder = DynamicGraphODE(self.ntoken, self.d_model, self.hyper_dim, self.time_step, dropout_rate=opt.dropout_rate, self_embedding=False)

        self.dy_att = nn.Parameter(torch.ones(1, self.d_model))
        self.dy_att_m = nn.Parameter(torch.zeros(self.d_model, self.d_model))
        init.xavier_normal_(self.dy_att.data)
        init.xavier_normal_(self.dy_att_m.data)

        self.merger_att = nn.Parameter(torch.ones(1, self.d_model))
        self.merger_att_m = nn.Parameter(torch.zeros(self.d_model, self.d_model))
        init.xavier_normal_(self.merger_att.data)
        init.xavier_normal_(self.merger_att_m.data)

        self.dropout = nn.Dropout(opt.dropout_rate)

    def user_st_g_rep(self, inputs, static_graph=None):
        user_embedding = self.st_input_gating(self.node_embedding.weight)
        static_node_embedding_lookup = self.static_graph_encoder(user_embedding, static_graph).to(DEVICE)
        user_static_embedding = F.embedding(inputs.to(DEVICE), static_node_embedding_lookup.to(DEVICE))
        user_static_embedding = self.dropout(user_static_embedding)

        return user_static_embedding, static_node_embedding_lookup

    def user_dy_g_rep(self, inputs, dynamic_graph=None):
        user_embedding = self.dy_input_gating(self.node_embedding.weight)
        dynamic_node_embedding_lookup, last_node_embedding_lookup = self.dy_graph_encoder(user_embedding, dynamic_graph)
        user_dynamic_embedding, dynamic_node_embedding_lookup = self.UserDyEmbeddingMerger(inputs, dynamic_node_embedding_lookup)

        return user_dynamic_embedding, dynamic_node_embedding_lookup

    def forward(self, inputs, static_graph=None, diffusion_graph=None):

        if self.with_st_graph: 
            user_st_seq_rep, all_st_u_embeddings = self.user_st_g_rep(inputs, static_graph)

        if self.with_dy_graph:
            user_dy_seq_rep, all_dy_u_embeddings =  self.user_dy_g_rep(inputs, diffusion_graph)

        if self.with_dy_graph and self.with_st_graph:
            user_seq_rep, all_u_embeddings  = self.UserEmbeddingMerger(inputs, all_st_u_embeddings, all_dy_u_embeddings)
            return user_seq_rep, all_u_embeddings
        elif self.with_st_graph:
            return user_st_seq_rep, all_st_u_embeddings[:self.user_size]
        elif self.with_dy_graph: 
            return user_dy_seq_rep, all_dy_u_embeddings[:self.user_size]
        else: 
            raise "Require at least one of the encoders"
            
    def UserDyEmbeddingMerger(self, inputs, all_dy_u_embeddings):
            
        all_u_dy_embeddings, _ = self.channel_attention_1(*all_dy_u_embeddings)
        
        user_seq_rep = F.embedding(inputs.to(DEVICE), all_u_dy_embeddings.to(DEVICE))
        user_seq_rep = self.dropout(user_seq_rep)
                
        return user_seq_rep, all_u_dy_embeddings
    
    def UserEmbeddingMerger(self, inputs, all_st_u_embeddings, all_dy_u_embeddings):
        
        all_u_embeddings, _ = self.channel_attention_2(*[all_st_u_embeddings[:self.user_size], all_dy_u_embeddings[:self.user_size]])
        
        user_seq_rep = F.embedding(inputs.to(DEVICE), all_u_embeddings.to(DEVICE))
        user_seq_rep = self.dropout(user_seq_rep)
                
        return user_seq_rep, all_u_embeddings

    def channel_attention_1(self, *channel_embeddings):
    
        weights = []
        for embedding in channel_embeddings:
            weights.append(
                torch.sum(torch.multiply(self.dy_att, torch.matmul(embedding, self.dy_att_m)),1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim = -1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings, score

    def channel_attention_2(self, *channel_embeddings):
    
        weights = []
        for embedding in channel_embeddings:
            weights.append(
                torch.sum(torch.multiply(self.merger_att, torch.matmul(embedding, self.merger_att_m)),1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim = -1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings, score

 