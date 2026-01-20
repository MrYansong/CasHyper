import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv
from config.Constants import DEVICE

class NormalGraphNN(nn.Module):
    def __init__(self, ntoken, input_dim, hyper_dim, dropout_rate=0.1, self_embedding=True):
        super(NormalGraphNN, self).__init__()

        self.ntoken = ntoken
        self.input_dim = input_dim
        self.hyper_dim = hyper_dim
        
        self.self_embedding = self_embedding
        if self.self_embedding:
            self.embedding = nn.Embedding(self.ntoken, self.input_dim, padding_idx=0)
            init.xavier_normal_(self.embedding.weight)

        self.gnn1 = GCNConv(self.input_dim, self.input_dim * 2) 
        self.gnn2 = GCNConv(self.input_dim * 2, self.input_dim) 

        self.dropout = nn.Dropout(dropout_rate)
            
    def forward(self, node_embedding=None, graph=None):
        assert graph != None

        graph_edge_index = graph.edge_index.to(DEVICE)
        graph_x_embeddings = self.gnn1(self.embedding.weight if self.self_embedding else node_embedding, graph_edge_index) 
        graph_x_embeddings = F.normalize(graph_x_embeddings, dim=-1, p=2)
        graph_x_embeddings = self.dropout(graph_x_embeddings)
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)
        
        return graph_output.to(DEVICE)
    
