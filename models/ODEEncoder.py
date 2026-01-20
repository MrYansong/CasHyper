
import torch.nn.functional as F
import torch.nn.init as init

from torch_geometric.nn import Sequential, GCNConv
from models.hypercomplex_func import Quaternion, hypercomplex_Linear, Octonion, Complex, Sedenion

from config.Constants import DEVICE

# from torchdiffeq import odeint_adjoint as odeint
from torchdyn.core import NeuralODE
from torchdyn.utils import *


class NodeEncoder(nn.Module):
    """Node ODE function."""

    def __init__(self, input_dim, output_size, hyper_dim, node_size, dropout=0.9):
        super(NodeEncoder, self).__init__()

        self.input_dim = input_dim
        self.hyper_dim = hyper_dim
        self.node_size = node_size
        self.out_dim = output_size // hyper_dim
        self.layer_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.dropout = nn.Dropout(dropout)
        self.gnn1 = GCNConv(input_dim, input_dim)

        self.w_node2edge = nn.Linear(input_dim * 2, input_dim)
        self.init_network_weights(self.w_node2edge)
        self.w_edge2value = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.SiLU(),
            nn.Linear(self.input_dim // 2, 1))
        self.init_network_weights(self.w_edge2value)

    def init_network_weights(self, net, std=0.1):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=std)
                nn.init.constant_(m.bias, val=0)

    def set_x0(self, x_0):
        self.x_0 = x_0.clone().detach()

    def set_edge_index(self, edge_index):
        self.edge_index = edge_index
        self.node_send, self.node_rec = edge_index[0], edge_index[1]

    def rel_rec_compute(self, node_inputs):
        rel_send, rel_rec = F.embedding(self.node_send.to(DEVICE), node_inputs.to(DEVICE)), F.embedding(self.node_rec.to(DEVICE), node_inputs.to(DEVICE))
        
        rel_send = rel_send.view(rel_send.shape[0], self.hyper_dim, -1)
        rel_rec = rel_rec.view(rel_rec.shape[0], self.hyper_dim, -1)
        if self.hyper_dim == 4:
            edges = Quaternion(rel_send).multiply(Quaternion(rel_rec)).tensor.flatten(start_dim=-2)
        elif self.hyper_dim == 8:
            edges = Octonion(rel_send).multiply(Octonion(rel_rec)).tensor.flatten(start_dim=-2)
        elif self.hyper_dim == 2:
            edges = Complex(rel_send).multiply(Complex(rel_rec)).tensor.flatten(start_dim=-2)
        elif self.hyper_dim == 16:
            edges = Sedenion(rel_send).multiply(Sedenion(rel_rec)).tensor.flatten(start_dim=-2)
        edges_from_node = F.silu(edges)
        
        edges_z = self.dropout(edges_from_node)
        edge_2_value = torch.squeeze(F.sigmoid(self.w_edge2value(edges_z)), dim=-1)

        return edge_2_value 

    def set_edge_weight0(self, edge_weights):
        self.edge_weight0 = edge_weights

    def forward(self, node_inputs, perturbed=False):
        edge_weight = self.rel_rec_compute(node_inputs)
        inputs = self.layer_norm(node_inputs)
        x_hidden = self.gnn1(inputs, self.edge_index, edge_weight)
        x_hidden = self.dropout(x_hidden)

        return x_hidden - inputs


class CoupledODEFunc(nn.Module):
    def __init__(self, node_ode_func_net, hyper_dim, node_size, dropout_rate):
        super(CoupledODEFunc, self).__init__()
        self.hyper_dim = hyper_dim
        self.node_size = node_size
        self.dropout = nn.Dropout(dropout_rate)

        self.node_ode_func_net = node_ode_func_net

        self.alpha = nn.Parameter(0.8 * torch.ones(self.node_size))
        self.w = nn.Parameter(torch.eye(self.node_ode_func_net.input_dim))
        self.d = nn.Parameter(torch.ones(self.node_ode_func_net.input_dim))

    def set_edge_index(self, edge_index):
        self.node_ode_func_net.set_edge_index(edge_index)

    def set_edge_weight(self, edge_weight):
        self.node_ode_func_net.set_edge_weight0(edge_weight)

    def set_x0(self, x_0):
        self.x_0 = x_0.clone().detach()


    def init_network_weights(self, net, std=0.1):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=std)
                nn.init.constant_(m.bias, val=0)

    def forward(self, input_embedding):

        alpha = torch.sigmoid(self.alpha).unsqueeze(-1)
        assert (not torch.isnan(input_embedding).any())

        node_attributes = input_embedding
        grad_node = self.node_ode_func_net(node_attributes)  # [K*N,D]
        assert (not torch.isnan(grad_node).any())

        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        grad_node2 = torch.einsum('il, lm->im', input_embedding, w)
        grad_node2 = self.dropout(grad_node2)

        return alpha / 2 * grad_node - input_embedding + grad_node2 - input_embedding


class DynamicGraphODE(nn.Module):
    def __init__(self, ntoken, input_dim, hyper_dim, t_size=5, dropout_rate=0.1, self_embedding=True):
        super(DynamicGraphODE, self).__init__()

        # hypers for the model
        self.input_dim = input_dim
        self.token_size = ntoken
        self.hyper_dim = hyper_dim
        self.t_size = t_size
        self.t_span = torch.linspace(0, 1, t_size)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.input_Lin = nn.Linear(self.input_dim, self.input_dim * 2)
        self.output_Lin = nn.Linear(self.input_dim * 2, self.input_dim)
        self.node_ode_func_net = NodeEncoder(input_dim=self.input_dim*2, output_size=self.input_dim*2, hyper_dim=self.hyper_dim,
                                             node_size=self.token_size, dropout=0.8)

        self.NeuralFunc = CoupledODEFunc(
            node_ode_func_net=self.node_ode_func_net, hyper_dim=self.hyper_dim,
            node_size=self.token_size, dropout_rate=0.1)
        self.neuralDE = NeuralODE(self.NeuralFunc, solver='rk4')  #

    def forward(self, node_embeddings, input_graph):
        assert input_graph != None
        edge_index = input_graph.edge_index.to(DEVICE)
        edge_weights = input_graph.edges_weights.to(DEVICE)

        graph_x_embeddings = self.input_Lin(node_embeddings)
        graph_x_embeddings = F.normalize(graph_x_embeddings, dim=-1, p=2)
        graph_x_embeddings = self.dropout(graph_x_embeddings)

        self.NeuralFunc.set_x0(graph_x_embeddings)
        self.node_ode_func_net.set_x0(graph_x_embeddings)
        self.NeuralFunc.set_edge_index(edge_index)
        self.NeuralFunc.set_edge_weight(edge_weights)
        _, graph_x_embeddings = self.neuralDE(graph_x_embeddings, t_span=self.t_span)

        output_graph_embeddings = []
        for i in range(1, self.t_size):  
            user_dynamic_embedding = self.output_Lin(F.silu(graph_x_embeddings[i])).to(DEVICE)
            output_graph_embeddings.append(user_dynamic_embedding)

        return output_graph_embeddings, output_graph_embeddings[-1]

