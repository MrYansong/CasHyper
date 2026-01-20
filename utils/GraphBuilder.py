
import math
import os
import pickle
import random
from collections import defaultdict

from torch_geometric.data import Data


import config.Constants as Constants
from config.Constants import *


def LoadGraph(data_name, Type):
    options = Options(data_name)
    _ui2idx = {}
    _idx2ui = []

    with open(options.ui2idx_dict, 'rb') as handle:
        _ui2idx = pickle.load(handle)
    with open(options.idx2ui_dict, 'rb') as handle:
        _idx2ui = pickle.load(handle)

    edges_list = []
    edges_type_list = []
    edges_weight_list = []

    if Type.find("item") != -1:
        with open(options.item_net_data, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            relation_list = [edge.split(',') for edge in relation_list]
            relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                            edge[0] in _ui2idx and edge[1] in _ui2idx]
            # print(relation_list)
            relation_list_reverse = [edge[::-1] for edge in relation_list]
            temp_edges_type_list = [0] * len(relation_list_reverse)
            edges_list += relation_list_reverse
            edges_type_list += temp_edges_type_list
            edges_weight_list+=[1.0] * len(relation_list_reverse)

    if Type.find("social") != -1:
        if os.path.exists(options.net_data) is False:
            print("There exists no social grpah!!")
        else:
            with open(options.net_data, 'r') as handle:
                relation_list = handle.read().strip().split("\n")
                relation_list = [edge.split(',') for edge in relation_list]
                relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                                edge[0] in _ui2idx and edge[1] in _ui2idx]
                relation_list_reverse = [edge[::-1] for edge in relation_list]
                temp_edges_type_list = [1] * len(relation_list_reverse)
                edges_list += relation_list_reverse
                edges_type_list += temp_edges_type_list
                edges_weight_list+=[1.0] * len(relation_list_reverse)
        print("load graph!")
    
    edges_list_tensor = torch.LongTensor(edges_list).t().to(DEVICE)
    edges_type = torch.LongTensor(edges_type_list).to(DEVICE)
    edges_weights = torch.FloatTensor(edges_weight_list).to(DEVICE)

    graph = Data(edge_index=edges_list_tensor, edge_type=edges_type, edges_weights=edges_weights)

    return graph
