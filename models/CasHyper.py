import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Encodings import TimeEncoder
from models.Encoders import GraphEncoder
from models.Decoders import FusionDecoder

class CasHyper(nn.Module):
    def __init__(self,opt):
        super(CasHyper, self).__init__()
        
        # hypers 
        self.opt = opt
        self.dataset = opt.data
        self.user_size = opt.user_size
        self.ninp = opt.d_word_vec
        self.pos_dim = opt.pos_dim
        self.transformer_dim = opt.transformer_dim
        self.dropout_rate=opt.dropout_rate
        self.__name__ = "CasHyper"
        
        # dropout module
        self.dropout = nn.Dropout(self.dropout_rate)
        self.drop_timestamp = nn.Dropout(self.dropout_rate)

        # Encoder modules 
        self.encoder = GraphEncoder(opt)
        self.time_encoder = TimeEncoder(opt)
        self.decoder = FusionDecoder(opt)
        
        print(self)

    def forward(self, inputs, input_timestamp, input_id, epoch, train=True, static_graph=None, diffusion_graph =None):

        inputs = inputs[:, :-1]

        user_embedding, all_token_embeddings = self.encoder(inputs, static_graph=static_graph, diffusion_graph= diffusion_graph)

        time_embedding, input_timestamp = self.time_encoder(inputs, input_timestamp, train)

        if train:
            output, mseloss = self.decoder(user_embedding, all_token_embeddings, time_embedding, inputs, input_timestamp, input_id, epoch, train)
            user_pred = output.view(-1, output.size(-1))
            return user_pred, mseloss
        else:
            output = self.decoder(user_embedding, all_token_embeddings, time_embedding, inputs, input_timestamp, input_id, epoch, train)
            user_pred = output.view(-1, output.size(-1))
            return user_pred
