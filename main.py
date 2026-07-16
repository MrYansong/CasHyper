import os, sys, time

import numpy as np
import torch
import torch.nn as nn
import pynvml

import config.Constants as Constants
from config.Constants import (ConfigFromParser, Options, DEVICE)

from models.CasHyper import CasHyper

from utils.DataConstructor import DataConstructor, Split_data
from utils.GraphBuilder import *
from utils.LossUtil import LossComputing
from utils.Metrics import Metrics
from utils.Optim import TunedScheduledOptim




def train_epoch(model, training_data, optimizer, epoch, criterion, static_graph, diffusion_graph):
    ''' Epoch operation in training phase'''
    
    model.train()

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0
    total_same_user = 0.0
    n_total_uniq_user = 0.0
    batch_num = 0.0
    
    LossFunction = criterion

    for i, batch in enumerate(training_data):
        tgt, tgt_timestamp, tgt_id = batch
        tgt.to(DEVICE); tgt_timestamp.to(DEVICE); tgt_id.to(DEVICE)
        user_gold = tgt[:, 1:].to(DEVICE)

        n_words = user_gold.data.ne(Constants.PAD).sum().float()
        n_total_words += n_words
        
        optimizer.zero_grad()
            
        user_pred, mseloss = model(tgt, 
                          tgt_timestamp, 
                          tgt_id,
                          epoch = epoch, 
                          train=True, 
                          static_graph=static_graph, 
                          diffusion_graph = diffusion_graph)
        
        # get loss and backward
        loss, n_correct, same_user, input_users = LossFunction(user_pred, user_gold)
        loss += mseloss
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate(epoch)

        # note keeping
        batch_num += tgt.size(0)
        n_total_correct += n_correct
        total_loss = total_loss + loss.item()
        total_same_user += same_user
        n_total_uniq_user += input_users

    return total_loss / n_total_words, n_total_correct / n_total_words, total_same_user / n_total_uniq_user

def test_epoch(model, validation_data, epoch, static_graph, diffusion_graph, k_list=[10, 50, 100]):
    ''' Epoch operation in evaluation phase '''
    model.eval()
    
    metric = Metrics()
    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0
        
    n_total_words = 0
    for i, batch in enumerate(validation_data):
        tgt, tgt_timestamp, tgt_id = batch
        tgt.to(DEVICE); tgt_timestamp.to(DEVICE); tgt_id.to(DEVICE)
        user_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()
        
        user_pred = model(tgt, 
                                                                tgt_timestamp, 
                                                                tgt_id, 
                                                                epoch = epoch, 
                                                                train=False, 
                                                                static_graph=static_graph, 
                                                                diffusion_graph = diffusion_graph)

        user_pred = user_pred.detach().cpu().numpy()
        scores_batch, scores_len = metric.compute_metric(user_pred, user_gold, k_list)

        n_total_words += scores_len
        for k in k_list:
            scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
            scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    return scores

def train_model(data_path):
    user_size, total_cascades, timestamps, train, valid, test = Split_data(data_path, opt.train_ratio, opt.valid_ratio, load_dict=True, max_len=opt.max_len)

    train_data = DataConstructor(train, batch_size=opt.batch_size, load_dict=True, cuda=False, _need_shsuffle=True , max_len = opt.max_len)
    valid_data = DataConstructor(valid, batch_size=opt.batch_size, load_dict=True, cuda=False, _need_shsuffle=False, max_len = opt.max_len)
    test_data  = DataConstructor(test , batch_size=opt.batch_size, load_dict=True, cuda=False, _need_shsuffle=False, max_len = opt.max_len)
    opt.user_size = user_size
    option = Options(opt.data_path)
    _ui2idx = {}
    with open(option.ui2idx_dict, 'rb') as handle:
        _ui2idx = pickle.load(handle)
    opt.token_size = len(_ui2idx)

    static_graph = LoadGraph(opt.data_path, Type=opt.static_graph_type)
    dynamic_graph = LoadGraph(opt.data_path,Type=opt.dynamic_graph_type)

    model = CasHyper(opt)
    print("The model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    params = filter(lambda p: p.requires_grad, model.parameters())
    Origin_optimizer = torch.optim.Adam(params, betas=(0.9, 0.98),weight_decay = opt.l2, eps=1e-09)
    optimizer = TunedScheduledOptim(Origin_optimizer, opt.d_model, opt.n_warmup_steps, data_path) 
    criterion = LossComputing(opt)

    if torch.cuda.is_available():
        model = model.to(DEVICE)
        criterion = criterion.to(DEVICE)
    
    test_history = 0.0
    val_history = 0.0
    patience = 0.0
    best_scores = {}
    for epoch_i in range(opt.epoch):
        print('\n[ Epoch', epoch_i, ']')
        start = time.time()
        train_loss, train_accu, train_pred = train_epoch(model, train_data, optimizer,epoch_i,criterion,static_graph,dynamic_graph)
        print('  - (Training)   loss: {loss: 4.3f}, accuracy: {accu:3.3f} %, predected:{pred:3.3f} %, elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu, pred=100 * train_pred,
            elapse=(time.time() - start) / 60))
        
        if epoch_i >= 0:
            start = time.time()
            val_scores = test_epoch(model, valid_data, epoch_i, static_graph, dynamic_graph)
            
            # validate the models
            print('  - ( Validation )) ')
            for metric in val_scores.keys():
                print(metric + ' ' + str(val_scores[metric]))
            print("Validation use time: ", (time.time() - start) / 60, "min")
            if val_history <= sum(val_scores.values()):
                val_history = sum(val_scores.values())
                patience = 0.0
            else:
                patience += 1
                if patience == 50:
                    print('Early stopping! ')
                    break

            # test the model
            print('  - (Test) ')
            scores = test_epoch(model, test_data, epoch_i, static_graph, dynamic_graph)
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
            
            if test_history <= sum(scores.values()):
                print("Best Validation hit@100:{} at Epoch:{}".format(scores["hits@100"], epoch_i))
                test_history = sum(scores.values())
                best_scores = scores
                print("Save best model!!!") 

    print(" -(Finished!!) \n Best scores: ")        
    for metric in best_scores.keys():
        print(metric + ' ' + str(best_scores[metric]))


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        # pass

if __name__ == "__main__":
    opt = ConfigFromParser()

    print('start..................')
    log_name = './log/'+ opt.data + '.log'
    sys.stdout = Logger(log_name, sys.stdout)
    for key, value in vars(opt).items():
        print(f"{key}: {value}")
    train_model(opt.data_path)

    # 
    
