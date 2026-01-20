
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.loss import _Loss

from config.Constants import DEVICE
import config.Constants as Constants

def get_performance(crit, pred, gold):
    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]

    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(
        gold.ne(Constants.PAD).data).sum().float()

    true_set = set()
    for items in gold.cpu().numpy().tolist():
        true_set.add(items)
    pre_set = set()
    for item in pred.cpu().numpy().tolist():
        if item in true_set:
            pre_set.add(item)

    return loss, n_correct, len(pre_set), len(true_set)

class LossComputing(_Loss):
    def __init__(self, opt):
        super(LossComputing, self).__init__()

        self.UserPrediction_loss = 0
        self.additional_loss = 0

        self.UserLoss_func = nn.CrossEntropyLoss(size_average=False, ignore_index=Constants.PAD)
        if torch.cuda.is_available():
            self.UserLoss_func = self.UserLoss_func.to(DEVICE)

    def UserLoss(self, user_pred, user_gold):

        loss = self.UserLoss_func(user_pred, user_gold.contiguous().view(-1))
        return loss

    def forward(self, user_pred, user_gold, additional_loss=0):
    
        self.UserPrediction_loss = self.UserLoss(user_pred, user_gold)
        self.Additional_loss = additional_loss


        loss = self.UserPrediction_loss + self.Additional_loss

        user_pred = user_pred.max(1)[1]
        user_gold = user_gold.contiguous().view(-1)
        n_correct = user_pred.data.eq(user_gold.data)
        n_correct = n_correct.masked_select(
            user_gold.ne(Constants.PAD).data).sum().float()

        true_set = set()
        for items in user_gold.cpu().numpy().tolist():
            true_set.add(items)
        pre_set = set()
        for item in user_pred.cpu().numpy().tolist():
            if item in true_set:
                pre_set.add(item)

        return loss, n_correct, len(pre_set), len(true_set)



