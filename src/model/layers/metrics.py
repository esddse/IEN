
from collections import Counter

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_TP_FP_FN(pred, gold, TP, FP, FN):
    '''
        pred: [label1, label2]
        gold: [label2, label2]
        TP, FP, FN: {label1: int, label2: int} 
    '''
    for i in range(pred.size(0)):
        p = pred[i].item()
        g = gold[i].item()
        if p == g:
            TP[p] += 1 
        else:
            FP[p] += 1 
            FN[g] += 1

def get_P_R_F1(TP, FP, FN):
    ''' TP, FP, FN: {label1: int, label2: int}   '''
    items = {item for S in [TP, FP, FN] for item in S} 
    epsilon = 1e-30
    results = {}
    for item in items: 
        results[item] = {}
        results[item]["P"] = TP[item] / (TP[item] + FP[item] + epsilon)
        results[item]["R"] = TP[item] / (TP[item] + FN[item] + epsilon)
        results[item]["F"] = 2 * results[item]["P"] * results[item]["R"] / (results[item]["P"] + results[item]["R"] + epsilon)
    return results