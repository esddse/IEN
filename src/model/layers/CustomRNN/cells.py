import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import math
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.rnn import RNNCellBase


def split_last(x, shape):
    ''' split the last dimension to given shape '''
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    ''' merge the last n_dims to a dimension '''
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class RNNCell(nn.Module):
    ''' basic rnn cell '''
    def __init__(self, input_dim, hidden_dim):
        super(RNNCell, self).__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.input_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.output_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.state_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

    def reset_parameters(self):
        pass

    def forward(self, input_, hx):
        hx, cx = hx
        input_ = self.input_transform(input_)
        c_next = input_ + cx
        h_next = self.output_transform(c_next)
        c_next = self.state_transform(c_next)
        return (h_next, c_next) 


class AttBlock(nn.Module):
    def __init__(self, input_dim_q, input_dim_k, input_dim_v, hidden_dim, head_num, dropout_rate):
        super(AttBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.head_num = head_num

        self.proj_q = nn.Linear(input_dim_q, hidden_dim)
        self.proj_k = nn.Linear(input_dim_k, hidden_dim)
        self.proj_v = nn.Linear(input_dim_v, hidden_dim)
        self.drop_att = nn.Dropout(dropout_rate)
        self.att_head_num = head_num

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.Tanh(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def attention(self, q, k, v, mask=None):
        ''' 
            args:
                q, k, v:    [batch_size, entity_size, hidden_dim] 
                mask: [batch_size, entity_size]
        '''
        pq, pk, pv = self.proj_q(q), self.proj_k(k), self.proj_v(v)
        q, k, v = (split_last(x, (self.att_head_num, -1)).transpose(1,2) for x in [pq, pk, pv]) # [batch_size, head_num, entity_size, head_dim]
        # (B, H, E, D) @ (B, H, D, E) -> (B, H, E, E)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float() # [batch_size, ?, ?, entity_size]
            scores -= 1e10 * (1.0 - mask)
        att = F.softmax(scores, dim=-1)
        scores = self.drop_att(att)
        # (B, H, E, E) @ (B, H, E, D) -> (B, H, E, D)
        h = (scores @ v).transpose(1, 2).contiguous()
        h = merge_last(h, 2)
        return h + pq, att

    def forward(self, q, k, v, mask=None):
        x, att = self.attention(q, k, v, mask)
        x = self.norm1(x)
        x = self.norm2(self.ff(x)+x)
        return x, att

class StateGraphGRUCell(nn.Module):
    ''' '''
    def __init__(self, input_dim, hidden_dim):
        super(StateGraphGRUCell, self).__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        # gate
        self.gate_r = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.gate_u = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        # c
        self.proj_h = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh()
        )

        att_input_dim = input_dim + hidden_dim
        self.att = AttBlock(att_input_dim, att_input_dim, att_input_dim, hidden_dim, 4, 0.5)

        self.reset_parameters() 

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    


    def forward(self, input_, hx):
        '''
            args:
                input_: [batch_size, entity_size, input_dim]
                hx:
                    hx: [batch_size, entity_size, hidden_dim]
            return:
                h_next: output
        '''

        ix = input_

        # --------- entity ---------

        # gate
        z = self.gate_u(torch.cat([ix, hx], dim=-1))
        r = self.gate_r(torch.cat([ix, hx], dim=-1))
        # update cx
        ix = torch.cat([ix, r*hx], dim=-1)
        h_next = self.proj_h(self.att(ix, ix, ix))
        # output 
        h_next = (1-z) * hx + z * h_next

        return h_next

class StateLocationGraphGRUCell(nn.Module):
    ''' '''
    def __init__(self, input_dim, hidden_dim):
        super(StateLocationGraphGRUCell, self).__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        
        # visualization
        self.attes = []
        self.attls = []

        # gate entity
        self.gate_re = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.gate_ue = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        # c
        self.proj_he = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # gate location
        self.gate_rl = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.gate_ul = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        # c
        self.proj_hl = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.att_e = AttBlock(input_dim+hidden_dim, input_dim+hidden_dim, input_dim+hidden_dim, hidden_dim, 4, 0.5)
        self.att_l = AttBlock(input_dim+hidden_dim, hidden_dim, hidden_dim, hidden_dim, 4, 0.5)

        self.reset_parameters() 

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def reset_visualization(self):
        self.attes, self.attls = [], []


    def forward(self, input_, hx, entity_size):
        '''
            args:
                input_: [batch_size, entity_size + location_size, input_dim]
                hx:
                    hx: [batch_size, entity_size + location_size, hidden_dim]
            return:
                h_next: output
        '''

        eix, lix = input_[:, :entity_size, :], input_[:, entity_size:, :]
        ehx, lhx = hx[:, :entity_size, :], hx[:, entity_size:, :]

        # --------- entity ---------

        # gate
        ez = self.gate_ue(torch.cat([eix, ehx], dim=-1))
        er = self.gate_re(torch.cat([eix, ehx], dim=-1))
        # update cx
        eix = torch.cat([eix, er*ehx], dim=-1)
        eh_next, atte = self.att_e(eix, eix, eix)
        eh_next = self.proj_he(eh_next)
        # output 
        eh_next = (1-ez) * ehx + ez * eh_next
        
        # --------- location ---------

        # gate
        lz = self.gate_ul(torch.cat([lix, lhx], dim=-1))
        lr = self.gate_rl(torch.cat([lix, lhx], dim=-1))
        # update cx
        lix = torch.cat([lix, lr*lhx], dim=-1)
        lh_next, attl = self.att_l(lix, eh_next, eh_next)
        lh_next = self.proj_hl(lh_next)
        # output 
        lh_next = (1-lz) * lhx + lz * lh_next

        h_next = torch.cat([eh_next, lh_next], dim=-2)

        self.attes.append(atte)
        self.attls.append(attl)

        return h_next


class LocationGraphGRUCell(nn.Module):
    ''' '''
    def __init__(self, input_dim, hidden_dim):
        super(LocationGraphGRUCell, self).__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        # visualization
        self.attes = []
        self.attls = []

        # gate entity
        self.gate_re = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.gate_ue = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        # c
        self.proj_he = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # gate location
        self.gate_rl = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.gate_ul = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        # c
        self.proj_hl = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.att_l = AttBlock(input_dim+hidden_dim, hidden_dim, hidden_dim, hidden_dim, 4, 0.5)

        self.reset_parameters() 

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def reset_visualization(self):
        self.attes, self.attls = [], []    


    def forward(self, input_, hx, entity_size):
        '''
            args:
                input_: [batch_size, entity_size + location_size, input_dim]
                hx:
                    hx: [batch_size, entity_size + location_size, hidden_dim]
            return:
                h_next: output
        '''

        eix, lix = input_[:, :entity_size, :], input_[:, entity_size:, :]
        ehx, lhx = hx[:, :entity_size, :], hx[:, entity_size:, :]

        # --------- entity ---------

        # gate
        ez = self.gate_ue(torch.cat([eix, ehx], dim=-1))
        er = self.gate_re(torch.cat([eix, ehx], dim=-1))
        # update cx
        eix = torch.cat([eix, er*ehx], dim=-1)
        eh_next = self.proj_he(eix)
        # output 
        eh_next = (1-ez) * ehx + ez * eh_next
        
        # --------- location ---------

        # gate
        lz = self.gate_ul(torch.cat([lix, lhx], dim=-1))
        lr = self.gate_rl(torch.cat([lix, lhx], dim=-1))
        # update cx
        lix = torch.cat([lix, lr*lhx], dim=-1)
        lh_next, attl = self.att_l(lix, eh_next, eh_next)
        lh_next = self.proj_hl(lh_next)
        # output 
        lh_next = (1-lz) * lhx + lz * lh_next

        h_next = torch.cat([eh_next, lh_next], dim=-2)

        self.attls.append(attl)

        return h_next

