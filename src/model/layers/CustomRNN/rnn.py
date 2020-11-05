import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

import copy
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.layers.CustomRNN.cells import RNNCell


class CustomRNN(nn.Module):
    ''' ref: https://github.com/Shivanshu-Gupta/Pytorch-POS-Tagger/blob/master/rnn.py '''
    def __init__(self, cell_class, input_dim, hidden_dim, batch_first=False, bidirectional=True, **kwargs):
        super(CustomRNN, self).__init__()
        self.cell_class  = cell_class
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        # visualization
        self.attes_forward, self.attes_backward = [], []
        self.attls_forward, self.attls_backward = [], []

        self.cell = cell_class(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)
        self.cell.reset_parameters()
        if bidirectional:
            self.cell_rev = cell_class(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)
            self.cell_rev.reset_parameters()
        
    def reset_visualization(self):
        self.attes_forward, self.attes_backward = [], []
        self.attls_forward, self.attls_backward = [], []

    def forward(self, input_, hx=None, **kwargs):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, *data_size = input_.size()
        if hx is None:
            h_size = [batch_size] + data_size[:-1] + [self.hidden_dim]
            hx = autograd.Variable(input_.data.new(*h_size).zero_(), requires_grad=False)
            if self.cell_class == RNNCell:
                hx = (hx, hx)
        output, h_n, self.attes_forward, self.attls_forward = self._forward_rnn_no_mask(cell=self.cell, input_=input_, hx=hx, **kwargs)

        if self.bidirectional:
            input_rev = torch.flip(input_, [0])
            hx_rev = copy.deepcopy(hx)
            output_rev, h_n_rev, self.attes_backward, self.attls_backward = self._forward_rnn_no_mask(cell=self.cell_rev, input_=input_rev, hx=hx_rev, **kwargs)
            self.attes_backward.reverse()
            self.attls_backward.reverse()
            output = torch.cat([output, torch.flip(output_rev, [0])], dim=-1)
            # TODO: h_n
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, h_n

    def _forward_rnn_no_mask(self, cell, input_, hx, **kwargs):
        max_time = input_.size(0)
        output = []
        cell.reset_visualization()
        for time in range(max_time):
            if isinstance(cell, RNNCell):
                h_next, c_next = cell(input_[time], hx=hx, **kwargs)
                hx = (h_next, c_next)
            else:
                h_next = cell(input_=input_[time], hx=hx, **kwargs)
                hx = h_next
            output.append(h_next)
        output = torch.stack(output)
        return output, hx, cell.attes, cell.attls

# ============= main ===================

if __name__ == '__main__':

    max_len = 50
    input_dim = 10
    hidden_dim = 10
    entity_size = 20
    batch_size = 1
    step = 1000

    # rnn = CustomRNN(StateGraphGRUCell, input_dim, hidden_dim, batch_first=True)
    rnn = CustomRNN(RNNCell, input_dim, hidden_dim, batch_first=True)
    # rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(rnn.parameters(), lr=1e-4)

    for i in range(step):
        input_ = torch.randn(batch_size, max_len, entity_size, input_dim)
        gold = torch.zeros_like(input_)
        j = 0
        for k in range(max_len):
            gold[:, k, 0, :] = input_[:, k, j, :]
            j = (j+1)%entity_size

        # gold   = torch.tanh(input_ * 2 - 1) 
        output, h = rnn(input_)
        # h0 = torch.zeros(1, batch_size, hidden_dim)
        # output, h = rnn(input_, h0)
        loss = loss_func(output, gold)

        # print(output)
        print("step: %d, loss: %.4f" % (i, loss))
        loss.backward()
        # for name, x in rnn.named_parameters():
        #     if x.grad is not None:
        #         print(name, torch.mean(x.grad).item())
        optimizer.step()
        
