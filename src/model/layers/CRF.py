
from itertools import zip_longest

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

def potential_index(y, num_tags, SOS_IDX):
    batch_size, max_len = y.size()
    for t in range(max_len-1, 0, -1):
        y[:, t] += y[:, t-1] * num_tags
    y[:, 0] += SOS_IDX * num_tags
    return y 

class CRF(nn.Module):
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.PAD_IDX = 0
        self.SOS_IDX = 1
        self.EOS_IDX = 2
        self.num_tags = num_tags
        # self.transition = nn.Parameter(torch.ones(num_tags, num_tags) * 1/num_tags)
        self.transition = nn.Parameter(torch.zeros(num_tags, num_tags))

    def forward(self, h, y, mask):
        pass
         

    def nll(self, h, y, lengths, mask):
        '''
            计算 negative log likelyhood
            参数:
                h:       [batch_size, max_len, num_tags]
                y:       [batch_size, max_len]
                lengths: [batch_size]
                mask:    [batch_size, max_len]
            返回:
                loss: float
        '''
        device = h.device
        batch_size, max_len = y.size()

        # crf_scores: [batch_size, max_len, num_tags, num_tags]
        crf_scores = h.unsqueeze(2).expand(-1, -1, self.num_tags, -1) + self.transition.view(1,1,self.num_tags,self.num_tags)
        y = potential_index(y, self.num_tags, self.SOS_IDX)

        # golden
        y = y.masked_select(mask) # [sum(real_len)]
        flatten_scores = crf_scores.masked_select(
            mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores)
        ).view(-1, self.num_tags*self.num_tags).contiguous() # [sum(real_len), num_tags*num_tags]
        golden_scores = flatten_scores.gather(dim=1, index=y.unsqueeze(1)).sum() 


        # partition
        alpha_t = torch.zeros(batch_size, self.num_tags).to(device) # 前向变量
        for t in range(max_len):
            batch_size_t = (lengths > t).sum().item()
            if t == 0:
                alpha_t[:batch_size_t] = crf_scores[:batch_size_t, t, self.SOS_IDX, :]
            else:
                alpha_t[:batch_size_t] = torch.logsumexp(
                    crf_scores[:batch_size_t, t, :, :]+alpha_t[:batch_size_t].unsqueeze(2),
                    dim=1
                )
        Z = alpha_t[:, self.EOS_IDX].sum()

        # loss
        loss = (Z - golden_scores) / batch_size 
        return loss

        
    def decode(self, h, lengths, mask): 
        '''
            viterbi decode
            参数：
                h:       [batch_size, max_len, num_tags]
                lengths: [batch_size]
                mask:    [batch_size, max_len]
        '''

        device = h.device
        batch_size, max_len = mask.size()
        crf_scores = h.unsqueeze(2).expand(-1, -1, self.num_tags, -1) + self.transition.view(1,1,self.num_tags,self.num_tags)

        # viterbi decode
        viterbi = torch.zeros(batch_size, max_len, self.num_tags).to(device)
        backpointer = (torch.ones(batch_size, max_len, self.num_tags).long() * self.EOS_IDX).to(device)
        for t in range(max_len):
            batch_size_t = (lengths > t).sum().item()
            if t == 0:
                viterbi[:batch_size_t, t, :] = crf_scores[:batch_size_t, t, self.SOS_IDX, :]
                backpointer[:batch_size_t, t, :] = self.SOS_IDX
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, t-1, :].unsqueeze(2)+crf_scores[:batch_size_t, t, :, :],  # [batch_size, num_tags, num_tags]
                    dim=1
                )
                viterbi[:batch_size_t, t, :] = max_scores
                backpointer[:batch_size_t, t, :] = prev_tags

        # backtrace
        backpointer = backpointer.view(batch_size, -1) # [batch_size, max_len * num_tags]
        tag_idxs = []
        tags_t = None
        for t in range(max_len-1, 0, -1):
            batch_size_t = (lengths > t).sum().item()
            if t == max_len-1:
                idxs = (torch.ones(batch_size_t).long() * (t * self.num_tags)).to(device)
                idxs += self.EOS_IDX
            else:
                prev_batch_size_t = len(tags_t)
                new_in_batch = torch.LongTensor([self.EOS_IDX] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat([tags_t, new_in_batch], dim=0)
                idxs = (torch.ones(batch_size_t).long() * (t * self.num_tags)).to(device)
                idxs += offset.long()

            tags_t = backpointer[:batch_size_t].gather(dim=1, index=idxs.unsqueeze(1).long()).squeeze(1)
            tag_idxs.append(tags_t.tolist())

        tag_idxs = list(zip_longest(*reversed(tag_idxs), fillvalue=self.EOS_IDX))
        tag_idxs = [list(idxs) + [self.EOS_IDX] for idxs in tag_idxs]
        tag_idxs = torch.LongTensor(tag_idxs).to(device)
        return tag_idxs
