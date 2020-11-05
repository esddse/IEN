import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from allennlp.commands.elmo import ElmoEmbedder

from model.layers.CustomRNN.cells import StateLocationGraphGRUCell, RNNCell, LocationGraphGRUCell
from model.layers.CustomRNN.rnn import CustomRNN
from model.layers.LSTMEncoder import LSTMEncoder
from model.layers.CRF import CRF

class IENModel(nn.Module):
    def __init__(self, config):
        super(IENModel, self).__init__()
        self.use_elmo = config.use_elmo
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.label_size = config.label_size 
        self.embed_fropout_rate = config.embed_dropout_rate
        self.dropout_rate = config.dropout_rate
        self.batch_size = config.batch_size
        self.max_word_length = config.max_word_length
        self.max_sent_length = config.max_sent_length
        self.bidirectional = config.bidirectional

        # visualization 
        self.attes_forward, self.attes_backward = [], []
        self.attls_forward, self.attls_backward = [], []

        self.elmo = ElmoEmbedder(options_file=config.path_elmo_options, weight_file=config.path_elmo_weights, cuda_device=-1 if not config.gpu_id else 0)
        self.alpha = nn.Parameter(torch.randn(3, 1))
        if config.path_word_vector:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(config.embeddings))
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(self.dropout_rate)
        self.word_lstm = LSTMEncoder(self.embedding_dim+1, self.hidden_dim, self.batch_size, self.max_word_length)

        # entity state
        self.state_location_rnn = CustomRNN(cell_class=StateLocationGraphGRUCell, input_dim=self.hidden_dim*4, hidden_dim=self.hidden_dim, batch_first=True, bidirectional=self.bidirectional)

        self.state_mlp = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim*(2 if self.bidirectional else 1), self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.label_size),
            )
        self.crf = CRF(self.label_size)

        self.location_mlp_1 = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim*(2 if self.bidirectional else 1), self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
            )
        self.location_mlp_2 = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim*(2 if self.bidirectional else 1), self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
            )



    def forward(self, words, words_idxs, verbs, words_length, sents_length, verbs_idxs_sents, entity_idxs_sents, entity_to_idx, idx_to_entity, location_candidate_idxs_sents, location_candidate_to_idx, idx_to_location_candidate):
        ''' '''
        device = self.embedding.weight.device 
        words_idxs = torch.LongTensor(words_idxs).to(device)   # [max_word_length, 1]
        verbs = torch.FloatTensor(verbs).to(device)  # [max_word_length]
        
        # embedding
        if self.use_elmo:
            with torch.no_grad():
                words = [[word[0] for word in words]]
                word_embeddings_elmo, word_mask = self.elmo.batch_to_embeddings(words)
                word_embeddings_elmo = word_embeddings_elmo.permute(0, 2, 3, 1) @ self.alpha 
                word_embeddings_elmo = word_embeddings_elmo.view(-1, 1024)
            word_embeddings = word_embeddings_elmo
        else:
            word_embeddings = self.embedding(words_idxs)  
        word_embeddings = self.embed_drop(word_embeddings)
        word_verb_embeddings = torch.cat([word_embeddings, verbs.unsqueeze(dim=-1)], dim=-1).unsqueeze(dim=0)

        # lstm word encoding
        word_lstm_encoding = self.word_lstm(word_verb_embeddings, [words_length]).squeeze(dim=0) # [max_length, 2*hidden_dim]

        #-------------------------------------#
        #       entity state tracking         #
        #-------------------------------------#


        # rnn verb encoding
        verbs_sents = []
        for idxs in verbs_idxs_sents:
            if not idxs:
                verbs = torch.zeros(self.hidden_dim*2).to(device)
            else:
                idxs = torch.stack(idxs).view(1, -1).to(device)
                verbs = torch.mean(F.embedding(idxs, word_lstm_encoding), dim=1).view(-1)
            verbs_sents.append(verbs)
        start_encoding = torch.zeros(verbs.size()).to(device)
        end_encoding = torch.zeros(verbs.size()).to(device)
        verbs_sents = [start_encoding] + verbs_sents + [end_encoding]
        verbs_sents = torch.stack(verbs_sents) # [seq_len, hidden_dim*2]
        
        # rnn entity encoding
        entity_states_input = [None for i in range(len(entity_to_idx))] # placeholder
        for entity, entity_idxs in entity_idxs_sents.items():
            entity_sents = []
            for i, idxs in enumerate(entity_idxs):
                if not idxs:
                    entity_encoding = torch.zeros(self.hidden_dim*4).to(device)
                else:
                    entity_encoding = word_lstm_encoding[idxs[0].item():idxs[1].item()+1]
                    entity_encoding = torch.mean(entity_encoding, dim=0).view(-1)
                    entity_encoding = torch.cat([entity_encoding, verbs_sents[i]], dim=-1)   # add verb
                entity_sents.append(entity_encoding)
            start_encoding = torch.zeros(entity_encoding.size()).to(device)
            end_encoding = torch.zeros(entity_encoding.size()).to(device)
            entity_sents = [start_encoding] + entity_sents + [end_encoding]
            entity_sents = torch.stack(entity_sents).unsqueeze(dim=0) # [batch_size, seq_len, hidden_dim*4]
            entity_states_input[entity_to_idx[entity]] = entity_sents.unsqueeze(dim=-2)
        # concat all entities
        entity_states_input = torch.cat(entity_states_input, dim=-2)  # [batch_size, seq_len, entity_size, hidden_dim*4]

        #-------------------------------------#
        #       entity location tracking      #
        #-------------------------------------#

        location_candidates_input = [None for i in range(len(location_candidate_to_idx))]
        for location_candidate, location_candidate_idxs in location_candidate_idxs_sents.items():
            location_sents = []
            for i, idxs in enumerate(location_candidate_idxs):
                if not idxs:
                    location_encoding = torch.zeros(self.hidden_dim*4).to(device)
                else:
                    location_encoding = word_lstm_encoding[idxs[0].item(): idxs[1].item()+1]
                    location_encoding = torch.mean(location_encoding, dim=0).view(-1)
                    location_encoding = torch.cat([location_encoding, verbs_sents[i]], dim=-1) # add verb
                location_sents.append(location_encoding)
            start_encoding = torch.zeros(location_encoding.size()).to(device)
            end_encoding = torch.zeros(location_encoding.size()).to(device)
            location_sents = [start_encoding] + location_sents + [end_encoding]
            location_sents = torch.stack(location_sents).unsqueeze(dim=0) # [batch_size, sents_len, hidde_dim * 4]
            location_candidates_input[location_candidate_to_idx[location_candidate]] = location_sents.unsqueeze(dim=-2)
        # concat all location
        location_candidates_input = torch.cat(location_candidates_input, dim=-2)

        states_locations_input = torch.cat([entity_states_input, location_candidates_input], dim=-2) # [batch_size, sents_len, entity_size + location_size, hidden_dim * 4]
        

        # states location rnn encoding
        states_locations_rnn_encoding, _ = self.state_location_rnn(states_locations_input, entity_size=len(entity_to_idx))
        entity_states_rnn_encoding = states_locations_rnn_encoding[:, :, :len(entity_to_idx), :]  # [batch_size, sents_len, entity_size, hidden_dim*2]
        location_candidates_rnn_encoding = states_locations_rnn_encoding[:, :, len(entity_to_idx):, :] # [batch_size, sents_len, location_size, hidden_dim*2]

        # visualization
        self.attes_forward  = self.state_location_rnn.attes_forward
        self.attes_backward = self.state_location_rnn.attes_backward
        self.attls_forward  = self.state_location_rnn.attls_forward
        self.attls_backward = self.state_location_rnn.attls_backward

        # entity states logit
        states_logit = self.state_mlp(entity_states_rnn_encoding)
        entity_states_logit = {}
        for entity in entity_to_idx:
            entity_states_logit[entity] = states_logit[:, :, entity_to_idx[entity], :].squeeze(dim=-2)

        # entity location logit
        entity_locations_logit = {}
        entity_states = self.location_mlp_1(entity_states_rnn_encoding).squeeze(dim=0) # [sents_len, entity_size, hidden_dim]
        entity_locations = self.location_mlp_2(location_candidates_rnn_encoding).squeeze(dim=0) # [sents_len, entity_size, hidden_dim]
        relevence_mat = torch.bmm(entity_states, entity_locations.transpose(1,2)) # [sents_len, entity_size, location_size]
        entity_locations_logit = {}
        for entity in entity_to_idx:
            entity_locations_logit[entity] = relevence_mat[:, entity_to_idx[entity], :].contiguous().view(1, sents_length[0], -1)

        return entity_states_logit, entity_locations_logit
        

if __name__ == '__main__':
    pass