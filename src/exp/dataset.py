import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import math
import random

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

from util.path import *
from util.data import *
from config.propara import NCETConfig


# ====================== main task =============================

class ProParaDataset(Dataset):
    ''' dataset for ProPara main task '''
    def __init__(self, path_data_dir, word2index, label2index, max_word_length, padding=False, sent_location=False, max_data_size=None):
        ''' '''
        datafile_name = "preprocess.pkl"
        self.datas = load_pkl(os.path.join(path_data_dir, datafile_name))
        self.max_data_size = max_data_size
        self.word2index  = word2index
        self.label2index = label2index
        self.pad_idx = 0
        self.unk_idx = 1
        self.max_word_length = max_word_length
        self.padding = padding

        if self.max_data_size:
            self.datas = self.datas[:max_data_size]


    def __getitem__(self, index):
        data = self.datas[index]
        doc_idx = data["doc_idx"]
        # sentence
        words, words_idxs, sent_lens = [], [], []
        for sent in data["sentence"]:
            words += sent
            words_idxs += [self.word2index.get(word, self.unk_idx) for word in sent]
            sent_lens.append(len(sent))
        words_length = len(words)
        sents_length = len(data["sentence"]) + 2 # add <SOS> & <EOS>
        # verb
        verbs = [0] * len(words)
        verbs_idxs_sents = [] 
        base_idx = 0
        for i, verb_idxs in enumerate(data["verb_idxs"]):
            verbs_idxs_sent = []
            for idx in verb_idxs:
                verbs[base_idx+idx] = 1
                verbs_idxs_sent.append(base_idx+idx)
            verbs_idxs_sents.append(verbs_idxs_sent)
            base_idx += sent_lens[i]

        # -------------------#
        #      entity        #
        # -------------------#

        # entity
        entity_to_idx, idx_to_entity = {}, {}
        for i, entity in enumerate(data["entities"]):
            entity_to_idx[entity] = i 
            idx_to_entity[i] = entity
        # entity linking
        entity_idxs_sents = {entity:[] for entity in data["entities"]}
        for entity, entity_idxs in data["entity_idxs"].items():
            base_idx = 0
            for i, idxs in enumerate(entity_idxs):
                entity_idxs_sent = []
                for idx in idxs[0] if idxs else []:
                    entity_idxs_sent.append(base_idx+idx)
                entity_idxs_sents[entity].append(entity_idxs_sent)
                base_idx += sent_lens[i]
        # stats gold label
        entity_states_gold = {entity:[] for entity in data["entities"]}
        for entity, gold in data["gold"].items():
            for action in gold["action"]:
                entity_states_gold[entity].append(self.label2index[action])
            entity_states_gold[entity] = [1] + entity_states_gold[entity] + [2]

        # -------------------------------#
        #      location candidates       #
        # -------------------------------#

        # location candidates
        location_candidate_to_idx, idx_to_location_candidate = {"-": 0, "?": 1}, {0: "-", 1: "?"}
        for i, location_candidate in enumerate(data["location_candidates"]):
            location_candidate_to_idx[location_candidate] = i+2
            idx_to_location_candidate[i+2] = location_candidate

        # location candidate linking
        location_candidate_idxs_sents = {location_candidate:[] for location_candidate in data["location_candidates"]}
        for location_candidate, location_candidate_idxs in data["location_candidate_idxs"].items():
            base_idx = 0
            for i, idxs in enumerate(location_candidate_idxs):
                location_candidate_idxs_sent = []
                for idx in idxs[0] if idxs else []:
                    location_candidate_idxs_sent.append(base_idx+idx)
                location_candidate_idxs_sents[location_candidate].append(location_candidate_idxs_sent)
                base_idx += sent_lens[i]
        location_candidate_idxs_sents["-"] = [[] for _ in range(len(sent_lens))]
        location_candidate_idxs_sents["?"] = [[] for _ in range(len(sent_lens))]

        # location gold label
        entity_locations_gold = {entity:[] for entity in data["entities"]}
        for entity, gold in data["gold"].items():
            for location in gold["location_after"]:
                entity_locations_gold[entity].append(location_candidate_to_idx.get(location, 1))
            entity_locations_gold[entity] = [location_candidate_to_idx.get(gold["location_before"][0], 0)] + entity_locations_gold[entity] + [0]

        # padding
        if self.padding:
            words_idxs = padding_sequence(words_idxs, self.max_word_length, self.pad_idx)
            verbs = padding_sequence(verbs, self.max_word_length, self.pad_idx)

        data = {
            "doc_idx": doc_idx,
            "words": words,
            "words_idxs": words_idxs,
            "verbs": verbs,
            "words_length": words_length,
            "sents_length": sents_length,
            "verbs_idxs_sents": verbs_idxs_sents,
            "entity_idxs_sents": entity_idxs_sents,
            "entity_states_gold": entity_states_gold,
            "location_candidate_idxs_sents": location_candidate_idxs_sents,
            "entity_locations_gold": entity_locations_gold,
            "entity_to_idx": entity_to_idx,
            "idx_to_entity": idx_to_entity,
            "location_candidate_to_idx": location_candidate_to_idx,
            "idx_to_location_candidate": idx_to_location_candidate
        }

        return data

    def __len__(self):
        return len(self.datas)



# ========= main ========

if __name__ == '__main__':

    config = NCETConfig()
    dataset = ProParaDataset(path_leaderboard_train_dir, config.word2index, config.label2index, config.max_word_length)
    dataset[0]