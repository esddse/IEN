import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import io
import json
import pickle
from collections import defaultdict, Counter
from tqdm import tqdm

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from util.path import *


# ===================== load & save =========================

def load_json(file_path):
    ''' load json file '''
    with open(file_path, "r", encoding="utf8") as f:
        data = json.load(f)
        return data

def dump_json(data, file_path):
    ''' save json file '''
    with open(file_path, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_pkl(path):
    ''' load pkl '''
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pkl(data, path):
    ''' save pkl '''
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=2)


def load_str_lst(path):
    ''' load string list '''
    strs = []
    with open(path, "r", encoding="utf8") as f:
        for line in tqdm(f):
            strs.append(line.strip())
    return strs

def dump_str_lst(lst, path):
    ''' save string list '''
    with open(path, "w", encoding="utf8") as f:
        for string in tqdm(lst):
            f.write(string+'\n')

def load_str_dict(path, seperator="\t", reverse=False, add_pad=False):
    ''' load string dict '''
    if add_pad:
        dictionary, reverse_dictionay = {0: "<pad>"}, {"<pad>": 0}
    else:
        dictionary, reverse_dictionay = {}, {}
    with open(path, "r", encoding="utf8") as f:
        for line in tqdm(f):
            try:
                key, value = line.strip().split(seperator)
                key = int(key)+1 if add_pad else int(key)
                dictionary[key] = value
                reverse_dictionay[value] = key
            except:
                pass
    if reverse:
        return dictionary, reverse_dictionay, len(dictionary)
    return dictionary, len(dictionary)


def load_propara_data(path_data_dir):
    ''' load ProPara data '''
    print("loading propara data from %s ..." % path_data_dir)
    path_sentences = os.path.join(path_data_dir, "sentences.tsv")
    path_answers   = os.path.join(path_data_dir, "answers.tsv")
    # sentences
    docs = {}
    for line in load_str_lst(path_sentences):
        doc_idx, sent_idx, sentence = line.split("\t")
        if doc_idx not in docs:
            docs[doc_idx] = {}
        docs[doc_idx][sent_idx] = sentence
    # answers
    tables = {}
    for line in load_str_lst(path_answers):
        doc_idx, sent_idx, participants, action, location_before, location_after = line.split("\t")
        if doc_idx not in tables:
            tables[doc_idx] = {}
        if sent_idx not in tables[doc_idx]:
            tables[doc_idx][sent_idx] = []
        gold = {}
        gold["entity"] = tuple(s for s in participants.split(";"))
        gold["action"] = action 
        gold["location_before"] = location_before
        gold["location_after"]  = location_after
        tables[doc_idx][sent_idx].append(gold)
    return docs, tables

def load_vocab(file_path):
    ''' load vocab '''
    word2index, index2word, idx = {}, {}, 0
    print(file_path)
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            word = line.strip()
            word2index[word] = idx
            index2word[idx] = word 
            idx += 1 
    vocab_size = len(word2index)
    return vocab_size, word2index, index2word

def load_word_vector(file_path, max_vocab_size=None, additional_words=None):
    '''
        读取词向量
        param:
            file_path：  路径
            max_vocab_size: 读取的词的数量，会读取词频最高的max_vocab_size个
            additional_words: 新增的词，随机初始化
        return:
            vocab_size:     读取的词数量
            embedding_dim:  embedding维度
            word2index:     词到index的转换
            index2word:     index到词的转换
            embeddings:     embedding矩阵
    '''
    
    print("loading word vectors...")
    word2index, index2word = {}, {}
    embeddings = []

    # load
    with open(file_path, "r", encoding="utf8") as f:
        # first row
        line = f.readline()
        if " " in line:
            line = line.strip().split(" ")
        elif "\t" in line:
            line = line.strip().split("\t")
        vocab_size    = int(line[0]) if max_vocab_size is None else max_vocab_size
        embedding_dim = int(line[1])

        # rest lines
        index = 0
        log_thr = 0.1
        for line in f:
            line = line.strip().split(" ")
            word = line[0]
            vector = list(map(float, line[1:]))

            if len(vector) != embedding_dim:
                continue

            word2index[word]    = index
            index2word[index]   = word
            embeddings.append(vector)

            percentage = float(index) / vocab_size
            if percentage > log_thr:
                print("%f%% loaded" % (percentage * 100))
                log_thr += 0.1
            index += 1
            
            if max_vocab_size is not None and index == max_vocab_size:
                break

    vocab_size = len(word2index)
    embedding_dim = len(embeddings[0])

    # check <pad>
    if "<pad>" not in word2index:
        print("add <pad> into wordvec ...")
        # vocab_size
        vocab_size += 1
        # word2index
        new_word2index = {"<pad>": 0}
        for word, index in word2index.items():
            new_word2index[word] = index+1
        word2index = new_word2index
        # index2word
        index2word = {}
        for word, index in word2index.items():
            index2word[index] = word
        # embeddings
        embeddings = [[0.] * embedding_dim] + embeddings

    # check <unk>
    if "<unk>" not in word2index:
        print("add <unk> into wordvec ...")
        # vocab_size
        vocab_size += 1
        # word2index
        new_word2index = {"<unk>": 1}
        for word, index in word2index.items():
            if index < 1:
                new_word2index[word] = index
            else:
                new_word2index[word] = index+1
        word2index = new_word2index
        # index2word
        for word, index in word2index.items():
            index2word[index] = word
        # embeddings
        def normalize(vec):
            '''
                归一化
            '''
            vec = np.array(vec)
            norm = np.linalg.norm(vec)
            if norm == 0:
                return vec
            return vec / norm
        UNK_embedding = np.array(embeddings[1:])
        UNK_embedding = np.mean(UNK_embedding, axis=0)
        UNK_embedding = normalize(UNK_embedding)
        UNK_embedding = list(UNK_embedding)
        embeddings = [embeddings[0]] + [UNK_embedding] + embeddings[1:]

    # additional words
    if additional_words:
        print("add additional_words into wordvec ...")
        for word in additional_words:
            embedding = list(np.random.rand(embedding_dim))
            embeddings.append(embedding)
            word2index[word] = vocab_size
            index2word[vocab_size] = word 
            vocab_size += 1

    print("complete!!!")
    return vocab_size, embedding_dim, word2index, index2word, embeddings


# ============== data transformation ==================

def padding_sequence(indices, max_length, pad_idx):
    '''
        param:
            indices: [1,5,23]
            max_length: int 
            pad_idx: int
        return:
            padded_indices: []
    '''
    if len(indices) >= max_length:
        return indices[:max_length]
    else:
        return indices + [pad_idx] * (max_length - len(indices))

def batch_padding_lm(batch, max_length=float("inf")):
    batch_pad = []
    max_length = min(max([length for seq, forward, backward, length in batch]), max_length)
    for seq, forward, backward, length in batch:
        if max_length > len(seq):
            seq      += [0] * (max_length-len(seq))
            forward  += [0] * (max_length-len(forward))
            backward += [0] * (max_length-len(backward))
        else:
            seq = seq[:max_length]
            forward = forward[:max_length]
            backward = backward[:max_length]
            length = max_length
        batch_pad.append((seq, forward, backward, length))
    return batch_pad, max_length


def gen_mask(batch_size, lengths, max_length):
    ''' generate mask matrix '''
    mat = []
    for length in lengths:
        one  = torch.ones((1, length), dtype=torch.uint8)
        if length < max_length:
            zero = torch.zeros((1, max_length-length), dtype=torch.uint8)
            vec = torch.cat([one, zero], dim=1)
        else:
            vec = one
        mat.append(vec)
    mat = torch.cat(mat, dim=0)
    return mat

# ===================== main =========================

def main():
    pass

if __name__ == '__main__':
    main()