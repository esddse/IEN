import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import json
import re
from collections import defaultdict

import nltk
from nltk.parse import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.stem import WordNetLemmatizer
import numpy as np

from util.data import *
from util.path import *

# ================ function ==============

def get_noun_phrases(word_postag):
    ''' 
    parameters:
        word_postag: [(word, postag), (word, postag), ...]
    return:
        set(np1, np2, ...)
    '''
    nps = set()
    i = 0
    # print(word_postag)
    while i < len(word_postag):
        word, postag = word_postag[i]
        if postag.startswith("NN"):
            if i < len(word_postag)-1 and word_postag[i+1][1].startswith("NN"):
                i += 1
                continue
            np = [wnl.lemmatize(word)]
            nps.add(np[0])
            # search backward
            j = i-1
            while j >= 0 and word_postag[j][1].startswith("NN") or word_postag[j][1]=="JJ" or word_postag[j][1]=="CC":
                np = [wnl.lemmatize(word_postag[j][0])] + np
                j -= 1
            # search forward
            j = i+1 
            while j < len(word_postag) and word_postag[j][1].startswith("NN"):
                np.append(wnl.lemmatize(word_postag[j][0]))
                j += 1 
            # add to set
            nps.add(" ".join(np)) 
        i += 1
    return nps

def entity_matching(sentence, entity_dict):
    ''' 
        replacing entities with entity tags 
        参数:
            sentence: str 
            entity_dict: {entity:tag}
        返回:
            replaced_sentence: str
    '''

    sentence = " ".join([wnl.lemmatize(word) for word in nltk.word_tokenize(sentence.lower())])
    for entity, tag in sorted(entity_dict.items(), key=lambda item: len(item[0].split()), reverse=True):
        entity = " ".join([wnl.lemmatize(word) for word in nltk.word_tokenize(entity.lower())])
        sentence = sentence.replace(entity, tag)
    return sentence


def word_list_matching(word_lst, pattern):
    ''' match 2 list, return idxs '''
    idxs = []
    for i, word in enumerate(word_lst):
        if word == pattern[0]:
            start, end, j = i, i, 0 
            while end < len(word_lst) and j < len(pattern) and word_lst[end] == pattern[j]:
                end += 1
                j += 1
            if end-start == len(pattern):
                idxs.append((start, end-1))
    return idxs

# ================ process ===============

def preprocess(path_data_dir, is_train=False):
    ''' '''
    # load data
    vocab_size, word2index, index2word = load_vocab(path_fasttext_vocab)
    docs, tables = load_propara_data(path_data_dir)
    data_size = len(docs)

    # POS tagging, report the location recall 
    # About tag: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    global st, wnl
    st = CoreNLPParser(url='http://', tagtype='pos')
    wnl = WordNetLemmatizer()

    datas = []
    loc_num, noun_num, total_match_num = 0, 0, 0
    max_loc_num, max_noun_num = 0, 0
    cnt = 0
    for doc_idx, doc in docs.items():

        # entities
        entities = {gold["entity"] for gold in tables[doc_idx]["1"]}

        # postagging && location && verb_idxs && entity_idxs
        locations, location_candidates = set(), set()
        sents, verb_idxs, entity_idxs = [], [], {entity:[] for entity in entities}
        golds = {entity:{"action":[], "location_before":[], "location_after":[]} for entity in entities}
        for sent_idx, sentence in doc.items():
            # postag
            word_tag = st.tag(sentence.lower().split())
            # location
            for gold in tables[doc_idx][sent_idx]:
                entities.add(gold["entity"])
                locations.update([loc for loc in [gold["location_before"], gold["location_after"]] if loc != "?" and loc != "-"])
            nps = get_noun_phrases(word_tag)

            location_candidates.update(nps) 
            # sentence
            sent = [item[0] for item in word_tag]
            sents.append(sent)
            # verb_idxs
            verb_idx = tuple([idx for idx, item in enumerate(word_tag) if item[1].startswith("VB")])
            verb_idxs.append(verb_idx)
            # entity_idxs
            for entity_tuple in entities:
                for entity in sorted(entity_tuple, key=len, reverse=True):
                    pattern = entity.strip().split()
                    idxs = word_list_matching(sent, pattern)
                    if idxs: break
                entity_idxs[entity_tuple].append(idxs)
            # golds
            for gold in tables[doc_idx][sent_idx]:
                entity = gold["entity"]
                golds[entity]["action"].append(gold["action"])
                golds[entity]["location_before"].append(gold["location_before"])
                golds[entity]["location_after"].append(gold["location_after"])

        if is_train:
            location_candidates.update(locations)

        # location candidates idxs
        location_candidate_idxs = {candidate:[] for candidate in location_candidates}
        for sent in sents:
            sent = [wnl.lemmatize(word) for word in sent]
            for candidate in location_candidates:
                idxs = word_list_matching(sent, candidate.split())
                location_candidate_idxs[candidate].append(idxs)

        # locating statistics
        loc_num += len(locations)
        noun_num += len(location_candidates)
        max_loc_num = max(max_loc_num, len(locations))
        max_noun_num = max(max_noun_num, len(location_candidates))
        match_num = 0
        for location in locations:
            if location in location_candidates:
                match_num += 1

        total_match_num += match_num

        # assemble
        data = {
            "doc_idx": doc_idx,            # str
            "sentence": sents,             # [[word1, word2, ...], [], ...]
            "verb_idxs": verb_idxs,        # [(idx), (), ...]
            "entities": entities,          # set((alias1, alias2))
            "entity_idxs": entity_idxs,    # {entity:[(start, end), (start, end), ...]}
            "location_candidates": location_candidates,   # set(str)
            "location_candidate_idxs": location_candidate_idxs, # {location_candidate:[(start, end), (start, end), ...]}
            "gold":golds                   # {entity: {"action":[], "location_before": [], "location_after":[]}}
        }
        datas.append(data)

    print("avg_location_num: %f, avg_noun_num: %f" % (loc_num/data_size, noun_num/data_size))
    print("max_location_num: %f, max_noun_num: %f" % (max_loc_num, max_noun_num))
    print("recall: %d/%d=%f" % (total_match_num, loc_num, total_match_num/loc_num))

    # save
    path_save = os.path.join(path_data_dir, "preprocess.pkl")
    print("saving data to %s" % path_save)
    dump_pkl(datas, path_save)



# ================= main =================

def main():
    preprocess(path_propara_train_dir, is_train=True)
    preprocess(path_propara_dev_dir)
    preprocess(path_propara_test_dir)

if __name__ == '__main__':
    main()