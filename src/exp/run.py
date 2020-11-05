import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import copy
import math
import random
from collections import Counter
import argparse

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, confusion_matrix

from util.path import *
from util.data import *

from exp.dataset import ProParaDataset
from config.propara import NCETConfig, IENConfig
from model.NCET import NCETModel
from model.IEN import IENModel
from model.layers.metrics import *
from postprocess.postprocess import *


def run(method):

    print("init config")
    if method == "NCET":
        config = NCETConfig()
    elif method == "IEN":
        config = IENConfig()

    # model dir
    model_name = config.model_name
    path_model = os.path.join(path_model_dir, "propara", model_name)
    path_model_result = os.path.join(path_data_dir, "result", model_name)
    if not os.path.exists(path_model):
        os.makedirs(path_model)
    if not os.path.exists(path_model_result):
        os.makedirs(path_model_result)
    print("model name:", model_name)

    # load data
    print("init dataset ...")
    dataset_train = ProParaDataset(path_propara_train_dir, config.word2index, config.label2index, config.max_word_length, padding=False, max_data_size=None)
    dataset_dev   = ProParaDataset(path_propara_dev_dir, config.word2index, config.label2index, config.max_word_length, padding=False, max_data_size=None)
    dataset_test  = ProParaDataset(path_propara_test_dir, config.word2index, config.label2index, config.max_word_length, padding=False)
    print("train data size:", len(dataset_train))
    print("dev   data size:", len(dataset_dev))
    print("test  data size:", len(dataset_test))

    # gpu
    print("set gpu and init model ...")
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

    # model
    print("init model ...")
    if not config.gpu_id:
        if method == "NCET":
            model = NCETModel(config)
        elif method == "IEN":
            model = IENModel(config)
    else:
        if method == "NCET":
            model = NCETModel(config).cuda()
        elif method == "IEN":
            model = IENModel(config).cuda()

    if config.test_only:
        last_path = config.model_path
        last_ckpt = last_path.split("/")[-1]
    else:
        # train
        print("start training ...")
        train_loader = DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=True, pin_memory=True)
        dev_loader   = DataLoader(dataset=dataset_dev, batch_size=config.test_batch_size, shuffle=False, pin_memory=True)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=config.patience, factor=config.decay_factor, verbose=True, min_lr=config.min_learning_rate)
        max_macro_f1, last_path, last_ckpt = -1, None, None
        for epoch in range(config.max_epoch_num):
            print("\n\n========== epoch %d ===========" % epoch)

            # warm_up
            if epoch < config.warm_up_epoch:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = (epoch + 1) / config.warm_up_epoch * config.learning_rate

            # # train step
            model.train()
            step, accu_loss_state, accu_loss_location = 0, 0, 0
            acc_num_state, acc_num_location, total_num_state, total_num_location = 0, 0, 0, 0
            for data in train_loader:
                model.zero_grad()
                words, words_idxs, verbs, words_length, sents_length, verbs_idxs_sents = data["words"], data["words_idxs"], data["verbs"], data["words_length"], data["sents_length"], data["verbs_idxs_sents"]
                entity_idxs_sents, entity_states_gold = data["entity_idxs_sents"], data["entity_states_gold"]
                entity_to_idx, idx_to_entity = data["entity_to_idx"], data["idx_to_entity"]
                location_candidate_idxs_sents, entity_locations_gold = data["location_candidate_idxs_sents"], data["entity_locations_gold"]
                location_candidate_to_idx, idx_to_location_candidate = data["location_candidate_to_idx"], data["idx_to_location_candidate"]

                sents_length = sents_length.view(-1)
                entity_states_logit, entity_locations_logit = model(words, words_idxs, verbs, words_length, sents_length, verbs_idxs_sents, entity_idxs_sents, entity_to_idx, idx_to_entity, location_candidate_idxs_sents, location_candidate_to_idx, idx_to_location_candidate)

                loss_state = 0
                for entity, states_gold in entity_states_gold.items():
                    states_logit = entity_states_logit[entity]
                    states_gold = torch.LongTensor([gold.item() for gold in states_gold]).unsqueeze(dim=0).to(states_logit.device)
                    mask = torch.ones(states_gold.size(), dtype=torch.bool).to(states_logit.device)
                    loss_state += model.crf.nll(states_logit, states_gold, sents_length, mask)
                    # metrics
                    acc_num_state += torch.sum(torch.argmax(states_logit, dim=-1) == states_gold).item()
                    total_num_state += states_gold.size(1)

                loss_location = 0
                for entity, locations_gold in entity_locations_gold.items():
                    locations_logit = entity_locations_logit[entity].transpose(1, 2)
                    locations_gold = torch.LongTensor([gold.item() for gold in locations_gold]).unsqueeze(dim=0).to(locations_logit.device)
                    loss_location += loss_func(locations_logit, locations_gold)
                    # metrics
                    acc_num_location += torch.sum(torch.argmax(locations_logit.transpose(1, 2), dim=-1) == locations_gold).item()
                    total_num_location += locations_gold.size(1)

                loss = loss_state + loss_location
                loss.backward()
                optimizer.step()

                accu_loss_state += loss_state.item()
                accu_loss_location += loss_location.item()
                step += 1

            print("train_loss_state: %f, train_loss_location: %f, train_acc_state: %f, train_acc_location: %f" % (accu_loss_state/step, accu_loss_location/step, acc_num_state/total_num_state, acc_num_location/total_num_location))
            print("***")

            if epoch < config.warm_up_epoch:
                continue

            # dev step
            model.eval()
            step, accu_loss_state, accu_loss_location = 0, 0, 0
            acc_num_state, acc_num_location, total_num_state, total_num_location = 0, 0, 0, 0
            total_labels, total_preds = [], []
            TP, FP, FN = Counter(), Counter(), Counter()
            for data in dev_loader:
                words, words_idxs, verbs, words_length, sents_length, verbs_idxs_sents = data["words"], data["words_idxs"], data["verbs"], data["words_length"], data["sents_length"], data["verbs_idxs_sents"]
                entity_idxs_sents, entity_states_gold = data["entity_idxs_sents"], data["entity_states_gold"]
                entity_to_idx, idx_to_entity = data["entity_to_idx"], data["idx_to_entity"]
                location_candidate_idxs_sents, entity_locations_gold = data["location_candidate_idxs_sents"], data["entity_locations_gold"]
                location_candidate_to_idx, idx_to_location_candidate = data["location_candidate_to_idx"], data["idx_to_location_candidate"]

                entity_states_logit, entity_locations_logit = model(words, words_idxs, verbs, words_length, sents_length, verbs_idxs_sents, entity_idxs_sents, entity_to_idx, idx_to_entity, location_candidate_idxs_sents, location_candidate_to_idx, idx_to_location_candidate)

                loss_state = 0
                for entity, states_gold in entity_states_gold.items():
                    states_logit = entity_states_logit[entity]
                    states_gold = torch.LongTensor([gold.item() for gold in states_gold]).unsqueeze(dim=0).to(states_logit.device)
                    mask = torch.ones(states_gold.size(), dtype=torch.bool).to(states_logit.device)
                    loss_state += model.crf.nll(states_logit, copy.deepcopy(states_gold), sents_length, mask)
                    states_predict = model.crf.decode(states_logit, sents_length, mask)
                    # metrics
                    get_TP_FP_FN(states_predict[0][1:-1], states_gold[0][1:-1], TP, FP, FN)
                    acc_num_state += torch.sum(states_predict[0][1:-1] == states_gold[0][1:-1]).item()
                    total_num_state += states_gold.size(1) - 2

                loss_location = 0
                for entity, locations_gold in entity_locations_gold.items():
                    locations_logit = entity_locations_logit[entity].transpose(1, 2)
                    locations_gold = torch.LongTensor([gold.item() for gold in locations_gold]).unsqueeze(dim=0).to(locations_logit.device)
                    loss_location += loss_func(locations_logit, locations_gold)

                    # print
                    states_logit = entity_states_logit[entity]
                    states_gold  = entity_states_gold[entity]
                    states_gold = torch.LongTensor([gold.item() for gold in states_gold]).unsqueeze(dim=0).to(states_logit.device)
                    mask = torch.ones(states_gold.size(), dtype=torch.bool).to(states_logit.device)
                    states_predict = model.crf.decode(states_logit, sents_length, mask)

                    states_predict = [config.index2label[state.item()] for state in states_predict[0][1:-1]]
                    states_gold    = [config.index2label[state.item()] for state in  states_gold[0][1:-1]]
                    location_predict = [idx_to_location_candidate[loc.item()][0] for loc in  torch.argmax(locations_logit.transpose(1,2), dim=-1)[0][1:-1]]
                    location_gold  = [idx_to_location_candidate[loc.item()][0] for loc in  locations_gold[0][1:-1]]

                    # metrics
                    for sp, sg, lp, lg in zip(states_predict, states_gold, location_predict, location_gold):
                        if sp != "NONE":
                            acc_num_location += 1 if lp == lg else 0
                            total_num_location += 1

                accu_loss_state += loss_state.item()
                accu_loss_location += loss_location.item()
                step += 1 

            P_R_F1 = get_P_R_F1(TP, FP, FN)
            for item, result in P_R_F1.items():
                print(config.index2label[item], result)
            macro_f1 = (P_R_F1[config.label2index["CREATE"]]["F"] + P_R_F1[config.label2index["DESTROY"]]["F"]) / 2
            # macro_f1 = (P_R_F1[config.label2index["CREATE"]]["F"] + P_R_F1[config.label2index["DESTROY"]]["F"] + P_R_F1[config.label2index["MOVE"]]["F"]) / 3
            print("dev_loss_state: %f, dev_loss_location: %f, dev_acc_state: %f, dev_acc_location: %f, macro_f1_state: %f" % (accu_loss_state/step, accu_loss_location/step, acc_num_state/total_num_state, acc_num_location/total_num_location, macro_f1))
            print()

            # save
            if macro_f1 > max_macro_f1:
                if last_path:
                    print("remove %s" % (last_path))
                    os.remove(last_path)
                ckpt_name = model_name + "_%.4f" % (macro_f1)
                save_path = os.path.join(path_model, ckpt_name)
                last_path, last_ckpt = save_path, ckpt_name
                print("macro_f1 from %.4f -> %.4f, saving model to %s" % (max_macro_f1, macro_f1, save_path))
                torch.save(model.state_dict(), save_path)
                max_macro_f1 = macro_f1
            # scheduler
            scheduler.step(macro_f1)


    # test
    print("start testing ...")
    # load best model
    print("loading model from %s" % (last_path))
    model.load_state_dict(torch.load(last_path))
    test_loader  = DataLoader(dataset=dataset_test, batch_size=config.test_batch_size, shuffle=False, pin_memory=True)
    model.eval()
    step, accu_loss = 0, 0
    acc_num, total_num = 0, 0
    total_labels, total_preds = [], []
    TP, FP, FN = Counter(), Counter(), Counter()
    results = {}
    for data in test_loader:
        words, words_idxs, verbs, words_length, sents_length, verbs_idxs_sents = data["words"], data["words_idxs"], data["verbs"], data["words_length"], data["sents_length"], data["verbs_idxs_sents"]
        entity_idxs_sents, entity_states_gold = data["entity_idxs_sents"], data["entity_states_gold"]
        entity_to_idx, idx_to_entity = data["entity_to_idx"], data["idx_to_entity"]
        location_candidate_idxs_sents, entity_locations_gold = data["location_candidate_idxs_sents"], data["entity_locations_gold"]
        location_candidate_to_idx, idx_to_location_candidate = data["location_candidate_to_idx"], data["idx_to_location_candidate"]
        doc_idx = data["doc_idx"][0]
        entity_states_logit, entity_locations_logit = model(words, words_idxs, verbs, words_length, sents_length, verbs_idxs_sents, entity_idxs_sents, entity_to_idx, idx_to_entity, location_candidate_idxs_sents, location_candidate_to_idx, idx_to_location_candidate)

        loss = 0
        results[doc_idx] = {}
        for entity, states_gold in entity_states_gold.items():
            states_logit = entity_states_logit[entity]
            locations_logit = entity_locations_logit[entity]
            states_gold = torch.LongTensor([gold.item() for gold in states_gold]).unsqueeze(dim=0).to(states_logit.device)
            mask = torch.ones(states_gold.size(), dtype=torch.bool).to(states_logit.device)
            loss += model.crf.nll(states_logit, copy.deepcopy(states_gold), sents_length, mask)
            states_predict = model.crf.decode(states_logit, sents_length, mask)
            # metrics
            get_TP_FP_FN(states_predict[0][1:-1], states_gold[0][1:-1], TP, FP, FN)
            acc_num += torch.sum(states_predict[0][1:-1] == states_gold[0][1:-1]).item()
            total_num += states_gold.size(1) - 2
            # get result
            results[doc_idx][entity] = {}
            results[doc_idx][entity]["states"] = [config.index2label[state.item()] for state in states_predict[0][1:-1]]
            results[doc_idx][entity]["locations"] = [idx_to_location_candidate[loc.item()][0] for loc in torch.argmax(locations_logit, dim=-1)[0][:-1]]


        accu_loss += loss.item()
        step += 1 

    P_R_F1 = get_P_R_F1(TP, FP, FN)
    for item, result in P_R_F1.items():
        print(config.index2label[item], result)
    macro_f1 = (P_R_F1[config.label2index["CREATE"]]["F"] + P_R_F1[config.label2index["DESTROY"]]["F"] + P_R_F1[config.label2index["MOVE"]]["F"]) / 3
    print("test_loss: %f, test_acc: %f, macro_f1: %f" % (accu_loss/step, acc_num/total_num, macro_f1))
    print()

    # save
    path_save_document = os.path.join(path_model_result, last_ckpt+".tsv")
    result_save_document_format(results, path_save_document)
    # evaluation
    path_gold_answer = os.path.join(path_propara_test_dir, "answers.tsv")
    path_evaluator = os.path.join(path_propara_evaluator_dir, "evaluator.py")
    cmd = "%s -p %s -a %s" % (path_evaluator, path_save_document, path_gold_answer)
    eval_result_document = os.popen(cmd).read()
    print(eval_result_document)
    

# ====================== main =========================

def main():
    # cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", default="IEN")
    args = parser.parse_args()
    # run 
    if args.method in ["IEN", "NCET"]:
        run(args.method)

if __name__ == '__main__':
    main()