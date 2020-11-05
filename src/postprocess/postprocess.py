import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import copy
import json
import re
import nltk
from nltk.parse import CoreNLPParser
from nltk.stem import WordNetLemmatizer
import numpy as np

from util.data import *
from util.path import *


def result_save_document_format(results, path_save=None):
    '''
        ProPara standard format (for each line):
        doc_idx\tsent_idx\tentity\tstate\tlocation_before\tlocation_after

        Para:
            results: {
                "doc_idx": {
                    entity:{
                        "states": [state1, state2, ...]
                        "locations": [location1, location2, ...]
                    }
                }    
            }

            path_save: str
    '''
    
    lines = []
    for doc_idx, result in results.items():
        for entity, items in result.items():
            entity = ";".join(entity)
            states = items["states"]
            locations_before_pred = items["locations"][:-1]
            locations_after_pred  = items["locations"][1:]

            # locations_before = ["?"] * len(states)
            # locations_after  = ["?"] * len(states)
            locations_before = copy.deepcopy(locations_before_pred)
            locations_after  = copy.deepcopy(locations_after_pred)
            for i in range(len(states)):
                if states[i] == "CREATE":
                    locations_before[i] = "-"
                    locations_after[i]  = locations_after_pred[i] if locations_after_pred[i] != "-" else "?"
                    for j in range(i-1, -1, -1):
                        if states[j] == "NONE":
                            locations_before[j] = "-"
                            locations_after[j]  = "-"
                    for j in range(i+1, len(states)):
                        if states[j] != "NONE":
                            break 
                        locations_before[j] = locations_after[i]
                        locations_after[j]  = locations_after[i]
                elif states[i] == "DESTROY":
                    locations_after[i] = "-"
                    locations_before[i] = locations_before_pred[i] if locations_before_pred[i] != "-" else "?"
                    for j in range(i-1, -1, -1):
                        if states[j] != "NONE":
                            break
                        locations_before[j] = locations_before[i]
                        locations_after[j]  = locations_before[i]
                    for j in range(i+1, len(states)):
                        if states[j] == "NONE":
                            locations_before[j] = "-"
                            locations_after[j]  = "-"
                elif states[i] == "MOVE":
                    locations_before[i] = locations_before_pred[i] if locations_before_pred[i] != "-" else "?"
                    locations_after[i]  = locations_after_pred[i] if locations_after_pred[i] != "-" else "?"
                    for j in range(i-1, -1, -1):
                        if states[j] != "None":
                            break 
                        locations_before[j] = locations_before[i]
                        locations_after[j]  = locations_before[i]
                    for j in range(i+1, len(states)):
                        if states[j] != "NONE":
                            break 
                        locations_before[j] = locations_after[i]
                        locations_after[j]  = locations_after[i]
                elif states[i] == "NONE":
                    if locations_after[i] != locations_before[i]:
                        locations_after[i] = locations_before[i]
                else: # other invalid actions
                    states[i] = "NONE"

            for i in range(len(states)):
                sent_idx = str(i+1)
                line = "\t".join([doc_idx, sent_idx, entity, states[i], locations_before[i], locations_after[i]])
                lines.append(line)

    # save
    if path_save:
        dump_str_lst(lines, path_save)

# ========== main ===========

def main():
    pass

if __name__ == '__main__':
    main()
