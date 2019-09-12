import json
import os
# import tensorflow as tf
# from tensorflow.keras.layers import Dense
# from tensorflow.data import Dataset
# from tensorflow.losses import sigmoid_cross_entropy, Reduction
from corti_data_manager.synthetic.generate_data import process_conf
from corti_data_manager.synthetic.streaming_data_generator import StreamingDataGenerator
import numpy as np


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from corti_common.data.entity import EntityDictionary

def main():
    path = "corti-data-manager/tests/data/" 
    entity_file_path = os.path.join(path,"may10-entity.json")
    intent_file_path = os.path.join(path,"may10-intent.json")

    num_labels = 0
    data = []
    conflicted = 0
    pattern_dict = defaultdict(set)
    for line in open(intent_file_path):
        d = json.loads(line)
        data.append(d)
        pattern = d["_update"]["pattern"]
        action = d["_update"]["action"]
        if pattern in pattern_dict:
            if action not in pattern_dict[pattern]:
                print(pattern)
                print(action)
                print(pattern_dict[pattern])
                conflicted+=1
        pattern_dict[pattern].add(action)
    print(conflicted)
    # Data Generator
    # conf = process_conf("corti-data-manager/tests/data/confs/stream_data_gen.yml")          
    # print(conf)
    # sqg = StreamingDataGenerator(conf)  
    # number_of_patterns = len(sqg.streaming_parser.patterns)
    # print("Number of patterns: ",number_of_patterns)

    # patterns_set = set()
    # patterns_dic = defaultdict(int)
    # # patterns_dic = defaultdict(set)
    # count = 0
    # for p in sqg.streaming_parser.patterns:
    #     count += 1
    #     phrase = p.phrase
    #     # print(phrase)
    #     # prev = len(patterns_set)
    #     patterns_set.add(phrase)
    #     patterns_dic[phrase]+=1
    #     # aft = len(patterns_set)
    #     # print(aft)
    #     # patterns_dic[phrase].add
    # print(len(patterns_set))
    # print("there are %d duplicates" %(number_of_patterns-len(patterns_set)))
    # print(count)
    # for k in patterns_dic:
    #     if patterns_dic[k] >1:
    #         print(k,patterns_dic[k])



if __name__ == "__main__":
    main()
