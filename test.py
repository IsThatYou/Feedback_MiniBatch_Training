from corti_data_manager.synthetic.generate_data import process_conf, DatasetUtils
from corti_data_manager.synthetic.streaming_data_generator import StreamingDataGenerator
import numpy as np
import os
import json
import scipy
from corti_common.data.entity import EntityDictionary
q=[]
l = []
i = []
with open("output/train_small/train.json") as json_file:
    q = json.load(json_file)
with open("output/train_small/label.json") as json_file:
    l = json.load(json_file)
with open("output/train_small/index.json") as json_file:
    i = json.load(json_file)
print(q[:10])

conf = process_conf("corti-data-manager/tests/data/confs/stream_data_gen.yml")          
sqg = StreamingDataGenerator(conf)  
number_of_patterns = len(sqg.streaming_parser.patterns)
print("Number of patterns: ",number_of_patterns)
# for p in sqg.streaming_parser.patterns:
#     print(p.phrase)
entities = sqg.streaming_parser.entities
count = 0
for t in entities.entity_types:
    if (type(entities.entity_types[t]) is EntityDictionary.EntityGroup):
        entity_list = entities.entity_types[t]._weights_by_language["all"]["entities"]
    else:
        for (qu,a) in entities.entity_types[t].entities_by_action.items():
            entity_list = a._weights_by_language["all"]["entities"]
    count += len(entity_list)
print("number of entities: %d"%(count))
            