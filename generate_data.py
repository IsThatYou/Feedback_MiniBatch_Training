from corti_data_manager.synthetic.generate_data import process_conf, DatasetUtils
from corti_data_manager.synthetic.streaming_data_generator import StreamingDataGenerator
from corti_data_manager.synthetic.processors import DataProcessors
from corti_data_manager.synthetic.processors import DataProcessors
import tensorflow as tf

import numpy as np
from scipy.special import softmax

import os
import json
import scipy

conf = process_conf("corti-data-manager/tests/data/confs/stream_data_gen.yml")          
sqg = StreamingDataGenerator(conf)  
number_of_patterns = len(sqg.streaming_parser.patterns)
print("Number of patterns: ",number_of_patterns)

pattern_dist = np.zeros(number_of_patterns).tolist()
#used for training
norm_dist = scipy.stats.norm(number_of_patterns//2,5000)

# for testing
norm_dist = scipy.stats.norm(0,5000)
print(norm_dist.pdf(0))
print(norm_dist.pdf(number_of_patterns//2))
print(norm_dist.pdf(number_of_patterns))
for i in range(number_of_patterns):
    pattern_dist[i] = norm_dist.pdf(i)

# uniform distribution
# pattern_dist = softmax(np.zeros(number_of_patterns)).tolist()

# total = 833000
total = 200000
num_batch = 200
batch_size = total//num_batch
qarr = []
larr = []
iarr = []
counter = 0
config = process_conf("corti-data-manager/confs/generate_tv_synthetic_data.yml")

proc = DataProcessors(config)
for q,i in sqg.stream_processed_data(pattern_distribution=pattern_dist,  minibatch_size=batch_size):
    queries = np.array([" ".join(x["queries"][0]) for x in proc.process_queries(q)])
    labels = np.array([x["logical_forms"][0]["predicate"] for x in proc.process_queries(q)])
    print(queries)
    print(labels)
    print(i)
 
    qarr.extend(queries)
    larr.extend(labels)
    iarr.extend(i)
    counter += 1
    if (counter>200):
        break
print(len(qarr))
print(len(larr))
print(len(iarr))
dirp = "test_norm"
with open("output/"+ dirp+"/train.json","w") as json_file:
    json_file.write(json.dumps(qarr))
with open("output/"+ dirp+"/label.json","w") as json_file:
    json_file.write(json.dumps(larr))
with open("output/"+ dirp+"/index.json","w") as json_file:
    json_file.write(json.dumps(iarr))
# DatasetUtils.write_files(arr,config)