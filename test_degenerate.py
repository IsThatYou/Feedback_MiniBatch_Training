import tensorflow as tf
import numpy as np
import os
import json
import corti_data_manager.synthetic.generate_data as gd     
from collections import defaultdict
from corti_common.data.entity import EntityDictionary
from scipy import special
from corti_data_manager.synthetic.streaming_data_genertor import StreamingDataGenerator
def main():
    path = "corti-data-manager/tests/data/" 
    entity_file_path = os.path.join(path,"may10-entity.json")
    intent_file_path = os.path.join(path,"may10-intent.json")
    # Data Generator
    conf = gd.process_conf("corti-data-manager/tests/data/confs/stream_data_gen.yml")          
    sqg = StreamingDataGenerator(conf)  
    number_of_patterns = sqg.total_patterns
    print("Number of patterns: ",number_of_patterns)
    patterns = sqg.streaming_parser.patterns # sqg.get_patterns() 
    mask_idxs_exclude = []
    mask_idxs_include = []
    if os.path.isfile("output/before/pattern_losses.txt"):
        f = open("output/before/pattern_losses.txt", "r")
        old = f.read()
        old = old[1:-1]
        old = np.array([float(x) for x in old.split(", ")])
        mask_idxs_exclude = np.where(old==10.0)[0]
        mask_idxs_include = np.where(old!=10.0)[0]
    
    
    folder = "after"
    f = open("output/"+folder+"/pattern_losses.txt", "r")
    txt = f.read()
    txt = txt[1:-1]
    txt = np.array([float(x) for x in txt.split(", ")])
    
    idxs1 =np.where(np.bitwise_and((txt==10.0),(old!=10.0)))
    idxs2 = 11525
    #print(len(idxs[0]))
    pattern_dist = np.zeros(number_of_patterns)
    pattern_dist[idxs2] = 2
    for q,i in sqg.stream_processed_data(pattern_dist):
        print(q,i)
if __name__ == "__main__":
    main()
