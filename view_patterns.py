import tensorflow as tf
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.data import Dataset
from tensorflow.losses import sigmoid_cross_entropy, Reduction
import corti_data_manager.synthetic.generate_data as gd     
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict

from corti_common.data.entity import EntityDictionary
from scipy import special
from corti_data_manager.synthetic.streaming_data_genertor import StreamingDataGenerator


tf.enable_eager_execution()
def main():
    path = "corti-data-manager/tests/data/" 
    entity_file_path = os.path.join(path,"may10-entity.json")
    intent_file_path = os.path.join(path,"may10-intent.json")
    # Data Generator
    conf = gd.process_conf("corti-data-manager/tests/data/confs/stream_data_gen.yml")          
    sqg = StreamingDataGenerator(conf)  
    number_of_patterns = sqg.total_patterns
    print("Number of patterns: ",number_of_patterns)
    pattern_dist = tf.math.softmax(np.zeros(number_of_patterns)).numpy().tolist()
    patterns = sqg.streaming_parser.patterns # sqg.get_patterns() 


    mask_idxs_exclude = []
    mask_idxs_include = []
    '''
    if os.path.isfile("output/before/pattern_losses.txt"):
        f = open("output/before/pattern_losses.txt", "r")
        old = f.read()
        old = old[1:-1]
        old = np.array([float(x) for x in old.split(", ")])
        mask_idxs_exclude = np.where(old==10.0)[0]
        mask_idxs_include = np.where(old!=10.0)[0]
    '''

    folder = "after3"
    f = open("output/"+folder+"/pattern_losses.txt", "r")
    txt = f.read()
    txt = txt[1:-1]
    txt = np.array([float(x) for x in txt.split(", ")])
    idxs = np.where(txt==10.0)
    with open("output/"+folder+"/patterns_not_generated.txt","w") as f:
        for i in idxs[0]:
            f.write(patterns[i].phrase)
            f.write("\n")
    bin_num = 100
    hist,bins = np.histogram(txt,bins=bin_num)
    vals = np.digitize(txt,bins)
    with open("output/"+folder+"/hist.txt","w") as f: 
        print(hist)
        f.write(json.dumps(list(map(int,hist))))
    
    counts = np.load("counts3.npy",allow_pickle=True)
    counts = counts.item()
    with open("output/"+folder+"/bad_patterns.txt","w") as f:
        for i in range(8,bin_num):
            f.write(str(i))
            f.write("\n")
            idxs = np.where(vals==(i+1))
            for each in idxs[0]:
                print(patterns[each].phrase)
                f.write(patterns[each].phrase)
                f.write(", id:"+str(each))
                f.write(", counts:"+str(counts[each]))
                f.write(", loss:"+str(txt[each]))
                label = get_label(sqg,number_of_patterns,each)
                f.write(", label:"+label)

                f.write("\n")
    
def get_label(sqg,numpat,pattern_id):
    pattern_dist= np.zeros(numpat)
    pattern_dist[pattern_id] = 1.0
    for q,i in sqg.stream_processed_data(pattern_distribution=pattern_dist,minibatch_size=10):
        labels = np.array([x["logical_forms"][0]["predicate"] for x in q])
        break
    print(labels)
    return labels[0]
if __name__ == '__main__':
	main()
