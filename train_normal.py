import json
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.data import Dataset
from tensorflow.losses import sigmoid_cross_entropy, Reduction
from corti_data_manager.synthetic.generate_data import process_conf
from corti_data_manager.synthetic.streaming_data_generator import StreamingDataGenerator
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from corti_common.data.entity import EntityDictionary

tf.enable_eager_execution()

class LSTMClassifier(tf.keras.Model):
    def __init__(self,vocab_size,num_labels,embedding_dim,lstm_units):
        super(LSTMClassifier, self).__init__()

        self.lstm_units = lstm_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(lstm_units,dropout=0.2, recurrent_dropout=0.2)
        self.pred = tf.keras.layers.Dense(num_labels,activation=None)

    def call(self, x):
        x = self.embedding(x)
        o = self.lstm(x)     
        o = self.pred(o)
        return o
def data_generator(tokenizer,maxlen,label_encoder,onehot_encoder, queries,labels,indexes):

    
    def sub(batch_size=10):
        print("batch size: ",batch_size)
        l = len(queries)
        for z in range(l - batch_size + 1):
            batch_q = queries[z:z+batch_size]
            batch_l = labels[z:z+batch_size]
            i = indexes[z:z+batch_size]
            tokenized_queries = tokenizer.texts_to_sequences(batch_q)
            # print(tokenized_queries)
            queries2 = sequence.pad_sequences(tokenized_queries, maxlen=maxlen,truncating='post')
            # print(queries2)
            # print(queries2.shape)
            int_labels = label_encoder.transform(batch_l)
            int_labels = int_labels.reshape(len(int_labels), 1)
            onehot_labels = onehot_encoder.transform(int_labels)
            yield queries2,onehot_labels,i
    return sub

def get_tokenizer(sqg):
    vocab = []
    entities = sqg.streaming_parser.entities
    for p in sqg.streaming_parser.patterns:
        vocab.extend(p.phrase.split())
    # print(entities.entity_types["channel"]._weights_by_language["all"]["entities"])
    # print(entities.entity_types["trailer"]._weights_by_language["all"]["entities"])
    # print(entities.entity_types)
    for t in entities.entity_types:
        if (type(entities.entity_types[t]) is EntityDictionary.EntityGroup):
            entity_list = entities.entity_types[t]._weights_by_language["all"]["entities"]
        else:
            # print(entities.entity_types[t].entities_by_action)
            for (qu,a) in entities.entity_types[t].entities_by_action.items():
                # print(a)
                entity_list = a._weights_by_language["all"]["entities"]
        for e in entity_list:
            vocab.extend(e.phrase.split())
    # print(vocab)
    tokenizer = Tokenizer(lower=True, filters=',',oov_token="<unk>")
    vocab.append("series")
    vocab.append("other")
    vocab.append("person")
    vocab.append("movie")
    print("vocab size",len(vocab))
    #print(vocab)
    tokenizer.fit_on_texts(vocab)
    #print(tokenizer.word_index)
    return tokenizer

def get_labels(path): 
    allla = []
    json_data = []
    with open(path) as json_file:
        for line in json_file:
            json_data.append(json.loads(line))
    for d in json_data:
        action = d["_update"]["action"]
        if action not in allla:
            allla.append(action)
    allla.append("SERIES")
    allla.append("MOVIE")
    allla.append("PERSON")
    allla.append("OTHER")
    
    return allla
def evaluate_accuracy(labels,preds):
    total = len(labels)
    counter = 0
    for l,p in zip(labels,preds):
        dummy_l = np.argmax(l)
        dummy_p = np.argmax(p)
        if (dummy_l==dummy_p):
            counter+= 1
    return counter/total
def main():
    path = "corti-data-manager/tests/data/" 
    entity_file_path = os.path.join(path,"may10-entity.json")
    intent_file_path = os.path.join(path,"may10-intent.json")
    nb_epocs = 12
    
    minibatch_size = 256
    maximum_nb_iter = 700
    loss_retained = 10
    maxlen = 30
    pattern_dist_history = {}
    loss_dist_history = {}
    loss_history = []
    list_labels = [] # logical form predicates/actions

    embedding_dims = 300
    lstm_units = 128

    num_features = 512
    num_labels = 0
    # Data Generator
    conf = process_conf("corti-data-manager/tests/data/confs/stream_data_gen.yml")          
    print(conf)
    sqg = StreamingDataGenerator(conf)  
    number_of_patterns = len(sqg.streaming_parser.patterns)
    print("Number of patterns: ",number_of_patterns)

    tokenizer = get_tokenizer(sqg)
    print("Vocab Size: ",len(tokenizer.word_counts))
    vocab_size = len(tokenizer.word_counts)

    list_labels = get_labels(intent_file_path)
    label_encoder = LabelEncoder()
    label_encoder.fit(list_labels)
    integer_encoded = label_encoder.transform(list_labels)
    num_labels = int(np.max(integer_encoded) +1)
    print("Number of labels: ",num_labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder.fit(integer_encoded)
    # print(onehot_encoder.transform(integer_encoded))

    pdir = "train_medium"
    path1 = "output/"+pdir+"/train.json"
    path2 = "output/"+pdir+"/label.json"
    path3 = "output/"+pdir+"/index.json"
    with open(path1) as json_file:
        queries = json.load(json_file)
    with open(path2) as json_file:
        labels = json.load(json_file)
    with open(path3) as json_file:
        indexes = json.load(json_file)
    print(queries[0])
    print(len(queries))
    datagen = data_generator(tokenizer,maxlen, label_encoder,onehot_encoder,queries,labels,indexes)
    checkpoint_prefix = "saved/orgM_medium/"
    # Model
    model = LSTMClassifier(vocab_size+2, num_labels,embedding_dims, lstm_units)
    optimizer = tf.train.AdamOptimizer()

    # avg_loss = np.ones_like(pattern_dist) * 10
    BUFFER_SIZE = len(queries)
    for j in range(nb_epocs):
        losses = defaultdict(lambda:[0]*loss_retained)
        counts = defaultdict(int)
        queries_shape = [None,maxlen]
        labels_shape = [None,num_labels]
        indexes_shape = [None,]
        dataset = Dataset.from_generator(datagen, args=[minibatch_size], output_types=(tf.int32,tf.int32,tf.int32),output_shapes=(tf.TensorShape(queries_shape), tf.TensorShape(labels_shape),tf.TensorShape(indexes_shape))).shuffle(BUFFER_SIZE)
        for i,(queries,labels,indexes) in enumerate(dataset):
            if (i > maximum_nb_iter):
                break
            print(j,i)
            with tf.GradientTape() as tape:
                predictions = model(queries)
                # [batch_size, num_classes]
                accu = evaluate_accuracy(labels,predictions)
                np.set_printoptions(threshold=np.inf)
                print("accuracy: ",accu)
                # print(predictions.shape)
                loss = tf.losses.softmax_cross_entropy(labels, predictions, reduction=Reduction.NONE)
                # print(loss)
                # print(loss.shape)
                # print(loss)
                loss_reduced = tf.math.reduce_mean(loss)
            print('iter:{0} --> Average Loss:{1}'.format(i,loss_reduced.numpy()))
                
            # Backpropagation
            grad = tape.gradient(loss_reduced, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables), global_step=tf.train.get_or_create_global_step())
        root = tf.train.Checkpoint(optimizer=optimizer,
                                    model=model,
                                    optimizer_step=tf.train.get_or_create_global_step())
        root.save(checkpoint_prefix)

    
    root = tf.train.Checkpoint(optimizer=optimizer,
                                        model=model,
                                        optimizer_step=tf.train.get_or_create_global_step())
    root.save(checkpoint_prefix)
if __name__ == "__main__":
    main()
