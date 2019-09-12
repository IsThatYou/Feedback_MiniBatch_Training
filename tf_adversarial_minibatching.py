import tensorflow as tf
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.data import Dataset
from tensorflow.losses import sigmoid_cross_entropy, Reduction
from corti_data_manager.synthetic.generate_data import process_conf
from corti_data_manager.synthetic.streaming_data_generator import StreamingDataGenerator

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

def data_generator(sqg,tokenizer,maxlen,label_encoder,onehot_encoder):
    def sub(pattern_distribution=None,batch_size=10):
        print("batch size: ",batch_size)
        for q,i in sqg.stream_processed_data(pattern_distribution=pattern_distribution,  minibatch_size=batch_size):
            queries = np.array([" ".join(x["queries"][0]) for x in q])
            labels = np.array([x["logical_forms"][0]["predicate"] for x in q])
            # print(queries)
            # print(labels)
            # print(i)
            tokenized_queries = tokenizer.texts_to_sequences(queries)
            # print(tokenized_queries)
            queries2 = sequence.pad_sequences(tokenized_queries, maxlen=maxlen,truncating='post')
            # print(queries2)
            # print(queries2.shape)
            int_labels = label_encoder.transform(labels)
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
    nb_epocs = 700
    maximum_nb_iter = 48
    minibatch_size = 2048
    loss_retained = 10
    maxlen = 30
    pattern_dist_history = {}
    loss_dist_history = {}
    list_labels = [] # logical form predicates/actions

    embedding_dims = 300
    lstm_units = 128

    num_features = 512
    num_labels = 0

    list_loss1 = []
    list_loss2 = []
    list_loss3 = []

    # Data Generator
    conf = process_conf("corti-data-manager/tests/data/confs/stream_data_gen.yml")          
    print(conf)
    sqg = StreamingDataGenerator(conf)  
    number_of_patterns = len(sqg.streaming_parser.patterns)
    #number_of_patterns = sqg.total_patterns
    print("Number of patterns: ",number_of_patterns)
    print(sqg.total_patterns)
    pattern_dist = tf.math.softmax(np.zeros(number_of_patterns)).numpy()
    # mask for patterns that cannot be generated by data generator
    mask_idxs_exclude = []
    mask_idxs_include = []
    mask = np.ones_like(pattern_dist)
    if os.path.isfile("output/after2_3329/pattern_losses.txt"):
        f = open("output/after2_3329/pattern_losses.txt", "r")
        txt = f.read()
        txt = txt[1:-1]
        txt = np.array([float(x) for x in txt.split(", ")])
        mask_idxs_exclude = np.where(txt==10.0)[0]
        mask_idxs_include = np.where(txt!=10.0)[0]
        mask_idxs_include = mask_idxs_include[:-5]
        temp=tf.math.softmax(np.zeros(number_of_patterns)[mask_idxs_include]).numpy()
        pattern_dist[mask_idxs_include] = temp
    pattern_dist[mask_idxs_exclude] = 0

    print(pattern_dist[mask_idxs_exclude])
    pattern_dist = pattern_dist.tolist()
    print("pattern_dist = ",pattern_dist)

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
    datagen = data_generator(sqg,tokenizer,maxlen, label_encoder,onehot_encoder)

    # Model
    model = LSTMClassifier(vocab_size+2, num_labels,embedding_dims, lstm_units)
    optimizer = tf.train.AdamOptimizer()

    avg_loss = np.ones_like(pattern_dist) * 10
    checkpoint_prefix = "saved/after3/"

    counts = defaultdict(int)
    # Generate new minibatches 
    for j in range(nb_epocs):
        losses = defaultdict(lambda:[0]*loss_retained)
        queries_shape = [None,maxlen]
        labels_shape = [None,num_labels]
        indexes_shape = [None,]
        dataset = Dataset.from_generator(datagen, args=[pattern_dist, minibatch_size], output_types=(tf.int32,tf.int32,tf.int32),output_shapes=(tf.TensorShape(queries_shape), tf.TensorShape(labels_shape),tf.TensorShape(indexes_shape)))
        accuracies = 0
        if (j%5==0):
            list_loss3.append(list(avg_loss))
        # Iteration on one minibatch
        try:
            for i,(queries,labels,indexes) in enumerate(dataset):
                # print(labels,indexes)
                if (i > maximum_nb_iter):
                    break
                print("<------------------------- epoch %d iteration %d --------------------------->"%(j,i))

                # Forward pass
                print(queries.shape)
                with tf.GradientTape() as tape:
                    predictions = model(queries)
                    # [batch_size, num_classes]
                    #accu = evaluate_accuracy(labels,predictions)
                    np.set_printoptions(threshold=np.inf)
                    #print("accuracy: ",accu)
                    print(predictions.shape)
                    loss = tf.losses.softmax_cross_entropy(labels, predictions, reduction=Reduction.NONE)
                    #print(loss)
                    print(loss.shape)
                    # print(loss)
                    loss_reduced = tf.math.reduce_mean(loss)
                print('iter:{0} --> Average Loss:{1}'.format(i,loss_reduced.numpy()))
                
                # Backpropagation
                grad = tape.gradient(loss_reduced, model.trainable_variables)
                #print("out there")
                # Accumulate and group losses per pattern and compute average
                nplabels = labels.numpy()
                for i,(a_type, a_loss) in enumerate(zip(indexes,loss)):
                    a_type = int(a_type.numpy())
                    idx = int(counts[a_type]%loss_retained)
                    # print(a_loss)
                    # print(a_type,a_loss.shape)
                    max_idx = np.argmax(nplabels[i])
                    # print(max_idx)
                    # print(a_loss.numpy())
                    losses[a_type][idx]= float(np.mean(a_loss.numpy()))
                    counts[a_type] = counts.get(a_type,0.0)+1.0
                # Compute the average
                # print(losses)
                for a_type in losses:
                    # (avg_loss * (apperances_num-20) + cur_loss) / apperances_num
                    # print(avg_loss[a_type],counts[a_type],minibatch_size,losses[a_type])
                    # print(a_type)
                    num = int(counts[a_type])
                    if num >=loss_retained:
                        avg_loss[a_type] = np.mean(losses[a_type])
                    else:
                        avg_loss[a_type] = np.mean(losses[a_type][:num])
                print(avg_loss)
                # list_loss1.append(dict(losses))
                # list_loss2.append(dict(counts))
                # Parameter update
                optimizer.apply_gradients(zip(grad, model.trainable_variables), global_step=tf.train.get_or_create_global_step())

                temp_data = tf.math.softmax(avg_loss).numpy()
                temp= tf.math.softmax(avg_loss[mask_idxs_include]).numpy().tolist()
                temp_data[mask_idxs_include] = temp
                temp_data[mask_idxs_exclude] = 0
                #print('loss distribution', temp_data)

                # Keep History
                pattern_dist_history[j] = pattern_dist
                loss_dist_history[j] = temp_data

                pattern_dist = temp_data.tolist()
                print("<<<finished>>>\n")
        except:
            with open("avgloss_result3.txt","w") as f:
                f.write(json.dumps(list(avg_loss)))
            with open("dist_result3.txt","w") as f:
                f.write(json.dumps(pattern_dist))
        # IDEA: Update the pattern distribution as per the softmax of loss distribution

        
        with open("avgloss_result3.txt","w") as f:
            f.write(json.dumps(list(avg_loss)))
        with open("dist_result3.txt","w") as f:
            f.write(json.dumps(pattern_dist))
        np.save("history3.npy", list_loss3)
        np.save("counts3.npy",counts)
        root = tf.train.Checkpoint(optimizer=optimizer,
                                            model=model,
                                            optimizer_step=tf.train.get_or_create_global_step())
        root.save(checkpoint_prefix)

    # Plot history
    '''
    print("plotting")
    print(avg_loss.shape)
    plt.figure(figsize=(22, 8))
    x = [i for i in range(avg_loss.shape[0])]
    # plt.scatter(x,list(avg_loss),color='r')
    plt.plot(list(avg_loss), "ro",ms=0.1)
    plt.xlabel('Each pattern')
    plt.ylabel('Pattern distribution')
    plt.show()
    '''

    root = tf.train.Checkpoint(optimizer=optimizer,
                                        model=model,
                                        optimizer_step=tf.train.get_or_create_global_step())
    root.save(checkpoint_prefix)
if __name__ == '__main__':
    main()
