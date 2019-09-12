import corti_data_manager.synthetic.generate_data as gd     
from keras.preprocessing.text import Tokenizer

conf = gd.process_conf("corti-data-manager/tests/data/confs/stream_data_gen.yml")          
# print(conf)
sqg = gd.StreamingQueryGenerator(conf)    
first_batch =sqg.generate_queries()
print(len(sqg.get_patterns()))
print(sqg.get_entities())
# print(first_batch)
for q,i in first_batch:
    print(i,q)
    break
for each in q:
    print(each["logical_forms"][0]["predicate"])

vocab = []
entities = sqg.get_entities()
print(sqg.get_patterns())
for p in sqg.get_patterns():
    print(p.phrase)
    vocab.append(p.phrase)
print(entities.entity_types["channel"]._weights_by_language["all"]["entities"])
print(entities.entity_types["trailer"]._weights_by_language["all"]["entities"])
for t in entities.entity_types:
    entity_list = entities.entity_types[t]._weights_by_language["all"]["entities"]
    for e in entity_list:
        print(e.phrase)
        vocab.append(e.phrase)
print(vocab)
tokenizer = Tokenizer(lower=True, filters='')
tokenizer.fit_on_texts(vocab)
print(entities.action_dependent_entities)
print(entities.all_language_value)