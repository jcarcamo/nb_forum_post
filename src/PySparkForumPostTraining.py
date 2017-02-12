#!/usr/bin/python

'''
Spark ForumPost Classifier Naive Bayes Training program
Developer: Juan Carcamo
Purpose: Create the lookup table of stats for NB
         
Details: 
         
Credits: Juan Gonzalo Carcamo, Matias Gil
'''

from __future__ import print_function

import os
import sys
import cPickle as pickle
from classifier import Classifier

from pyspark import SparkContext

DEBUG = True

def map_get_vocabulary(doc):
    doc = doc.split(' ')
    doc.pop(0)
    return doc

def map_get_text_vocab(vocabulary_len, _class):
    _class_id = _class[0]
    _class_docs = _class[1]
    docs_len = len(_class_docs)
    text = [word for doc in _class_docs for word in doc]
    n = len(text)
    _class_id = (_class[0],docs_len,n,vocabulary_len)
    return (_class_id, text)

def map_get_not_in_text(vocabulary, _class):
    _class_id = _class[0]
    _class_docs = _class[1]
    docs_len = len(_class_docs)
    #for doc in _class_docs:
    #    for word in doc:
    #        text.append(word)
    text = [word for doc in _class_docs for word in doc]
    not_in_text = set(vocabulary) - set(text)
    vocabulary_len = len(vocabulary)
    n = len(text)
    _class_id = (_class[0],docs_len,n,vocabulary_len)
    return (_class_id, list(not_in_text))

def map_classes(doc):
    doc = doc.split(' ')
    _class = doc.pop(0)
    return (_class,[doc])

def map_get_class_prob(total_docs,_class):
    _class_id = _class[0][0]
    docs_len = _class[0][1]
    _class_prob = float(docs_len)/total_docs
    return((_class_id, _class_prob),_class[1])

def remap_class_word(_class_word):
    _class_id = (_class_word[0][0], _class_word[0][1], _class_word[0][2],_class_word[0][3], _class_word[1])
    return (_class_id,1)

def remap_class_word_not_in_text(_class_word):
    _class_id = (_class_word[0][0], _class_word[0][1], _class_word[0][2],_class_word[0][3], _class_word[1])
    return (_class_id,0)

def remap_class_word_dic(_class):
    _class_id = (_class[0][0],_class[0][1], _class[0][2],_class[0][3])
    n_k = _class[1]
    n = _class[0][2]
    vocabulary_len = _class[0][3]
    p_w_c = (n_k + 1.0) / (n + vocabulary_len)
    _dict = {_class[0][4]:p_w_c}
    return (_class_id, _dict)

def reducer_add(x,y):
     return x+y

def reducer_tuple_add(x,y):
    return (x[0]+y[0],y[1]+y[1])

def reducer_dict(x,y):
    x.update(y)
    return x

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: PySparkForumPostTraining.py <file>", file=sys.stderr)
        exit(-1)
    print("Started")
    sc = SparkContext(appName="PySparkForumPostTraining")
    training_data_file = sys.argv[1]
    lines = sc.textFile(training_data_file, 1)
    lines.cache()
    
    # to get the vocabulary   
    stage0 = lines.flatMap(map_get_vocabulary)
    
    vocabulary_len = stage0.distinct().count()

    # to get |docs|
    stage1 = lines.map(map_classes).reduceByKey(reducer_add) \

    # to get the the word count by class.
    stage2 = stage1.map(lambda x: map_get_text_vocab(vocabulary_len,x)).flatMapValues(lambda x: x) \
                   .map(remap_class_word).reduceByKey(reducer_add) \
		   .map(remap_class_word_dic).reduceByKey(reducer_dict)
   
    total_words = stage0.count()
    vocabulary = stage0.distinct().collect()
    stage3 = stage1.map(lambda x: map_get_not_in_text(vocabulary,x)).flatMapValues(lambda x: x) \
                   .map(remap_class_word_not_in_text) \
                   .map(remap_class_word_dic).reduceByKey(reducer_dict)

    total_docs_in_training = lines.count()
    final_stage = stage2.union(stage3).reduceByKey(reducer_dict) \
                                      .map(lambda x: map_get_class_prob(total_docs_in_training,x))
    output = final_stage.collect()

    if DEBUG:
        validation_word_counter = 0.0
        validation_class_counter = 0.0

    classifiers_dict = {}
    
    for (_class, words) in sorted(output):
        classifier = Classifier()
        classifier.name = _class[0]
        classifier.class_probability = _class[1]
        classifier.features = words
        classifiers_dict.update({_class[0]:classifier})
        if DEBUG:
            print("class:%s\t\tp_c:%.10f\t\tp_w_c_len:%d"%(_class[0],_class[1],len(words)))
            validation_word_probability = 0.0
            for word,count in words.iteritems():
                validation_word_probability += count
            print("sum of probabilities: %.20f"%(validation_word_probability))
            validation_word_counter += validation_word_probability
            validation_class_counter += _class[1]

    if DEBUG:        
        if abs(validation_word_counter - len(output)) < 0.0000000001 and abs(validation_class_counter - 1) < 0.0000000001:
            print("Great Job!!! \n total classes probabilities:%.20f"%(validation_class_counter))
        else:
            print("Something went wrong!!!")
            print("sum of each class prob  %.20f | validation probability %.20f"%(float(len(output)), validation_class_counter))
            print("sum of words prob  %.20f | validation probability %.20f"%(float(len(output)), validation_word_counter))
  

    print ("Classifier Stats:")
    for _class,classifier in classifiers_dict.iteritems():
        classifier.print_classifier()
        print('\n')

    output_filename = training_data_file.split('/')[-1].split('.')[0] 
    
    print ("Saving classifier for "+output_filename)
    
    new_file_path = '../output/'+output_filename+'.pkl'
    if not os.path.exists(os.path.dirname(new_file_path)):
        try:
            os.makedirs(os.path.dirname(new_file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    try:
        pickle.dump(classifiers_dict, open(new_file_path,'wb'))
        print ("Finished!")
    except (picke.PickleError, pickle.PicklingError) as e:
        print ("Error saving objects to pickle")
    sc.stop()
