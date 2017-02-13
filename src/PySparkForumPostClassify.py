#!/usr/bin/python

'''
Spark ForumPost Classifier Naive Bayes Classify program
Developer: Juan Carcamo
Purpose: Classify documents using NB
         
Details: 
         
Credits: Juan Gonzalo Carcamo, Matias Gil
'''

from __future__ import print_function

import os
import sys
import math
import cPickle as pickle
from classifier import Classifier

from pyspark import SparkContext

DEBUG = True
classifiers = {}

def load_classifiers(pickle_file_path):
    global classifiers
    if not classifiers:
        pkl_file = open(pickle_file_path, 'rb')
        classifiers = pickle.load(pkl_file)
        pkl_file.close()

def map_add_classifier(classifier, doc):
    doc_id = doc[1]
    doc_val = doc[0].split(' ')
    doc_class = doc_val.pop(0)
    return ((doc_id, doc_class, classifier),doc_val)

def filter_inexistent_words(class_word_obj):
    global classifiers
    classifier = classifiers[class_word_obj[0][2]]
    feature_probability = classifier.get_feature_probability(class_word_obj[1])
    if feature_probability != 0:
        return class_word_obj

def map_log_of_probability(class_word_obj):
    global classifiers
    classifier = classifiers[class_word_obj[0][2]]
    feature_probability = classifier.get_feature_probability(class_word_obj[1])
    if feature_probability != 0:
        log_p_w_c = math.log(feature_probability)
        return (class_word_obj[0],log_p_w_c)

def map_get_class_prob(class_doc):
    global classifiers
    classifier = classifiers[class_doc[0][2]]
    c_nb = math.log(classifier.class_probability) + class_doc[1]
    return ((class_doc[0][0],class_doc[0][1]),(class_doc[0][2],c_nb))


def map_accuracy_classification(doc):
    global classifiers
    classifier = classifiers[doc[1][0]]
    real_class = doc[0][1]
    predicted_class = classifier.name
    if real_class == predicted_class:
        return ((real_class,'correct'), 1)
    else:
        return ((real_class,'wrong'), 1)

def reducer_add(x,y):
     return x+y

def reducer_get_classification(c_nb1, c_nb2):
    if c_nb1[1] < c_nb2[1]:
        return c_nb2
    else:
        return c_nb1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: PySparkForumPostTraining.py <classifiers_file> <test_file>", file=sys.stderr)
        exit(-1)
    print("Started Classification")
    sc = SparkContext(appName="PySparkForumPostClassify", pyFiles=['./classifier.py'])
    
    classifiers_file_path = sys.argv[1]
    load_classifiers(classifiers_file_path)
    test_data_file = sys.argv[2]

    
    lines = sc.textFile(test_data_file, 1).zipWithUniqueId()
    lines.cache()
    
    stage0 = sc.emptyRDD()
    for classifier_key, classifier in classifiers.iteritems():
        tmp_rdd = lines.map(lambda x: map_add_classifier(classifier.name,x))
        stage0 = stage0.union(tmp_rdd)
    
    #print(stage0.first())
    
    # map by value each word in documents.
    stage1 = stage0.flatMapValues(lambda x: x).filter(filter_inexistent_words)  \
                   .map(map_log_of_probability) \
                   .reduceByKey(reducer_add) \
                   .map(map_get_class_prob) \
                   .reduceByKey(reducer_get_classification) \
                   .map(map_accuracy_classification) \
                   .reduceByKey(reducer_add)
    #print(stage1.take(20))
    
    if DEBUG:
        total_test_docs = lines.count()
        output = stage1.collect()
        
        correctly_classified = 0
        wrongly_classified = 0
        calc_test_docs = 0
        results_by_class = {}
        for k,v in output:
            if k[0] not in results_by_class:
                results_by_class[k[0]] = { 'correct':0, 'incorrect':0}
            
            if k[1] == 'correct':
                correctly_classified += v
                results_by_class[k[0]]['correct'] += v
            elif k[1] == 'wrong':
                wrongly_classified += v
                results_by_class[k[0]]['incorrect'] += v
            calc_test_docs += v
        
        if total_test_docs == calc_test_docs:
            print("Classified docs: %d"%(calc_test_docs))
            for result,v in results_by_class.iteritems():
                if v['correct'] != 0 or v['incorrect'] != 0:
                    print("Class: %s    Correct: %d    Incorrect: %d    Accuracy:%.10f" \
                        %(result, v['correct'], v['incorrect'], float(v['correct'])/(v['correct']+v['incorrect'])))
                else:
                   print("No documents for class: %s"%(result))
            print("\n\nAccuracy of predictions: %.10f"%(float(correctly_classified)/calc_test_docs))
        else:
            print("Input docs: %d    Classified docs: %d"%(total_test_docs, calc_test_docs))
            print("Right class: %d    Wrong class: %d"%(correctly_classified, wrongly_classified))
        
    print ("Finished!")
    sc.stop()
