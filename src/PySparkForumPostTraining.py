# wordcount program

from __future__ import print_function

import sys
from operator import add

from pyspark import SparkContext

def map_get_vocabulary(doc):
    doc = doc.split(' ')
    doc.pop(0)
    return doc

def map_get_text_vocab(_class):
    _class_id = _class[0]
    _class_docs = _class[1]
    text = []
    for doc in _class_docs:
	for word in doc:
		text.append(word)
    return (_class_id, text)

def map_classes(doc):
    doc = doc.split(' ')
    _class = doc.pop(0)
    return (_class,[doc])

def remap_class_word(_class_word):
    _class_id = (_class_word[0],_class_word[1])
    return (_class_id,1)

def remap_class_word_dic(_class):
    _class_id = _class[0][0]
    _dict = {_class[0][1]:_class[1]}
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
    sc = SparkContext(appName="PySparkForumPostTraining")
    lines = sc.textFile(sys.argv[1], 1)
    lines.cache()
    
    # to get the vocabulary   
    stage0 = lines.flatMap(map_get_vocabulary)

    # to get |docs|
    stage1 = lines.map(map_classes) \
                  .reduceByKey(reducer_add)
    
    # to get the the word count by class.
    stage2 = stage1.map(map_get_text_vocab).flatMapValues(lambda x: x) \
                   .map(remap_class_word).reduceByKey(reducer_add) \
		   .map(remap_class_word_dic).reduceByKey(reducer_dict)
    
    total_words = stage0.count()
    vocabulary = stage0.distinct().collect()
    docs_by_class = stage1.collect()
    words_by_class = stage2.collect() 

    print ("Total words in training set: %d, Unique words: %d"%(total_words,len(vocabulary)))

    print ("\n\nDocs in class")
    for (_class, docs) in sorted(docs_by_class):
        print("%s: %i" %(_class, len(docs)))
 
    validation_counter = 0
    print ("\n\nUnique words per doc in class")
    for (_class, words) in sorted(words_by_class):
        print("%s: %i" %(_class, len(words)))
        for word,count in words.iteritems():
            validation_counter += count

    if validation_counter == total_words:
        print("Great Job")
    else:
        print("Something went wrong total words: %d | validation words %d"%(total_words, validation_counter))
    sc.stop()

