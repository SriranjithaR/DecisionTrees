# -*- coding: utf-8 -*-
"""
Created on Fri Apr 07 21:32:53 2017

@author: Sriranjitha
"""

from preprocess import splitData
from preprocess import preprocess
from preprocess import filterData
from preprocess import getWordList
from NBCClassifier import featureVector
import sys
from decisionTree import decisionTree
from bagging import bagging
from randomForest import randomForest
         
# Defaults
trainData = 'Data\yelp_data.csv'
testData  = 'Data\yelp_data.csv'
percentage = 50
index = 1

# Deciding training and test set
if(len(sys.argv) == 4):
#    print "Training data file : ", str(sys.argv[1])
#    print "Test data file : ", str(sys.argv[2])
    trainData = sys.argv[1]
    testData = sys.argv[2]
    index = int(sys.argv[3])
    rid_train,x_train,y_train,rid_test,x_test,y_test = filterData(trainData,testData);

else:
#    print "Using yelp data set as entire data set"
    rid_train,x_train,y_train,rid_test,x_test,y_test = splitData(trainData,percentage); 

# Pre-processing data
x_train = preprocess(x_train)
x_test = preprocess(x_test)


# Creating dictionary from x_train
words,wordList = getWordList(x_train)

"""
 words is now a list of (word,frequency) tuples, ordered by descending order
 of frequency
 
 wordList is a list of all unique words in the training data
"""   

# Removing most frequent 100 words
for _ in range(100):
    words.pop(0)
    
wordList = [x for x,_ in words]
"""
 The 100 most frequent words have been removed from words
 
 wordList is now a list of the words in desc order of frequency
"""
# Forming feature vector
trainfv, trainfv0, trainfv1  = featureVector(wordList[:1000], x_train, y_train)

testfv, testfv0, testfv1 = featureVector(wordList[:1000], x_test, y_test)
    
maxDepth = 10;
minRows = 10;

index = 3

#if index == 1:
#    print "ZERO-ONE-LOSS-DT ", decisionTree(trainfv, testfv, maxDepth, minRows)
#elif index == 2 :
#    print "ZERO-ONE-LOSS-BT ", bagging(trainfv, testfv, maxDepth, minRows)
#else:
#    print "ZERO-ONE-LOSS-RF ", randomForest(trainfv, testfv, maxDepth, minRows)

    
from analysis_hw4 import analysis    
print "Analysis"
an = analysis()
an.analysisDriver()

#import csv
#
#with open("temp.csv", "wb") as f:
#    writer = csv.writer(f)
#    writer.writerows(testfv)
#f.close()
#

