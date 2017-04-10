# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 00:47:43 2017

@author: Sriranjitha
"""
import pandas as pd
import numpy as np
import string
import operator
from random import sample

def filterData(trainFile, testFile):
    
    df  = pd.read_csv(trainFile, sep='\t', dtype='str')
    df2 = pd.read_csv(testFile, sep='\t', dtype='str')
    
    train = df
    test = df2
    
    # Training data
    x_train = train.as_matrix()    
    rid_train = x_train[:,0]
    y_train = x_train[:,1]
    x_train = x_train[:,2]
    
    # Test data
    x_test = test.as_matrix()   
    rid_test = x_test[:,0]
    y_test = x_test[:,1]
    x_test = x_test[:,2]

    
    return rid_train,x_train,y_train,rid_test,x_test,y_test
    
def getLines(filename, percentage):
     # Get file as array
    infile = open(filename)
    lines = infile.readlines()
    
    lines = np.array(lines)
  
    l = len(lines) #length of data 
    
    g = (float(percentage)/100.0)
    f = float(l)*g  #number of elements you need
    indices = sample(range(l),int(f))
    
    train_data = lines[indices]
    test_data = np.delete(lines,indices)
    
    return train_data
    
def splitColumns(train_data):
    col1 = []
    col2 = []
    col3 = []
    
    
    n_train_data = [[0 for y in range(3)] for x in range(len(train_data))]
    
    for i in range(len(train_data)):
        n_train_data[i] = train_data[i].split("\t");
        col1.append(n_train_data[i][0]);
        col2.append(n_train_data[i][1]);
        col3.append(n_train_data[i][2]);
        
        
    x_train = np.array(n_train_data)

        
#    print len(n_train_data)
#    print len(n_train_data[0])

    rid_train = col1
    y_train = col2
    x_train = col3
    
    return rid_train,x_train,y_train

    
def splitData(filename,percentage):
    
    infile = open(filename)
    lines = infile.readlines()
    
    lines = np.array(lines)
  
    l = len(lines) #length of data 
    
    g = (float(percentage)/100.0)
    f = float(l)*g  #number of elements you need
    indices = sample(range(l),int(f))
    
    li = list(lines)
    train_data = lines[indices]
    test_data = np.delete(lines,indices)
    
    n_train_data = [[0 for y in range(3)] for x in range(len(train_data))]
    n_test_data = [[0 for y in range(3)] for x in range(len(test_data))]
                     
    col1 = []
    col2 = []
    col3 = []
    
    tcol1 = []
    tcol2 = []
    tcol3 = []
    
    for i in range(len(train_data)):
        n_train_data[i] = train_data[i].split("\t");
        col1.append(n_train_data[i][0]);
        col2.append(n_train_data[i][1]);
        col3.append(n_train_data[i][2]);
        
        
    
    for i in range(len(test_data)):
        n_test_data[i] = test_data[i].split("\t");
        tcol1.append(n_test_data[i][0]);
        tcol2.append(n_test_data[i][1]);
        tcol3.append(n_test_data[i][2]);
        
    x_train = np.array(n_train_data)
    x_test = np.array(n_test_data)
        
#    print len(n_train_data)
#    print len(n_train_data[0])

    rid_train = col1
    y_train = col2
    x_train = col3
    
    rid_test = tcol1
    y_test = tcol2
    x_test = tcol3

    return rid_train,x_train,y_train,rid_test,x_test,y_test
    
def preprocess(reviewText):
    for i in range(len(reviewText)):       
        reviewText[i] = str(reviewText[i]).lower()
        reviewText[i] = reviewText[i].translate(None,string.punctuation)
        reviewText[i] = reviewText[i].split()
        
    return reviewText
    
def getWordList(x_train):
    # Creating dictionary from x_train
    dlist = {}
    words = {}
    
    for i in range(len(x_train)):
        for word in x_train[i]:
            if(not words.has_key(word)):
                words[word] = 1;
                dlist[word] = i;
                
            else:
                if(dlist[word] != i):
                    dlist[word] = i;
                    words[word] += 1;
                  

    
    # Sorting words by frequency
    words = sorted(words.items(), key = operator.itemgetter(1), reverse = True)
    """
     words is now a list of (word,frequency) tuples, ordered by descending order
     of frequency
    """   

    wordList = [x for x,_ in words]
    return words,wordList