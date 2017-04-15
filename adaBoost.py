# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:21:18 2017

@author: Sriranjitha
"""


import numpy as np

from numpy.random import rand
from numpy import cumsum, sum, searchsorted

def trainBoosting(trainfv, testfv, numTrees ):
    
    n = len(trainfv)
          
    selClf = [] # selected classifiers
    alpha = [] # list of alpha values
    wt = np.ones(n) * 1.0 / float(n) # list of weights
    
    for k in range(numTrees):
        
        print "Tree : ",k
        
        for i in range(len(trainfv)):
            trainfv[i][-1] *= wt[i]
            print trainfv[i][-1]
            
        error = 0.0
#        indices = resample(wt)
#        wtrainfv = []
        
#        for i in range(n):
#            wtrainfv.append(trainfv[indices[i]])
            
#        hk = getTree(wtrainfv, testfv, wt, maxDepth=10, minRows=10)
        hk = getTree(trainfv, testfv, wt, maxDepth=10, minRows=10)
       

        predTrain_k = [prediction(hk,row) for row in trainfv]
#        predTest_k = [prediction(hk,row) for row in testfv]

    
        for i in range(len(predTrain_k)):
            predicted = predTrain_k[i]
            error += (predicted != trainfv[i][-1]) 
#            error += (predicted != trainfv[i][-1]) * wt[i]
            
#        if(error > 0.5):
#            continue
        if error == 0.0:
            continue;
            
        a = 0.5 * np.log((1 - error) / error)

        alpha.append(a)

        selClf.append(hk)
        
        for i in range(n):
            y = trainfv[i][-1]
            h = predTrain_k[i]
#            h = (-1 if h == -1 else 1)
#            y = (-1 if y == -1 else 1)
            wt[i] = wt[i] * np.exp(-a * h * y)
            
        wt = [w/sum(wt) for w in wt]
        print "sum of weight = ",sum(wt)

        
    return zip(alpha, selClf)

def resample(weights):
    t = cumsum(weights)
    s = sum(weights)
    return searchsorted(t,rand(len(weights))*s)

        
def classify(weight_classifier, example):
    classification = 0
    for (weight, classifier) in weight_classifier:
#        if prediction(classifier, example) == 1:
        if prediction(classifier, example) >= 0:    
            ex_class = 1
        else:
            ex_class = -1
        
        print "prediction : ",ex_class," weight = ", weight
        classification += weight * ex_class
        
    print "classification = ",classification
    return (1 if classification > 0 else -1)


def boosting(trainfv, testfv, maxDepth=10, minRows=10, numTrees=50):    
   
    for row in trainfv:
        if(row[-1]==0):
            row[-1] = -1;
               
    for row in testfv:
        if(row[-1]==0):
            row[-1] = -1;
               
    train_res = trainBoosting(trainfv, testfv, numTrees)    
    print "train boost over, ",train_res
    predTest = []
#    for row in testfv:
    for row in trainfv:
#        print "Appending to predtest"
        x = classify(train_res, row)
        predTest.append(x)
        print "test pred = ",x
        
    predTest = np.array(predTest)
    yTest = np.array([row[-1] for row in trainfv])
#    yTest = np.array([row[-1] for row in testfv])
    correctPred = (predTest == yTest).astype(int)
    acc = (float(sum(correctPred))/float(len(correctPred)))
    zeroOneLoss = 1-acc
    acc *= 100
    return zeroOneLoss

# Split data on given feature index
def splitOnFeature(trainfv, f, Dk):
    
    left, right = list(), list()
    leftWt, rightWt = list(), list()
    for i in range(len(trainfv)):
        if trainfv[i][f] == 0:
            left.append(trainfv[i])
            leftWt.append(Dk[i])
        else:
            right.append(trainfv[i])
            rightWt.append(Dk[i])
    
    return left, right, leftWt, rightWt
    

# Parse tree and make prediction on sample
def prediction(root, sample):
#        print type(root)
    if(sample[root['feature']] == 0):        
        if isinstance(root['left'],dict):
            return prediction(root['left'],sample)
        else:
            return root['left']           
    else:       
        if isinstance(root['right'],dict):
            return prediction(root['right'],sample)
        else:
            return root['right']
     
          
# Build DT
def getTree(trainfv, testfv, Dk, maxDepth=10, minRows=10):
    root = findNextSplit(trainfv, Dk)
    split(root, Dk, maxDepth, minRows, 1) 
    print root
    return root
    
    
    
# Find the next best feature to split on
def findNextSplit(trainfv, Dk):
    feat = len(trainfv[0])-1;  # no. of features   
    bestFeat = 0
    bestBranch= [list(), list()]
    minGini = 100
    branches  = [list(),list()]
    branchWts = [list(),list()]
    for f in range(feat):        
        branches[0],branches[1],branchWts[0],branchWts[1] = splitOnFeature(trainfv,f, Dk)
#        gini = giniGain(branches, branchWts)
        gini = giniGain(branches)
        if(gini < minGini):
            minGini = gini
            bestFeat = f
            bestBranch = branches           
    return {'feature':bestFeat,'branches':bestBranch}
        
    
# Get the max count class
def getLeafClass(data):
    classLabels = [row[-1] for row in data]
    classLabels = [-1 if classLabels<0 else 1]
    leaf =  max(set(classLabels),key=classLabels.count)
    print "leaf = ",leaf
    return leaf
  
    
# Recursively grow tree
def split(root, Dk, maxDepth, minRows, currDepth):
    """ 
    root['feature'] => which feature to was used to create root['branches']
    for the next recursive call, this feature has to be removed from the branches
    before calculating next best feature 
    """
    
    left, right = root['branches']
    del(root['branches'])
    
#        if not left and not right:
#            return
    
    # Check if the node is a leaf
    if not len(left): 
        root['left'] = root['right'] = getLeafClass(right)
        return
    elif not len(right):
        root['left'] = root['right'] = getLeafClass(left)
        return
        
    # Check for max depth
    if(currDepth >= maxDepth):
        root['left'], root['right'] = getLeafClass(left), getLeafClass(right)
        return
        
    # Process left branch
    if(len(left) <= minRows):
        root['left'] = getLeafClass(left)
    else:
        root['left'] = findNextSplit(left, Dk)
        split(root['left'], Dk, maxDepth, minRows, currDepth + 1)
    
    # Process right branch
    if(len(right) <= minRows):
        root['right'] = getLeafClass(right)
    else:
        root['right'] = findNextSplit(right, Dk)
        split(root['right'], Dk, maxDepth, minRows, currDepth + 1)
        
  # Find Gini Gain on splitting on that feature
def giniGain(branches):
 
    left = branches[0]
    right = branches[1]

#        print "lw = ",leftWts
    leftGini = 0.0
    rightGini = 0.0
    gini = 0.0
    
#        p = len(left[0])-1
    
#        print "len(left) = ",len(left)
#        print "len(left[0]) = ",len(left[0])
    leftSize = len(left)
    rightSize = len(right)
    
    lprop1 = 0
    lprop0 = 0
    for i in range(leftSize):
        if(left[i][-1] >= 0 ):
            lprop1 += 1
        else:
            lprop0 += 1
    
    rprop1 = 0
    rprop0 = 0
    for i in range(rightSize):
        if(right[i][-1] >= 0):
            rprop1 += 1
        else:
            rprop0 += 1
    
    gini = 0.0

    if(lprop1 or lprop0):        
        gini += np.square(lprop0/(lprop0+lprop1))
        gini += np.square(lprop1/(lprop0+lprop1)) 
    
    leftGini = 1.0 - gini
    
    gini = 0.0
    
    if(rprop1 or rprop0):
        gini += np.square(rprop0/(rprop0+rprop1))
        gini += np.square(rprop1/(rprop0+rprop1)) 
    
    rightGini = 1.0 - gini
    
    totGini = leftGini*((lprop0+lprop1)/(rprop0+rprop1+lprop1+lprop0)) + rightGini*((rprop0+rprop1)/(rprop0+rprop1+lprop1+lprop0));
#    print totGini
    return totGini      
    
## Find Gini Gain on splitting on that feature
#def giniGain(branches, Dk):
# 
#    left = branches[0]
#    right = branches[1]
#    leftWts = Dk[0]
#    rightWts = Dk[1]
##        print "lw = ",leftWts
#    leftGini = 0.0
#    rightGini = 0.0
#    gini = 0.0
#    
##        p = len(left[0])-1
#    
##        print "len(left) = ",len(left)
##        print "len(left[0]) = ",len(left[0])
#    leftSize = len(left)
#    rightSize = len(right)
#    
#    lprop1 = list()
#    lprop0 = list()
#    for i in range(leftSize):
#        if(left[i][-1]==1 ):
#            lprop1.append(leftWts[i])
#        else:
#            lprop0.append(leftWts[i])
#    
#    rprop1 = list()
#    rprop0 = list()
#    for i in range(rightSize):
#        if(right[i][-1]==1):
#            rprop1.append(rightWts[i])
#        else:
#            rprop0.append(rightWts[i])
#    
#    gini = 0.0
#
#    if(len(lprop1) or len(lprop0)):        
#        gini += np.square(sum(lprop0)/(sum(lprop0)+sum(lprop1)))
#        gini += np.square(sum(lprop1)/(sum(lprop0)+sum(lprop1))) 
#    
#    leftGini = 1.0 - gini
#    
#    gini = 0.0
#    
#    if(len(rprop1) or len(rprop0)):
#        gini += np.square(sum(rprop0)/(sum(rprop0)+sum(rprop1)))
#        gini += np.square(sum(rprop1)/(sum(rprop0)+sum(rprop1))) 
#    
#    rightGini = 1.0 - gini
#    
#    totGini = leftGini*((sum(lprop0)+sum(lprop1))/(sum(rprop0)+sum(rprop1)+sum(lprop1)+sum(lprop0))) + rightGini*((sum(rprop0)+sum(rprop1))/(sum(rprop0)+sum(rprop1)+sum(lprop1)+sum(lprop0)));
##    print totGini
#    return totGini
#
#def giniGain(branches):
#
#    
#    left = branches[0]
#    right = branches[1]
#    
#    leftGini = 0.0
#    rightGini = 0.0
#    gini = 0.0
#    
#    leftSize = float(len(left))
#    rightSize = float(len(right))
#    for y in [-1,1]:
#        size = leftSize
#        if size == 0:
#            continue
##        branchnp = np.array(left)
##        proportion = (branchnp[:,-1]).tolist().count(y) / float(size)
#        proportion = [row[-1] for row in left].count(y) / float(size)
#        gini += proportion*proportion
#    leftGini = 1.0 - gini
#    
#    gini = 0.0
#    
#    for y in [0,1]:
#        size = rightSize
#        if size == 0:
#            continue
##        branchnp = np.array(right)
##        proportion = (branchnp[:,-1]).tolist().count(y) / float(size)
#        proportion = [row[-1] for row in right].count(y) / float(size)
#        gini += proportion*proportion
#    rightGini = 1.0 - gini
#    
#    
#    totSize = leftSize + rightSize
#    
#    totGini = leftGini*(leftSize/totSize) + rightGini*(rightSize/totSize);
#    return totGini
#    