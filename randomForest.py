# -*- coding: utf-8 -*-
"""
Created on Sat Apr 08 20:48:47 2017

@author: Sriranjitha
"""
import numpy as np
from random import randrange
from decisionTree import decisionTree, testDT, prediction, giniGain, splitOnFeature, getLeafClass

def randomForest(trainfv, testfv, maxDepth=10, minRows=10,numTrees=50):
        
    train = np.array(trainfv)
    predVals = np.zeros(len(testfv))
    
    totFeatures = len(trainfv[0])-1
    numFeatures = np.sqrt(totFeatures)
    
    for iter in range(numTrees): # for 50 iterations
        indices = np.random.choice(np.arange(train.shape[0]),train.shape[0])
        trainBag = np.take(train, indices, 0)
        tree = getTree(trainBag.tolist(),testfv, maxDepth, minRows, numFeatures)
        for row in range(len(testfv)):
            pred = prediction(tree,testfv[row])
            predVals[row] = predVals[row] + pred
#        predVals = np.add(predVals + np.array(np.apply_along_axis(prediction, 1, tree, testfv)))
        
    predVals = np.divide(predVals,float(numTrees))
    predVals[predVals >= 0.5] = 1
    predVals[predVals < 0.5] = 0
    
    # Compare predicted values with actual values 
    test = np.array(testfv)
    actual = test[:,-1]
    predVals = np.array(predVals)
    
    correctPred = (predVals == actual).astype(int)
      
    acc = (float(sum(correctPred))/float(len(correctPred)))
    zeroOneLoss = 1-acc
    acc *= 100
    return zeroOneLoss
    
def split(root, maxDepth, minRows, numFeatures, currDepth):
        
    """ 
    root['feature'] => which feature to was used to create root['branches']
    for the next recursive call, this feature has to be removed from the branches
    before calculating next best feature 
    """
    
    left, right = root['branches']    
    del(root['branches'])
    
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
#    print "_________________________________________________________"
#    print "Processing left"
    if(len(left) <= minRows):
        root['left'] = getLeafClass(left)
#        print "len(left) : ",len(left)
    else:
        root['left'] = findNextSplit(left, numFeatures)
        split(root['left'], maxDepth, minRows, numFeatures, currDepth + 1)
    
    # Process right branch
#    print "_________________________________________________________"
#    print "Processing right"
    if(len(right) <= minRows):
        root['right'] = getLeafClass(right)
#        print "len(right) : ",len(right)
    else:
        root['right'] = findNextSplit(right, numFeatures)
        split(root['right'], maxDepth, minRows, numFeatures, currDepth + 1)
    
        
# Build DT
def getTree(trainfv, testfv, maxDepth, minRows, numFeatures):
    root = findNextSplit(trainfv, numFeatures)
    split(root, maxDepth, minRows, numFeatures, 1) 
    return root

    
# Find the next best feature to split on
def findNextSplit(trainfv, numFeatures):
    feat = len(trainfv[0])-1 ; # number of total features
    
    bestFeat = 0
    bestBranch= [list(), list()]
    minGini = 100
    selFeatures = list()
    while(len(selFeatures) < numFeatures):
        f = randrange(feat)
        if f not in selFeatures:
            selFeatures.append(f)
    
    for f in selFeatures:       
        branches = splitOnFeature(trainfv,f)
        gini = giniGain(branches)
        #print "intermediate gini for feature", f," : ",gini
        if(gini < minGini):
            minGini = gini
            bestFeat = f
            bestBranch = branches
            
    return {'feature':bestFeat,'branches':bestBranch}