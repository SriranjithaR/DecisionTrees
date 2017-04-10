# -*- coding: utf-8 -*-
"""
Created on Sat Apr 08 18:00:03 2017

@author: Sriranjitha
"""
import numpy as np
from decisionTree import decisionTree, getTree, testDT, prediction, split, giniGain, splitOnFeature, findNextSplit, getLeafClass

def bagging(trainfv, testfv, maxDepth, minRows):
    
    train = np.array(trainfv)
    predVals = np.zeros(len(testfv))
    
    numTrees = 50
    for iter in range(numTrees): # for 50 iterations
        indices = np.random.choice(np.arange(train.shape[0]),train.shape[0])
        trainBag = np.take(train, indices, 0)
        tree = getTree(trainBag.tolist(),testfv)
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