# -*- coding: utf-8 -*-
"""
Created on Fri Apr 07 21:30:40 2017

@author: Sriranjitha
"""
import numpy as np

# Find Gini Gain on splitting on that feature
def giniGain(branches):

    
    left = branches[0]
    right = branches[1]
    
    leftGini = 0.0
    rightGini = 0.0
    gini = 0.0
    
    leftSize = float(len(left))
    rightSize = float(len(right))
    for y in [0,1]:
        size = leftSize
        if size == 0:
            continue
#        branchnp = np.array(left)
#        proportion = (branchnp[:,-1]).tolist().count(y) / float(size)
        proportion = [row[-1] for row in left].count(y) / float(size)
        gini += proportion*proportion
    leftGini = 1.0 - gini
    
    gini = 0.0
    
    for y in [0,1]:
        size = rightSize
        if size == 0:
            continue
#        branchnp = np.array(right)
#        proportion = (branchnp[:,-1]).tolist().count(y) / float(size)
        proportion = [row[-1] for row in right].count(y) / float(size)
        gini += proportion*proportion
    rightGini = 1.0 - gini
    
    
    totSize = leftSize + rightSize
    
    totGini = leftGini*(leftSize/totSize) + rightGini*(rightSize/totSize);
    return totGini
    
    
# Split data on given feature index
def splitOnFeature(trainfv, f):
   
    left, right = list(), list()
    for row in trainfv:
        if row[f] == 0:
            left.append(row)
        else:
            right.append(row)
    
    return left, right
    

    
# Find the next best feature to split on
def findNextSplit(trainfv):
    feat = len(trainfv[0])-1;  # no. of features   
    bestFeat = 0
    bestBranch= [list(), list()]
    minGini = 100
    for f in range(feat):        
        branches = splitOnFeature(trainfv,f)
        gini = giniGain(branches)
        if(gini < minGini):
            minGini = gini
            bestFeat = f
            bestBranch = branches           
    return {'feature':bestFeat,'branches':bestBranch}
        
    
# Get the max count class
def getLeafClass(data):
    classLabels = [row[-1] for row in data]
    return max(set(classLabels),key=classLabels.count)
  
    
# Recursively grow tree
def split(root, maxDepth, minRows, currDepth):
    
#    print "******************************************"
#    print "Depth : ",currDepth
#    print "Feature : ",root['feature']
#    
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
#        print "Left/right not found"
        return
    elif not len(right):
        root['left'] = root['right'] = getLeafClass(left)
#        print "Left/right not found"
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
        root['left'] = findNextSplit(left)
        split(root['left'], maxDepth, minRows, currDepth + 1)
    
    # Process right branch
#    print "_________________________________________________________"
#    print "Processing right"
    if(len(right) <= minRows):
        root['right'] = getLeafClass(right)
#        print "len(right) : ",len(right)
    else:
        root['right'] = findNextSplit(right)
        split(root['right'], maxDepth, minRows, currDepth + 1)
    
        
        
# Build DT
def getTree(trainfv, testfv, maxDepth=10, minRows=10):
    root = findNextSplit(trainfv)
    split(root, maxDepth, minRows, 1) 
    return root
    

# Driver for decision Tree
def decisionTree(trainfv, testfv, maxDepth=10, minRows=10):
    tree = getTree(trainfv, testfv, maxDepth, minRows)
    return testDT(tree,testfv)

    
    
# Calculating accuracy
def testDT(root,testfv):
    predVals = []
    
    # Get predicted values using treee
    for sample in testfv:
        pred = prediction(root,sample)
        predVals.append(pred)   
        
    # Compare predicted values with actual values 
    test = np.array(testfv)
    actual = test[:,-1]
    predVals = np.array(predVals)
    
    correctPred = (predVals == actual).astype(int)
      
    acc = (float(sum(correctPred))/float(len(correctPred)))
    zeroOneLoss = 1-acc
    acc *= 100
    return zeroOneLoss
    
    
# Parse tree and make prediction on sample
def prediction(root, sample):
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
    
    