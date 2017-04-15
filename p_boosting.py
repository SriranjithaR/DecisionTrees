# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 18:27:45 2017

@author: Sriranjitha
"""

import numpy as np

class boosting:
#we convert to list the array numpy
    def __init__(self,max_depth,min_size,num_trees):
        self.max_depth = max_depth
        self.min_size = min_size
        self.num_trees = num_trees
        self.lst_dt = None
        self.alphas = []
        
    def train_boost(self,X,Y):
        ep = 10**-6
        et_arr = []
        lst_dt = []
        #sz = int(dataset.shape[0]*self.sz_pro)
        wt = np.ones(shape = Y.shape)
        wt = wt/np.sum(wt)
        wt_y = np.multiply(wt,Y)
        dataset =  np.hstack((X,wt_y))
        for i in range(self.num_trees):
            print "tree : ",i
            #print('on {} tree'.format(i+1))
		#the tree gonaaa remain same, just count negatives instead of 0 and positives instead of 1
            d_t = getTree(dataset.tolist(),maxDepth=self.max_depth,minRows=self.min_size)
#            d_t.train(dataset.tolist())
            lst_dt.append(d_t)
            
            y_predict = [prediction(d_t, row) for row in dataset]
            y_predict = np.array(y_predict).reshape(Y.shape)
            print y_predict[:10]
            
    
            et = np.sum(wt[Y !=y_predict])
            et_arr.append(et)
            if et == 0:
                alpha = (0.5)*np.log(((1 - et + ep)/(et + ep)))
                self.alphas.append(alpha)
                break
            Y_Y_pred = np.multiply(Y,y_predict)
            alpha = 0.5*np.log((1-et)/et)
            self.alphas.append(alpha)
            
            wt = np.multiply(wt,np.exp(-1.0*alpha*Y_Y_pred))
            #print(np.sum(wt))
            wt = wt/np.sum(wt)
           # print(np.sum(wt))
            
#            for i in range(len(y_predict)):
#                if(Y[i]!=y_predict[i]):
#                    print "wrong,index : ",i," weight: ",wt[i]
#                else:
#                    print "right,index : ",i," weight: ",wt[i]
#                
#                
            
            reshape_shape = dataset[:,-1].shape
            dataset[:,-1] = np.multiply(wt,Y).reshape(reshape_shape) 
            print wt[:10]
            
        self.lst_dt = lst_dt
#        return et_arr
        
    def test_boost(self,X):
        pred_y_arr = []
        i=0
        for dt in self.lst_dt:
            pred_y = [prediction(dt,row) for row in  X.tolist()]
            #print(max(pred_y))
            #print(min(pred_y))
            pred_y_arr.append(self.alphas[i]*np.array(pred_y))
            i+=1
            #print('testing {}'.format(i))
        return pred_y_arr

    
def boosting_function(trainfv,testfv,maxDepth,minRows,numTrees):
    
    train , test = np.array(trainfv), np.array(testfv)
#    nTrain , nTest = train.shape[0], test.shape[0]
    mTrain , mTest = train.shape[1]-1, test.shape[1]-1
    Y_train = train[:,-1]
    X_train = train[:,:mTrain]
#    print X_train.shape
    Y_test = test[:,-1]
    X_test = test[:,:mTest]
    
    
    Y_train[Y_train==0] = -1.0
    Y_train[Y_train==1] = 1.0
    Y_test[Y_test==0] = -1.0
    Y_test[Y_test==1] = 1.0   
    
    Y_train = Y_train.reshape((X_train.shape[0],1))
    Y_test = Y_test.reshape((X_test.shape[0],1))
    
    boost_t= boosting(max_depth=maxDepth,min_size=minRows,num_trees=numTrees)
#    et_arr = boost_t.train_boost(X_train,Y_train)
    boost_t.train_boost(X_train,Y_train)
    
    pred_y_arr = np.array(boost_t.test_boost(X_test))

    y_test_predict = np.sum(pred_y_arr,axis=0)
    y_test_predict[y_test_predict>=0]=1
    y_test_predict[y_test_predict<0]=-1
    err = error(list(Y_test[:,0]),list(y_test_predict))
    return err
    
    
# Build DT
def getTree(trainfv, maxDepth=10, minRows=10):
    root = findNextSplit(trainfv)
    split(root, maxDepth, minRows, 1) 
    print root
    return root  


# Find the next best feature to split on
def findNextSplit(trainfv):
    feat = len(trainfv[0])-1;  # no. of features   
#    bestFeat = 0
    bestFeat = 99999
    bestBranch= [list(), list()]
#    minGini = 100
    minGini = 99999
    branches  = [list(),list()]
    for f in range(feat):        
        branches[0],branches[1] = splitOnFeature(trainfv,f)
#        gini = giniGain(branches, branchWts)
        gini = giniGain(branches)
        if(gini < minGini):
            minGini = gini
            bestFeat = f
            bestBranch = branches           
    return {'feature':bestFeat,'branches':bestBranch}
        

def splitOnFeature(trainfv, f):
    
    left, right = list(), list()
    for i in range(len(trainfv)):
        if trainfv[i][f] == 0:
            left.append(trainfv[i])
        else:
            right.append(trainfv[i])    
    return left, right
    
   
# Find Gini Gain on splitting on that feature
def giniGain(branches):
 
    left = branches[0]
    right = branches[1]

    lneg = abs(sum([row[-1] for row in left if row[-1]>=0]))
    lpos = sum([row[-1] for row in left if row[-1]<0])
    
    rneg = abs(sum([row[-1] for row in right if row[-1]>=0]))
    rpos = sum([row[-1] for row in right if row[-1]<0])   
    
    leftGini = 1.0
    if(lpos + lneg):        
        leftGini -= (lneg/(lneg+lpos))**2
        leftGini -= (lpos/(lneg+lpos))**2 
 
    rightGini = 1.0    
    if(rpos + rneg):
        rightGini -= (rneg/(rneg+rpos))**2
        rightGini -= (rpos/(rneg+rpos))**2 
    
    tot = float(lpos + lneg + rpos + rneg)
    totGini = leftGini*((lpos + lneg)/tot) + rightGini*((rpos + rneg)/tot);
#    print totGini
    return totGini


# Recursively grow tree
def split(root, maxDepth, minRows, currDepth):
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
    if(len(left) <= minRows):
        root['left'] = getLeafClass(left)
    else:
        root['left'] = findNextSplit(left)
        split(root['left'],  maxDepth, minRows, currDepth + 1)
    
    # Process right branch
    if(len(right) <= minRows):
        root['right'] = getLeafClass(right)
    else:
        root['right'] = findNextSplit(right)
        split(root['right'], maxDepth, minRows, currDepth + 1)
        
        
# Get the max count class
def getLeafClass(data):
    classLabels = [row[-1] for row in data]    
    leaf = (1.0 if sum(classLabels)>=0 else -1.0)
    print "leaf = ",leaf
    return leaf

  
    
def error(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)        
    correctPred = (actual == predicted).astype(int)
    acc = (float(sum(correctPred))/float(len(correctPred)))
    zeroOneLoss = 1-acc
    acc *= 100
    return zeroOneLoss


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
     
        