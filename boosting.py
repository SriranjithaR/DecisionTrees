# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:09:56 2017

@author: Sriranjitha
"""
import math
import numpy as np
from decimal import Decimal

class AdaBoost:
    
      
    def delta(self,a,b):
        if(a!=b):
            return 1
        else:
            return 0;
            
    def update(self,Dk, ak, preds, y):
        n = len(Dk)
        
        exparr = list()
        for i in range(n):
            exparr.append(math.exp(-1 * ak * preds[i] ))
#            exparr.append(math.exp(-1 * ak * y[i] * preds[i] ))
            
        exparr = np.array(exparr)
            
        Dk = np.array(Dk)
        Dk = np.multiply(Dk,exparr)
        
        Dk = Dk/np.sum(Dk)
        return Dk
        
    def trainBoosting(self,trainfv, testfv, maxDepth=10, minRows=10, numTrees=50):
                
        n = len(trainfv)    
        
        Dk = [(1.0/float(n)) for i in range(n)]            

        y = [row[-1] for row in trainfv]
        preds = [[0 for i in range(n)] for k in range(numTrees)] 

        h = [0 for x in range(numTrees)]
        ak =  [0 for x in range(numTrees)]
               
        for k in range(numTrees):
            print "Tree number : ",k
                                 
            h[k] = self.getTree(trainfv, testfv, Dk, maxDepth=10, minRows=10)                                     
            preds[k] = [self.prediction(h[k], row) for row in trainfv]
            print preds[k]
            
            ek = sum([Dk[i] * self.delta(preds[k][i],y[i]) for i in range(n)])
            

            if ek==0:
                break;
                
            print "ek = ", ek
            ak[k] = 0.5 * float((Decimal((1.0-float(ek))/float(ek)).ln()))
            Dk = self.update(Dk, ak[k], preds[k], y)
            print Dk
      
        return h,ak

        
    def boosting(self,trainfv, testfv, maxDepth=10, minRows=10, numTrees=50):
                
        train , test = np.array(trainfv), np.array(testfv)
        nTrain , nTest = train.shape[0], test.shape[0]
        mTrain , mTest = train.shape[1]-1, test.shape[1]-1
        yTrain = train[:,-1]
        xTrain = train[:,:mTrain]
        yTest = test[:,-1]
        xTest = test[:,:mTest]
        
        predTrain , predTest = [np.zeros(nTrain), np.zeros(nTest)]
        yTrain[yTrain == 0] = -1
        yTest[yTest == 0] = -1
             
        D = np.ones(nTrain)/nTrain
                   
        for k in range(numTrees):
            
            print "Tree number : ",k
            hk = self.getTree(train, test, D, maxDepth, minRows )
            predTrain_k = [self.prediction(hk,row) for row in train]
            predTest_k = [self.prediction(hk,row) for row in test]
            
            delta = [int(x) for x in predTrain_k != yTrain]
            
            mis = [x if x==1 else -1 for x in delta]
            
            e_k = np.dot(D, delta)/ sum(D)
            
            if e_k == 0:
                break;
                
#            a_k = 0.5 * np.log((1.0-e_k)/float(e_k))
            a_k = 0.5 * float((Decimal((1.0-float(e_k))/float(e_k)).ln()))
            
            D = np.multiply(D, np.exp([float(x)*a_k for x in mis]))
            
            predTrain = [sum(x) for x in zip(predTrain,[ x*a_k for x in predTrain_k])]
            predTest = [sum(x) for x in zip(predTest, [x*a_k for x in predTest_k])]
                        
        predTrain, predTest = np.sign(predTrain), np.sign(predTest)     
            
        correctPred = (predTrain == yTrain).astype(int)
#        for row in trainfv:
#            if(row[-1]==0):
#                row[-1] = -1;
#                
#        for row in testfv:
#            if(row[-1]==0):
#                row[-1] = -1;
#                    
#        n = len(testfv)
#        h, ak = self.trainBoosting(trainfv, testfv, maxDepth=10, minRows=10, numTrees=50)
#        preds = [[0 for i in range(n)] for k in range(numTrees)]
#                  
#        print len(preds), len(preds[0])
#        print len(h)
#        print len(ak)
#        
#        print preds
#        for k in range(numTrees):
#            if isinstance(h[k],dict):
#                for i in range(n):
#                    preds[k][i] = self.prediction(h[k],testfv[i])*ak[k]
#            
##        print preds
#        predsnp = np.array(preds)
#        predVals = np.sum(predsnp, 0)
#                    
#        predVals[predVals >= 0] = 1
#        predVals[predVals < 0] = -1
#            
        # Compare predicted values with actual values 
#        test = np.array(testfv)
#        actual = test[:,-1]
#        print actual.tolist()
#        print predVals.tolist()
#        
#        correctPred = (predVals == actual).astype(int)
          
        acc = (float(sum(correctPred))/float(len(correctPred)))
        zeroOneLoss = 1-acc
        acc *= 100
        return zeroOneLoss
        
            
    # Split data on given feature index
    def splitOnFeature(self, trainfv, f, Dk):
        
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
    def prediction(self, root, sample):
#        print type(root)
        if(sample[root['feature']] == 0):        
            if isinstance(root['left'],dict):
                return self.prediction(root['left'],sample)
            else:
                return root['left']           
        else:       
            if isinstance(root['right'],dict):
                return self.prediction(root['right'],sample)
            else:
                return root['right']
         
              
    # Build DT
    def getTree(self, trainfv, testfv, Dk, maxDepth=10, minRows=10):
        root = self.findNextSplit(trainfv, Dk)
        self.split(root, Dk, maxDepth, minRows, 1) 
        return root
        
        
        
    # Find the next best feature to split on
    def findNextSplit(self, trainfv, Dk):
        feat = len(trainfv[0])-1;  # no. of features   
        bestFeat = 0
        bestBranch= [list(), list()]
        minGini = 100
        branches  = [list(),list()]
        branchWts = [list(),list()]
        for f in range(feat):        
            branches[0],branches[1],branchWts[0],branchWts[1] = self.splitOnFeature(trainfv,f, Dk)
            gini = self.giniGain(branches, branchWts)
            if(gini < minGini):
                minGini = gini
                bestFeat = f
                bestBranch = branches           
        return {'feature':bestFeat,'branches':bestBranch}
            
        
    # Get the max count class
    def getLeafClass(self, data):
        classLabels = [row[-1] for row in data]
        return max(set(classLabels),key=classLabels.count)
      
        
    # Recursively grow tree
    def split(self, root, Dk, maxDepth, minRows, currDepth):
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
            root['left'] = root['right'] = self.getLeafClass(right)
            return
        elif not len(right):
            root['left'] = root['right'] = self.getLeafClass(left)
            return
            
        # Check for max depth
        if(currDepth >= maxDepth):
            root['left'], root['right'] = self.getLeafClass(left), self.getLeafClass(right)
            return
            
        # Process left branch
        if(len(left) <= minRows):
            root['left'] = self.getLeafClass(left)
        else:
            root['left'] = self.findNextSplit(left, Dk)
            self.split(root['left'], Dk, maxDepth, minRows, currDepth + 1)
        
        # Process right branch
        if(len(right) <= minRows):
            root['right'] = self.getLeafClass(right)
        else:
            root['right'] = self.findNextSplit(right, Dk)
            self.split(root['right'], Dk, maxDepth, minRows, currDepth + 1)
            
                
    # Find Gini Gain on splitting on that feature
    def giniGain(self, branches, Dk):
 
        left = branches[0]
        right = branches[1]
        leftWts = Dk[0]
        rightWts = Dk[1]
#        print "lw = ",leftWts
        leftGini = 0.0
        rightGini = 0.0
        gini = 0.0
        
#        p = len(left[0])-1
        
#        print "len(left) = ",len(left)
#        print "len(left[0]) = ",len(left[0])
        leftSize = len(left)
        rightSize = len(right)
        
        lprop1 = list()
        lprop0 = list()
        for i in range(leftSize):
            if(left[i][-1]==1):
                lprop1.append(leftWts[i])
            else:
                lprop0.append(leftWts[i])
        
        rprop1 = list()
        rprop0 = list()
        for i in range(rightSize):
            if(right[i][-1]==1):
                rprop1.append(rightWts[i])
            else:
                rprop0.append(rightWts[i])
        
        gini = 0.0

        if(len(lprop1) or len(lprop0)):        
            gini += np.square(sum(lprop0)/(sum(lprop0)+sum(lprop1)))
            gini += np.square(sum(lprop1)/(sum(lprop0)+sum(lprop1))) 
        
        leftGini = 1.0 - gini
        
        gini = 0.0
        
        if(len(rprop1) or len(rprop0)):
            gini += np.square(sum(rprop0)/(sum(rprop0)+sum(rprop1)))
            gini += np.square(sum(rprop1)/(sum(rprop0)+sum(rprop1))) 
        
        rightGini = 1.0 - gini
        
#        for y in [-1,1]:
#            size = leftSize
#            if size == 0:
#                continue
#
#            print "left[i][p] = ",left[0][p]
#            
#            proportion = sum([leftWts[i] for i in range(leftSize) if left[i][p]==y])/float(size)
##            proportion = [row[-1] for row in left].count(y) / float(size)
#            gini += proportion*proportion
#        leftGini = 1.0 - gini
        
#        gini = 0.0
#        
#        for y in [-1,1]:
#            size = rightSize
#            if size == 0:
#                continue
#                        
#            proportion = sum([rightWts[i] for i in range(rightSize) if right[i]==y])/float(size)
##            proportion = [row[-1] for row in right].count(y) / float(size)
#            gini += proportion*proportion
#        rightGini = 1.0 - gini
#        
        
#        totSize = leftSize + rightSize
        
        totGini = leftGini*((sum(lprop0)+sum(lprop1))/(sum(rprop0)+sum(rprop1)+sum(lprop1)+sum(lprop0))) + rightGini*((sum(rprop0)+sum(rprop1))/(sum(rprop0)+sum(rprop1)+sum(lprop1)+sum(lprop0)));
        return totGini