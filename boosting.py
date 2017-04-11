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
        
    
        for i in range(n):
            Dk[i] = (Dk[i]*math.exp(-1 * ak * y[i] * preds[i] ))
            
        Zk = sum(Dk)
        Dk = [float(row)/float(Zk) for row in Dk]
        return Dk
        
    def trainBoosting(self,trainfv, testfv, maxDepth=10, minRows=10, numTrees=50):
                
        n = len(trainfv)    
        m = len(trainfv[0])-1
        
        Dk = [(1.0/float(n)) for i in range(n)]            

        y = [row[-1] for row in trainfv]
        preds = [[0 for i in range(n)] for k in range(numTrees)] 
        wTrain = trainfv
        h = [0 for x in range(numTrees)]
        ak =  [0 for x in range(numTrees)]
               
        for k in range(numTrees):
            print "Tree number : ",k
           
#            print "wTrain : ", wTrain
#            temptrain = np.array(trainfv);
#            np.delete(temptrain,m,1)
#            temptrain = temptrain.tolist()
#            wTrain = [[elt * Dk[i] for elt in temptrain[i]] for i in range(n)]
            for i in range(n):
                for j in range(m):
                    wTrain[i][j] = Dk[i]*trainfv[i][j]
                       
            print "wTrainf after applying wts : ", wTrain
            h[k] = self.getTree(wTrain, testfv, maxDepth=10, minRows=10)   
#            
#            print "len(wTrain) : ", len(wTrain), "len(wTrain[0]) : ", len(wTrain[0])          
                      
            preds[k] = [self.prediction(h[k], row) for row in wTrain]
            print preds[k]
            print y
            
            print "k = ", k
            for i in range(n):
                if preds[k][i]>=0:
                    preds[k][i] = 1
                else:
                    preds[k][i] = -1;
            
            ek = sum([Dk[i] * self.delta(preds[k][i],y[i]) for i in range(n)])
            

            if ek==0:
                break;
                
            print "ek = ", ek
            ak[k] = 0.5 * float((Decimal((1.0-float(ek))/float(ek)).ln()))
            Dk = self.update(Dk, ak[k], preds[k], y)
            print Dk
      
        return h,ak
#        return hk
        
    def boosting(self,trainfv, testfv, maxDepth=10, minRows=10, numTrees=50):
                
#        for row in trainfv:
#            if(row[-1]==0):
#                row[-1] = -1;
#                
#        for row in testfv:
#            if(row[-1]==0):
#                row[-1] = -1;

        for i in range(len(trainfv)):
            for j in range(len(trainfv[0])):
                if(trainfv[i][j]==0):
                    trainfv[i][j] = -1
                    
        for i in range(len(testfv)):
            for j in range(len(testfv[0])):
                if(testfv[i][j]==0):
                    testfv[i][j] = -1
                    
        n = len(testfv)
        h, ak = self.trainBoosting(trainfv, testfv, maxDepth=10, minRows=10, numTrees=50)
        preds = [[0 for i in range(n)] for k in range(numTrees)]
                  
        print len(preds), len(preds[0])
        print len(h)
        print len(ak)
        
        for k in range(numTrees):
            if isinstance(h[k],dict):
                for i in range(n):
                    preds[k][i] = self.prediction(h[k],testfv[i]) * ak[k]
            
        predsnp = np.array(preds)
        predVals = np.sum(predsnp, 0)
        predVals[predVals >= 0] = 1
        predVals[predVals < 0] = 0
            
        # Compare predicted values with actual values 
        test = np.array(testfv)
        actual = test[:,-1]
        predVals = np.array(predVals)
        
        correctPred = (predVals == actual).astype(int)
          
        acc = (float(sum(correctPred))/float(len(correctPred)))
        zeroOneLoss = 1-acc
        acc *= 100
        return zeroOneLoss
        
            
    # Split data on given feature index
    def splitOnFeature(self, trainfv, f):
        
        left, right = list(), list()
        for row in trainfv:
            if row[f] < 0:
                left.append(row)
            else:
                right.append(row)
        
        return left, right
        
    
    # Parse tree and make prediction on sample
    def prediction(self, root, sample):
        if(sample[root['feature']] < 0):        
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
    def getTree(self, trainfv, testfv, maxDepth=10, minRows=10):
        root = self.findNextSplit(trainfv)
        self.split(root, maxDepth, minRows, 1) 
        return root
        
        
        
    # Find the next best feature to split on
    def findNextSplit(self, trainfv):
        feat = len(trainfv[0])-1;  # no. of features   
        bestFeat = 0
        bestBranch= [list(), list()]
        minGini = 100
        for f in range(feat):        
            branches = self.splitOnFeature(trainfv,f)
            gini = self.giniGain(branches)
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
    def split(self, root, maxDepth, minRows, currDepth):
        """ 
        root['feature'] => which feature to was used to create root['branches']
        for the next recursive call, this feature has to be removed from the branches
        before calculating next best feature 
        """
        
        left, right = root['branches']
        del(root['branches'])
        
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
            root['left'] = self.findNextSplit(left)
            self.split(root['left'], maxDepth, minRows, currDepth + 1)
        
        # Process right branch
        if(len(right) <= minRows):
            root['right'] = self.getLeafClass(right)
        else:
            root['right'] = self.findNextSplit(right)
            self.split(root['right'], maxDepth, minRows, currDepth + 1)
            
                
    # Find Gini Gain on splitting on that feature
    def giniGain(self, branches):
 
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
            
            proportion = [row[-1] for row in left].count(y) / float(size)
            gini += proportion*proportion
        leftGini = 1.0 - gini
        
        gini = 0.0
        
        for y in [0,1]:
            size = rightSize
            if size == 0:
                continue
            
            proportion = [row[-1] for row in right].count(y) / float(size)
            gini += proportion*proportion
        rightGini = 1.0 - gini
        
        
        totSize = leftSize + rightSize
        
        totGini = leftGini*(leftSize/totSize) + rightGini*(rightSize/totSize);
        return totGini