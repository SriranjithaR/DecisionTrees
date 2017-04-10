# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:09:56 2017

@author: Sriranjitha
"""
from decisionTree import getTree, prediction

class AdaBoost:
    
    def boosting(trainfv, testfv, maxDepth=10, minRows=10, numTrees=50):
    
        n = len(trainfv)
        
        D = [float(1/n) for i in range(n)]
             
        a = []
        
        for k in range(numTrees):
            
            hk = getTree(trainfv, testfv, maxDepth=10, minRows=10)
            
            wTrain = [[elt*D[i] for elt in trainfv[i]] for i in range(n)]
            preds = [prediction(hk, row) for row in wTrain]
            
            