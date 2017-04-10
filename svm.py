# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:53:31 2017

@author: Sriranjitha
"""
import numpy as np

def svm(trainfv, testfv):
    
    n = len(trainfv)
    m = len(trainfv[0])-1
    n2 = len(testfv)
    m = len(testfv[0])-1
    
    trainfv = np.array(trainfv)
    testfv = np.array(testfv)
    
    ytrain = trainfv[:,m]
    xtrain = trainfv[:,:m] 
    y = ytrain
    x = xtrain
    
    ytest = testfv[:,m]
    xtest = testfv[:,:m] 
    
    for i in range(len(ytest)):
        if ytest[i]==0:
            ytest[i] = -1;      
            
    for i in range(len(y)):
        if y[i]== 0:
            y[i] = -1;
        
    bias = np.ones((n,1))
    x = np.append(bias,x,axis = 1)


    bias2 = np.ones((n2,1))    
    xtest = np.append(bias2,xtest,axis = 1)

    
    m = m+1
    
    s = (1,m)
    w = np.zeros(s)
    wold = np.zeros(s)
    tol = np.zeros(s)
    yp = np.zeros(n)
    yptest = np.zeros(n2)


    
    s = (n,m)
    diff = np.zeros(s)
    
    delj = np.zeros((1,m))
    N = n    
    lam = 0.01
    eta = 0.5 
    
    #SVM code
    for _ in range(100): #in 100 iterations
        
        yp = np.dot(x,w.transpose())       
        yp[yp>=0] = 1
        yp[yp<0] = -1

        ycor = np.zeros(n)

        for i in range(n):
            if (y[i]==yp[i]):
                ycor[i] = 0
            else:
                ycor[i] =  y[i];
          
        for j in range(m):
            diff[:,j] = np.multiply(ycor, x[:,j]);
            inter = np.multiply(lam,w[0,j])
            diff[:,j] = np.subtract(inter,diff[:,j])
            
        delj = np.sum(diff,axis = 0)/float(n);
   
        w = np.subtract(w,(eta*delj))

        tol = np.linalg.norm(w-wold);

        if(tol<pow(10,-6)):
            break;
            
        wold = w;
        
        
    #Testing against test set    
    yptest = np.dot(xtest,w.transpose())
    yptest[yptest >=0] = 1
    yptest[yptest <0] = -1
            

    correctPred = (yptest[:,0] == ytest).astype(np.int)
    
  
    acc = (float(sum(correctPred))/float(len(correctPred)))
    zeroOneLoss = 1-acc
    acc *= 100
    

    return zeroOneLoss
