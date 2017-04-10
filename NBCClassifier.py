# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:57:34 2017

@author: Sriranjitha
"""
import numpy as np

def featureVector(words,x_train,y_train):
    n = len(y_train);
    p = len(words);
#    print "No. of words : ",p
    c = p;

    # Creating feature vector    
    
    fv = [[0 for y in range(p+1)] for x in range(n)]
           
    fv0 = [[0 for y in range(p+1)] ]
    fv1 = [[0 for y in range(p+1)] ]
    
    for i in range(n):
        for j in range(p):
            if(words[j] in x_train[i]):
                fv[i][j] = 1;
            else:
                fv[i][j]= 0;
        fv[i][c] = y_train[i]
        if(fv[i][c] == '1'):
            fv[i][c] = 1;
            fv1 = np.vstack([fv1,fv[i]]);
        else:
            fv[i][c] = 0;
            fv0 = np.vstack([fv0,fv[i]]);
             
    fv0 = np.delete(fv0, (0), axis=0)
    fv1 = np.delete(fv1, (0), axis=0)
          
    return fv,fv0,fv1

def ternaryFeatureVector(words,x_train,y_train):
    n = len(y_train);
    p = len(words);
    c = p;

    # Creating feature vector    
    
    fv = [[0 for y in range(p+1)] for x in range(n)]
           
    fv0 = [[0 for y in range(p+1)] ]
    fv1 = [[0 for y in range(p+1)] ]
    
    for i in range(n):
        for j in range(p):
            ct = x_train[i].count(words[j]) 
            if(ct >=2):
                fv[i][j] = 2;
            else:
                fv[i][j]= ct;
                
        fv[i][c] = y_train[i]
        if(fv[i][c] == '1'):
            fv[i][c] = 1;
            fv1 = np.vstack([fv1,fv[i]]);
        else:
            fv[i][c] = 0;
            fv0 = np.vstack([fv0,fv[i]]);

             
    fv0 = np.delete(fv0, (0), axis=0)
    fv1 = np.delete(fv1, (0), axis=0)
          
    return fv,fv0,fv1
    
def calcProb(fv,fv0,fv1):

    p = len(fv0[0])
    
    fv0sum = np.sum(fv0,axis=0)
    fv1sum = np.sum(fv1,axis=0)
   

    
    # Calculating probabilities       
    
    Nk = [0 for x in range(2)]
    Nkw = [[0 for y in range(p+1)] for x in range(2)]
    
    Pkw = [[0 for y in range(p+1)] for x in range(2)]
    Pk = [0 for x in range(2)]
            
    N = len(fv)
    Nk[0] = len(fv0)
    Nk[1] = len(fv1)
    
    Nkw[0] = fv0sum;
    Nkw[1] = fv1sum;
    
    
    Nkwfloat0 = (np.array(Nkw[0])).astype(float)
    Nkwfloat1 = (np.array(Nkw[1])).astype(float)
    
    # Performing Laplace correction
    
    Nkwfloat0 = np.add(Nkwfloat0,1.0)
    Nkwfloat1 = np.add(Nkwfloat1,1.0)  
    
    Pkw[0] = np.divide(Nkwfloat0,float(Nk[0])+2.0)
    Pkw[1] = np.divide(Nkwfloat1,float(Nk[1])+2.0)
    
    
    Pk[0] = Nk[0]/float(N)
    Pk[1] = Nk[1]/float(N)
    
    return Nk,Nkw,Pkw,Pk
    
def calcProbTernary(fv,fv0,fv1):

    p = len(fv0[0])
    
    fv0sum = np.sum(fv0,axis=0)
    fv1sum = np.sum(fv1,axis=0)
   
    
    two0 = [0 for x in range(p)]
    one0 = [0 for x in range(p)]
    zero0 = [0 for x in range(p)]
             
    two1 = [0 for x in range(p)]
    one1 = [0 for x in range(p)]
    zero1 = [0 for x in range(p)]
            
    for j in range(p):
        two0[j] = np.count_nonzero(fv0[:,j] == 2)
        one0[j] = np.count_nonzero(fv0[:,j] == 1)
        zero0[j] = np.subtract(len(fv0),(two0[j]+one0[j]))
        
    for j in range(p):
        two1[j] = np.count_nonzero(fv1[:,j] == 2)
        one1[j] = np.count_nonzero(fv1[:,j] == 1)
        zero1[j] = np.subtract(len(fv1),(two1[j]+one1[j]))
    
    
    # Calculating probabilities       
    
    Nk = [0 for x in range(2)]
   
    Nkw0 = [[0 for y in range(p+1)] for x in range(2)]          
    Nkw1 = [[0 for y in range(p+1)] for x in range(2)]          
    Nkw2 = [[0 for y in range(p+1)] for x in range(2)]
             
    
    Pkw0 = [[0 for y in range(p+1)] for x in range(2)]
    Pkw1 = [[0 for y in range(p+1)] for x in range(2)]
    Pkw2 = [[0 for y in range(p+1)] for x in range(2)]
    Pk = [0 for x in range(2)]
            
    N = len(fv)
    Nk[0] = len(fv0)
    Nk[1] = len(fv1)

    Nkw0[0] = zero0;
    Nkw0[1] = zero1;
          
    Nkw1[0] = one0;
    Nkw1[1] = one1;
    
    Nkw2[0] = two0;
    Nkw2[1] = two1;
    
#    Nkw[0] = fv0sum;
#    Nkw[1] = fv1sum;
    
    
    Nkwfloat0 = (np.array(Nkw0)).astype(float)
    Nkwfloat1 = (np.array(Nkw1)).astype(float)
    Nkwfloat2 = (np.array(Nkw2)).astype(float)
    
    # Performing Laplace correction
    
    Nkwfloat0 = np.add(Nkwfloat0,1.0)
    Nkwfloat1 = np.add(Nkwfloat1,1.0)
    Nkwfloat2 = np.add(Nkwfloat2,1.0)   
    
    Pkw0[0] = np.divide(Nkwfloat0[0],float(Nk[0])+3.0)
    Pkw0[1] = np.divide(Nkwfloat0[1],float(Nk[1])+3.0)
    Pkw1[0] = np.divide(Nkwfloat0[0],float(Nk[0])+3.0)
    Pkw1[1] = np.divide(Nkwfloat0[1],float(Nk[1])+3.0)
    Pkw2[0] = np.divide(Nkwfloat0[0],float(Nk[0])+3.0)
    Pkw2[1] = np.divide(Nkwfloat0[1],float(Nk[1])+3.0)

    
    
    Pk[0] = Nk[0]/float(N)
    Pk[1] = Nk[1]/float(N)
    
    return Nk,Nkwfloat0,Nkwfloat1,Nkwfloat2,Pkw0,Pkw1, Pkw2,Pk
        
def testNbc(Nk,Nkw,Pkw,Pk,trainfv, trainfv0, trainfv1, testfv, testfv0, testfv1):
    

    nt = len(testfv); # no. of test samples
            
    p = len(testfv[0]) -1 # no. of words

    
    testfvTranspose = np.array(testfv).transpose()
    
   
    # Computing probabilities
    predVals = [[1 for y in range(nt)] for x in range(2)]
    for j in range(nt):
        for i in range(p):
            predVals[0][j] *= ( (float(Pkw[0][i])*float(testfvTranspose[i][j])) +  ( (1-float(Pkw[0][i])) * (1-float(testfvTranspose[i][j])) ) )
        predVals[0][j] *= float(Pk[0])
        
        for i in range(p):
            predVals[1][j] *= ( (float(Pkw[1][i])*float(testfvTranspose[i][j])) +  ( (1-float(Pkw[1][i])) * (1-float(testfvTranspose[i][j])) ) )
        predVals[1][j] *= float(Pk[1])
            

    predLabels = np.argmax(predVals, axis=0)
   
    testfvMat = np.array(testfv) 
  
    actLabels = testfvMat[:,p]  
  
    correctPred = (predLabels == actLabels).astype(int)
  
    acc = float(sum(correctPred))/float(len(correctPred))
    zeroOneLoss = 1-acc
    acc *= 100


    # Calculating baseline error
    if len(trainfv1)>len(trainfv0):
        baselineLabels = [1 for x in range(len(testfv))]
    else:
        baselineLabels = [0 for x in range(len(testfv))]
                         
       
    baselineCorrect = (baselineLabels == actLabels).astype(int)
    baselineError = 1 - (float(sum(baselineCorrect))/float(len(baselineCorrect)))
                   
                         
    return zeroOneLoss,baselineError
    
   
def testNbcTernary(Nk, Nkwfloat0, Nkwfloat1, Nkwfloat2, Pkw0, Pkw1, Pkw2, Pk, trainfv, trainfv0, trainfv1, testfv, testfv0, testfv1):
    

    nt = len(testfv); # no. of test samples
            
    p = len(testfv[0]) -1 # no. of words

    
    testfvTranspose = np.array(testfv).transpose()
    
   
    # Computing probabilities
    predVals = [[1 for y in range(nt)] for x in range(2)]
    for j in range(nt):
        for i in range(p):
            if(testfvTranspose[i][j] == 2):
                predVals[0][j] *= Pkw2[0][i]
            elif(testfvTranspose[i][j] == 1):
                predVals[0][j] *= Pkw1[0][i]
            else:
                predVals[0][j] *= Pkw0[0][i]
        predVals[0][j] *= float(Pk[0])
        
        for i in range(p):
            if(testfvTranspose[i][j] == 2):
                predVals[1][j] *= Pkw2[1][i]
            elif(testfvTranspose[i][j] == 1):
                predVals[1][j] *= Pkw1[1][i]
            else:
                predVals[1][j] *= Pkw0[1][i]
        predVals[1][j] *= float(Pk[1])
            

    predLabels = np.argmax(predVals, axis=0)
   
    testfvMat = np.array(testfv) 
  
    actLabels = testfvMat[:,p]  
  
    correctPred = (predLabels == actLabels).astype(int)
  
    acc = float(sum(correctPred))/float(len(correctPred))
    zeroOneLoss = 1-acc
    acc *= 100


    # Calculating baseline error
    if len(trainfv1)>len(trainfv0):
        baselineLabels = [1 for x in range(len(testfv))]
    else:
        baselineLabels = [0 for x in range(len(testfv))]
                         
       
    baselineCorrect = (baselineLabels == actLabels).astype(int)
    baselineError = 1 - (float(sum(baselineCorrect))/float(len(baselineCorrect)))
                   
                         
    return zeroOneLoss,baselineError
    
            
            
            
            
            
            
            
            