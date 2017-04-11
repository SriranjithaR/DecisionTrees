from preprocess import splitData
from preprocess import preprocess
from preprocess import getWordList
from preprocess import getLines
from preprocess import splitColumns
import numpy as np
from NBCClassifier import featureVector
from cvFunctions import getTrainData
#from lr import lr
from svm import svm
#from nbc import nbc
import math
#from plot import plot

# Defaults
trainData = 'Data\yelp_data.csv'
percentage = 100

# Get data as array
train = getLines(trainData, 100)





"""
 words is now a list of (word,frequency) tuples, ordered by descending order
 of frequency
 
 wordList is a list of all unique words in the training data
"""   


"""
 The 100 most frequent words have been removed from words
 
 wordList is now a list of the words in desc order of frequency
"""

cv = [] 
#cv = [[[0 for z in range(4001)] for y in range(200)] for x in range(10)]
#cv = np.array(cv);
#trainfv = np.array(trainfv)

#np.random.shuffle(trainfv)

train = np.array(train)
np.random.shuffle(train)

for i in range(10):
    cv.append(train[i*200:(i+1)*200])


zoltemplr = [0 for xtemp in range(10)]
zoltempsvm = [0 for xtemp in range(10)]
zoltempnbc = [0 for xtemp in range(10)]

n = len(train)
#ratios = [0.01,0.03]
#it = 2
it = 10
#ratios = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]
ratios = [0.25]
avgzollr  = [0 for xtemp in range(len(ratios))]
avgzolsvm = [0 for xtemp in range(len(ratios))]
avgzolnbc = [0 for xtemp in range(len(ratios))]

stddevzollr  = [0 for xtemp in range(len(ratios))]
stddevzolsvm = [0 for xtemp in range(len(ratios))]
stddevzolnbc = [0 for xtemp in range(len(ratios))]

stderrzollr  = [0 for xtemp in range(len(ratios))]
stderrzolsvm = [0 for xtemp in range(len(ratios))]
stderrzolnbc = [0 for xtemp in range(len(ratios))]

             
testnew = []
trainnew = []

w = 200
#print "Starting "
for r in range(len(ratios)):
    print "ration : ",r
    
    for i in range(it):
        # print "CV step ",i
        trainnew = []
        testnew = cv[i]
        # print "test data shape : ",testfvnew.shape

        for j in range(it):
            if j != i:
                for k in range(200):
                    trainnew.append(cv[j][k])
#                    print len(trainnew), "i : ",i,"j : ",j

        # print "Trainfvnew shape : ", trainfvnew.shape
        temptrain = trainnew
#        temptrain = np.array(temptrain)
#        print temptrain.shape
        trainDataset  = getTrainData(temptrain, ratios[r])
        # print "Train data shape : ",trainDataset.shape
        
        
        rid_train,x_train,y_train = splitColumns(trainDataset);
        rid_test,x_test,y_test = splitColumns(testnew);
        
        # Pre-processing data
        x_train = preprocess(x_train)
        x_test = preprocess(x_test)
        
        # Creating dictionary from x_train
        words,wordList = getWordList(x_train)
        
        # Removing most frequent 100 words
        for _ in range(100):
            words.pop(0)
            
        wordList = [x for x,_ in words]
        
        # Forming feature vector, calculating Conditional probabilities, applying NBC
        trainfv, trainfv0, trainfv1  = featureVector(wordList[:w], x_train, y_train)
        testfv, testfv0, testfv1 = featureVector(wordList[:w], x_test, y_test)
       
    
#        zoltemplr[i]  = lr(trainfv,testfv)  
        zoltempsvm[i] = svm(trainfv,testfv)   
#        zoltempnbc[i] = nbc(trainfv,testfv)   
        
    avgzollr[r]  = np.mean(zoltemplr)
    avgzolsvm[r] = np.mean(zoltempsvm)
    avgzolnbc[r] = np.mean(zoltempnbc)
      
    stddevzollr[r]  = np.std(zoltemplr)
    stddevzolsvm[r]  = np.std(zoltempsvm)
    stddevzolnbc[r]  = np.std(zoltempnbc)
    
    stderrzollr[r] = stddevzollr[r]/math.sqrt(it)
    stderrzolsvm[r] = stddevzolsvm[r]/math.sqrt(it)
    stderrzolnbc[r] = stddevzolnbc[r]/math.sqrt(it)

print stderrzollr
print stderrzolsvm
print stderrzolnbc

print avgzollr
print avgzolsvm
print avgzolnbc

print stddevzollr
print stddevzolsvm
print stddevzolnbc
