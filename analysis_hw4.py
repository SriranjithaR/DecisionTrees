from preprocess import preprocess
from preprocess import getWordList
from preprocess import getLines
from preprocess import splitColumns
import numpy as np
from NBCClassifier import featureVector
from cvFunctions import getTrainData
from decisionTree import decisionTree
from randomForest import randomForest
from bagging import bagging
import math
#from plot import plot

class analysis:
   
    def __init__(self):
            
        # Defaults
        self.trainData = 'Data\yelp_data.csv'
        self.percentage = 100
        self.file = 'results.txt'

        f = open(self.file, "a+")
        f.write("\n RESULTS\n");
        f.close()
        


    def getPlots(self):
        
        # Get data as array
        train = getLines(self.trainData, 100)
        
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
        
        
        zoltempdt = [0 for xtemp in range(10)]
        zoltemprf = [0 for xtemp in range(10)]
        zoltempbag = [0 for xtemp in range(10)]
        
        n = len(train)
        #ratios = [0.01,0.03]
        #it = 2
        it = 10
        ratios = [0.025, 0.05, 0.125, 0.25]
        avgzoldt  = [0 for xtemp in range(len(ratios))]
        avgzolrf = [0 for xtemp in range(len(ratios))]
        avgzolbag = [0 for xtemp in range(len(ratios))]
        
        stddevzoldt  = [0 for xtemp in range(len(ratios))]
        stddevzolrf = [0 for xtemp in range(len(ratios))]
        stddevzolbag = [0 for xtemp in range(len(ratios))]
        
        stderrzoldt  = [0 for xtemp in range(len(ratios))]
        stderrzolrf = [0 for xtemp in range(len(ratios))]
        stderrzolbag = [0 for xtemp in range(len(ratios))]
        
                     
        testnew = []
        trainnew = []
        
        
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
                
                # Forming feature vector, calculating Conditional probabilities, applying bag
                trainfv, trainfv0, trainfv1  = featureVector(wordList[:1000], x_train, y_train)
                testfv, testfv0, testfv1 = featureVector(wordList[:1000], x_test, y_test)
               
            
                zoltempdt[i] = decisionTree(trainfv,testfv)  
#                zoltemprf[i] = randomForest(trainfv,testfv)   
#                zoltempbag[i] = bagging(trainfv,testfv)   
                
            avgzoldt[r]  = np.mean(zoltempdt)
            avgzolrf[r] = np.mean(zoltemprf)
            avgzolbag[r] = np.mean(zoltempbag)
              
            stddevzoldt[r]  = np.std(zoltempdt)
            stddevzolrf[r]  = np.std(zoltemprf)
            stddevzolbag[r]  = np.std(zoltempbag)
            
            stderrzoldt[r] = stddevzoldt[r]/math.sqrt(it)
            stderrzolrf[r] = stddevzolrf[r]/math.sqrt(it)
            stderrzolbag[r] = stddevzolbag[r]/math.sqrt(it)
        

        
        print avgzoldt
        print avgzolrf
        print avgzolbag
        
        print stddevzoldt
        print stddevzolrf
        print stddevzolbag

        print stderrzoldt
        print stderrzolbag
        print stderrzolrf   

        f = open(self.file,"a+");
        f.write("\n AVERAGE ZERO ONE LOSS")
        f.write("\n 1. Decision Tree")
        f.write(str(stderrzoldt))
        f.write("\n 2. Bagging")
        f.write(str(stderrzolbag))
        f.write("\n 3. Random forest")
        f.write(str(stderrzolrf))
        
        f.write("\n STANDARD DEVIATION ZERO ONE LOSS")
        f.write("\n 1. Decision Tree")
        f.write(str(stddevzoldt))     
        f.write("\n 2. Bagging")
        f.write(str(stddevzolbag))        
        f.write("\n 3. Random forest")
        f.write(str(stddevzolrf))
        
        f.write("\n STANDARD ERROR ZERO ONE LOSS")
        f.write("\n 1. Decision Tree")
        f.write(str(stderrzoldt))       
        f.write("\n 2. Bagging")
        f.write(str(stderrzolbag))        
        f.write("\n 3. Random forest")
        f.write(str(stderrzolrf))
        f.close();
     