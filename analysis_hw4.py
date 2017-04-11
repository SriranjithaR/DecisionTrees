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
from svm import svm
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
        

     
    def changeRatios(self, ratios, words):
        
        # Get data as array
        train = getLines(self.trainData, 100)
        
        cv = []         
        train = np.array(train)
        np.random.shuffle(train)
        
        for i in range(10):
            cv.append(train[i*200:(i+1)*200])
        
        
        zoltempdt = [0 for xtemp in range(10)]
        zoltemprf = [0 for xtemp in range(10)]
        zoltempbag = [0 for xtemp in range(10)]
        zoltempsvm = [0 for xtemp in range(10)]
        
        it = 10
        
        avgzoldt  = [0 for xtemp in range(len(ratios))]
        avgzolrf = [0 for xtemp in range(len(ratios))]
        avgzolbag = [0 for xtemp in range(len(ratios))]
        avgzolsvm = [0 for xtemp in range(len(ratios))]
        
        stddevzoldt  = [0 for xtemp in range(len(ratios))]
        stddevzolrf = [0 for xtemp in range(len(ratios))]
        stddevzolbag = [0 for xtemp in range(len(ratios))]
        stddevzolsvm = [0 for xtemp in range(len(ratios))]
        
        stderrzoldt  = [0 for xtemp in range(len(ratios))]
        stderrzolrf = [0 for xtemp in range(len(ratios))]
        stderrzolbag = [0 for xtemp in range(len(ratios))]
        stderrzolsvm = [0 for xtemp in range(len(ratios))]
        
                     
        testnew = []
        trainnew = []
        

        for w in words:
            for r in range(len(ratios)):
                print "ration : ",r
                
                for i in range(it):
                    trainnew = []
                    testnew = cv[i]
            
                    for j in range(it):
                        if j != i:
                            for k in range(200):
                                trainnew.append(cv[j][k])

                    temptrain = trainnew
                    trainDataset  = getTrainData(temptrain, ratios[r])
       
                    
                    
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
                    trainfv, trainfv0, trainfv1  = featureVector(wordList[:w], x_train, y_train)
                    testfv, testfv0, testfv1 = featureVector(wordList[:w], x_test, y_test)
                   
                
#                    zoltempdt[i] = decisionTree(trainfv,testfv)  
#                    zoltemprf[i] = randomForest(trainfv,testfv)   
#                    zoltempbag[i] = bagging(trainfv,testfv)      
                    zoltempsvm[i] = svm(trainfv,testfv)   
                    
                avgzoldt[r]  = np.mean(zoltempdt)
                avgzolrf[r] = np.mean(zoltemprf)
                avgzolbag[r] = np.mean(zoltempbag)
                avgzolsvm[r] = np.mean(zoltempsvm)
                  
                stddevzoldt[r]  = np.std(zoltempdt)
                stddevzolrf[r]  = np.std(zoltemprf)
                stddevzolbag[r]  = np.std(zoltempbag)
                stddevzolsvm[r]  = np.std(zoltempsvm)
                
                stderrzoldt[r] = stddevzoldt[r]/math.sqrt(it)
                stderrzolrf[r] = stddevzolrf[r]/math.sqrt(it)
                stderrzolbag[r] = stddevzolbag[r]/math.sqrt(it)
                stderrzolsvm[r] = stddevzolsvm[r]/math.sqrt(it)
            

        
            print avgzoldt
            print avgzolrf
            print avgzolbag
            print avgzolsvm
            
            print stddevzoldt
            print stddevzolrf
            print stddevzolbag
            print stddevzolsvm
    
            print stderrzoldt
            print stderrzolrf
            print stderrzolbag 
            print stderrzolsvm   
    
            f = open(self.file,"a+");
            f.write("\n No. of words : ")
            f.write(str(w))
            f.write("\n AVERAGE ZERO ONE LOSS")
            f.write("\n 1. Decision Tree")
            f.write(str(avgzoldt))
            f.write("\n 2. Bagging")
            f.write(str(avgzolbag))
            f.write("\n 3. Random forest")
            f.write(str(avgzolrf))
            f.write("\n 4. SVM")
            f.write(str(avgzolsvm))
            
            f.write("\n STANDARD DEVIATION ZERO ONE LOSS")
            f.write("\n 1. Decision Tree")
            f.write(str(stddevzoldt))     
            f.write("\n 2. Bagging")
            f.write(str(stddevzolbag))        
            f.write("\n 3. Random forest")
            f.write(str(stddevzolrf))        
            f.write("\n 4. SVM")
            f.write(str(stddevzolsvm))
            
            f.write("\n STANDARD ERROR ZERO ONE LOSS")
            f.write("\n 1. Decision Tree")
            f.write(str(stderrzoldt))       
            f.write("\n 2. Bagging")
            f.write(str(stderrzolbag))        
            f.write("\n 3. Random forest")
            f.write(str(stderrzolrf))       
            f.write("\n 4. SVM")
            f.write(str(stderrzolsvm))
            f.close();
    
   
    def changeWords(self, ratios, numWords):
        
        # Get data as array
        train = getLines(self.trainData, 100)
        
        cv = []         
        train = np.array(train)
        np.random.shuffle(train)
        
        for i in range(10):
            cv.append(train[i*200:(i+1)*200])
        
        
        zoltempdt = [0 for xtemp in range(10)]
        zoltemprf = [0 for xtemp in range(10)]
        zoltempbag = [0 for xtemp in range(10)]
        zoltempsvm = [0 for xtemp in range(10)]
        
        it = 10
        
        avgzoldt  = [0 for xtemp in range(len(numWords))]
        avgzolrf = [0 for xtemp in range(len(numWords))]
        avgzolbag = [0 for xtemp in range(len(numWords))]
        avgzolsvm = [0 for xtemp in range(len(numWords))]
        
        stddevzoldt  = [0 for xtemp in range(len(numWords))]
        stddevzolrf = [0 for xtemp in range(len(numWords))]
        stddevzolbag = [0 for xtemp in range(len(numWords))]
        stddevzolsvm = [0 for xtemp in range(len(numWords))]
        
        stderrzoldt  = [0 for xtemp in range(len(numWords))]
        stderrzolrf = [0 for xtemp in range(len(numWords))]
        stderrzolbag = [0 for xtemp in range(len(numWords))]
        stderrzolsvm = [0 for xtemp in range(len(numWords))]
        
                     
        testnew = []
        trainnew = []
        


        for r in range(len(numWords)):
            
            for i in range(it):
                trainnew = []
                testnew = cv[i]
        
                for j in range(it):
                    if j != i:
                        for k in range(200):
                            trainnew.append(cv[j][k])

                temptrain = trainnew
                trainDataset  = getTrainData(temptrain, ratios[0])
   
                
                
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
                trainfv, trainfv0, trainfv1  = featureVector(wordList[:numWords[r]], x_train, y_train)
                testfv, testfv0, testfv1 = featureVector(wordList[:numWords[r]], x_test, y_test)
               
            
                zoltempdt[i] = decisionTree(trainfv,testfv)  
                zoltemprf[i] = randomForest(trainfv,testfv)   
                zoltempbag[i] = bagging(trainfv,testfv)      
                zoltempsvm[i] = svm(trainfv,testfv)   
                
            avgzoldt[r]  = np.mean(zoltempdt)
            avgzolrf[r] = np.mean(zoltemprf)
            avgzolbag[r] = np.mean(zoltempbag)
            avgzolsvm[r] = np.mean(zoltempsvm)
              
            stddevzoldt[r]  = np.std(zoltempdt)
            stddevzolrf[r]  = np.std(zoltemprf)
            stddevzolbag[r]  = np.std(zoltempbag)
            stddevzolsvm[r]  = np.std(zoltempsvm)
            
            stderrzoldt[r] = stddevzoldt[r]/math.sqrt(it)
            stderrzolrf[r] = stddevzolrf[r]/math.sqrt(it)
            stderrzolbag[r] = stddevzolbag[r]/math.sqrt(it)
            stderrzolsvm[r] = stddevzolsvm[r]/math.sqrt(it)
        

    
        print avgzoldt
        print avgzolrf
        print avgzolbag
        print avgzolsvm
        
        print stddevzoldt
        print stddevzolrf
        print stddevzolbag
        print stddevzolsvm

        print stderrzoldt
        print stderrzolrf
        print stderrzolbag 
        print stderrzolsvm   

        f = open(self.file,"a+");
        f.write("\n Ratio : ")
        f.write(str(ratios[0]))
        f.write("\n AVERAGE ZERO ONE LOSS")
        f.write("\n 1. Decision Tree")
        f.write(str(avgzoldt))
        f.write("\n 2. Bagging")
        f.write(str(avgzolbag))
        f.write("\n 3. Random forest")
        f.write(str(avgzolrf))
        f.write("\n 4. SVM")
        f.write(str(avgzolsvm))
        
        f.write("\n STANDARD DEVIATION ZERO ONE LOSS")
        f.write("\n 1. Decision Tree")
        f.write(str(stddevzoldt))     
        f.write("\n 2. Bagging")
        f.write(str(stddevzolbag))        
        f.write("\n 3. Random forest")
        f.write(str(stddevzolrf))        
        f.write("\n 4. SVM")
        f.write(str(stddevzolsvm))
        
        f.write("\n STANDARD ERROR ZERO ONE LOSS")
        f.write("\n 1. Decision Tree")
        f.write(str(stderrzoldt))       
        f.write("\n 2. Bagging")
        f.write(str(stderrzolbag))        
        f.write("\n 3. Random forest")
        f.write(str(stderrzolrf))       
        f.write("\n 4. SVM")
        f.write(str(stderrzolsvm))
        f.close();
   
    def changeNumTrees(self, numTrees):
        
        # Get data as array
        train = getLines(self.trainData, 100)
        
        cv = []         
        train = np.array(train)
        np.random.shuffle(train)
        
        for i in range(10):
            cv.append(train[i*200:(i+1)*200])
        
        
        zoltempdt = [0 for xtemp in range(10)]
        zoltemprf = [0 for xtemp in range(10)]
        zoltempbag = [0 for xtemp in range(10)]
        zoltempsvm = [0 for xtemp in range(10)]
        
        ratios = [0.25]
        w = 1000
        it = 10
        
        avgzoldt  = [0 for xtemp in range(len(ratios))]
        avgzolrf = [0 for xtemp in range(len(ratios))]
        avgzolbag = [0 for xtemp in range(len(ratios))]
        avgzolsvm = [0 for xtemp in range(len(ratios))]
        
        stddevzoldt  = [0 for xtemp in range(len(ratios))]
        stddevzolrf = [0 for xtemp in range(len(ratios))]
        stddevzolbag = [0 for xtemp in range(len(ratios))]
        stddevzolsvm = [0 for xtemp in range(len(ratios))]
        
        stderrzoldt  = [0 for xtemp in range(len(ratios))]
        stderrzolrf = [0 for xtemp in range(len(ratios))]
        stderrzolbag = [0 for xtemp in range(len(ratios))]
        stderrzolsvm = [0 for xtemp in range(len(ratios))]
        
                     
        testnew = []
        trainnew = []
        

        
        for r in range(len(numTrees)):              
            for i in range(it):
                trainnew = []
                testnew = cv[i]
        
                for j in range(it):
                    if j != i:
                        for k in range(200):
                            trainnew.append(cv[j][k])

                temptrain = trainnew
                trainDataset  = getTrainData(temptrain, ratios[0])
   
                
                
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
                trainfv, trainfv0, trainfv1  = featureVector(wordList[:w], x_train, y_train)
                testfv, testfv0, testfv1 = featureVector(wordList[:w], x_test, y_test)
               
            
#                    zoltempdt[i] = decisionTree(trainfv,testfv)  
                zoltemprf[i] = randomForest(trainfv,testfv,10,10,num)   
                zoltempbag[i] = bagging(trainfv,testfv,10,10,num)      
#                    zoltempsvm[i] = svm(trainfv,testfv)   
                
            avgzoldt[r]  = np.mean(zoltempdt)
            avgzolrf[r] = np.mean(zoltemprf)
            avgzolbag[r] = np.mean(zoltempbag)
            avgzolsvm[r] = np.mean(zoltempsvm)
              
            stddevzoldt[r]  = np.std(zoltempdt)
            stddevzolrf[r]  = np.std(zoltemprf)
            stddevzolbag[r]  = np.std(zoltempbag)
            stddevzolsvm[r]  = np.std(zoltempsvm)
            
            stderrzoldt[r] = stddevzoldt[r]/math.sqrt(it)
            stderrzolrf[r] = stddevzolrf[r]/math.sqrt(it)
            stderrzolbag[r] = stddevzolbag[r]/math.sqrt(it)
            stderrzolsvm[r] = stddevzolsvm[r]/math.sqrt(it)
        

    
#            print avgzoldt
        print avgzolbag
        print avgzolrf
#            print avgzolsvm
        
#            print stddevzoldt

        print stddevzolbag
        print stddevzolrf
#            print stddevzolsvm

#            print stderrzoldt

        print stderrzolbag 
        print stderrzolrf
#            print stderrzolsvm   

        f = open(self.file,"a+");
        f.write("\n\n No. of trees : " +  str(num))
        f.write("\n AVERAGE ZERO ONE LOSS")
#            f.write("\n 1. Decision Tree")
#            f.write(str(avgzoldt))
        f.write("\n 1. Bagging")
        f.write(str(avgzolbag))
        f.write("\n 2. Random forest")
        f.write(str(avgzolrf))
#            f.write("\n 4. SVM")
#            f.write(str(avgzolsvm))
        
        f.write("\n STANDARD DEVIATION ZERO ONE LOSS")
#            f.write("\n 1. Decision Tree")
#            f.write(str(stddevzoldt))     
        f.write("\n 1. Bagging")
        f.write(str(stddevzolbag))        
        f.write("\n 2. Random forest")
        f.write(str(stddevzolrf))        
#            f.write("\n 4. SVM")
#            f.write(str(stddevzolsvm))
        
        f.write("\n STANDARD ERROR ZERO ONE LOSS")
#            f.write("\n 1. Decision Tree")
#            f.write(str(stderrzoldt))       
        f.write("\n 1. Bagging")
        f.write(str(stderrzolbag))        
        f.write("\n 2. Random forest")
        f.write(str(stderrzolrf))       
#            f.write("\n 4. SVM")
#            f.write(str(stderrzolsvm))
        f.close();
        

    def changeDepth(self, depths):
        
        # Get data as array
        train = getLines(self.trainData, 100)
        
        cv = []         
        train = np.array(train)
        np.random.shuffle(train)
        
        for i in range(10):
            cv.append(train[i*200:(i+1)*200])
        
        
        zoltempdt = [0 for xtemp in range(10)]
        zoltemprf = [0 for xtemp in range(10)]
        zoltempbag = [0 for xtemp in range(10)]
        zoltempsvm = [0 for xtemp in range(10)]
        
        w = 1000
        it = 10
        
        avgzoldt  = [0 for xtemp in range(len(depths))]
        avgzolrf = [0 for xtemp in range(len(depths))]
        avgzolbag = [0 for xtemp in range(len(depths))]
        avgzolsvm = [0 for xtemp in range(len(depths))]
        
        stddevzoldt  = [0 for xtemp in range(len(depths))]
        stddevzolrf = [0 for xtemp in range(len(depths))]
        stddevzolbag = [0 for xtemp in range(len(depths))]
        stddevzolsvm = [0 for xtemp in range(len(depths))]
        
        stderrzoldt  = [0 for xtemp in range(len(depths))]
        stderrzolrf = [0 for xtemp in range(len(depths))]
        stderrzolbag = [0 for xtemp in range(len(depths))]
        stderrzolsvm = [0 for xtemp in range(len(depths))]
        
                     
        testnew = []
        trainnew = []
        

        for r in range(len(depths)):
                
            for i in range(it):
                trainnew = []
                testnew = cv[i]
        
                for j in range(it):
                    if j != i:
                        for k in range(200):
                            trainnew.append(cv[j][k])

                temptrain = trainnew
                trainDataset  = getTrainData(temptrain, 0.25)
   
                
                
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
                trainfv, trainfv0, trainfv1  = featureVector(wordList[:w], x_train, y_train)
                testfv, testfv0, testfv1 = featureVector(wordList[:w], x_test, y_test)
               
            
                zoltempdt[i] = decisionTree(trainfv,testfv,depths[r])  
                zoltemprf[i] = randomForest(trainfv,testfv,depths[r])   
                zoltempbag[i] = bagging(trainfv,testfv,depths[r])      
#                    zoltempsvm[i] = svm(trainfv,testfv)   
    
            avgzoldt[r]  = np.mean(zoltempdt)
            avgzolrf[r] = np.mean(zoltemprf)
            avgzolbag[r] = np.mean(zoltempbag)
            avgzolsvm[r] = np.mean(zoltempsvm)
              
            stddevzoldt[r]  = np.std(zoltempdt)
            stddevzolrf[r]  = np.std(zoltemprf)
            stddevzolbag[r]  = np.std(zoltempbag)
            stddevzolsvm[r]  = np.std(zoltempsvm)
            
            stderrzoldt[r] = stddevzoldt[r]/math.sqrt(it)
            stderrzolrf[r] = stddevzolrf[r]/math.sqrt(it)
            stderrzolbag[r] = stddevzolbag[r]/math.sqrt(it)
            stderrzolsvm[r] = stddevzolsvm[r]/math.sqrt(it)
        

        
        print avgzoldt
        print avgzolbag
        print avgzolrf
#            print avgzolsvm
        
        print stddevzoldt
        print stddevzolbag
        print stddevzolrf
#            print stddevzolsvm

        print stderrzoldt
        print stderrzolbag
        print stderrzolrf 
#            print stderrzolsvm   

        f = open(self.file,"a+");
        f.write("\n AVERAGE ZERO ONE LOSS")
        f.write("\n 1. Decision Tree")
        f.write(str(avgzoldt))
        f.write("\n 2. Bagging")
        f.write(str(avgzolbag))
        f.write("\n 3. Random forest")
        f.write(str(avgzolrf))
#            f.write("\n 4. SVM")
#            f.write(str(avgzolsvm))
        
        f.write("\n STANDARD DEVIATION ZERO ONE LOSS")
        f.write("\n 1. Decision Tree")
        f.write(str(stddevzoldt))     
        f.write("\n 2. Bagging")
        f.write(str(stddevzolbag))        
        f.write("\n 3. Random forest")
        f.write(str(stddevzolrf))        
#            f.write("\n 4. SVM")
#            f.write(str(stddevzolsvm))
        
        f.write("\n STANDARD ERROR ZERO ONE LOSS")
        f.write("\n 1. Decision Tree")
        f.write(str(stderrzoldt))       
        f.write("\n 2. Bagging")
        f.write(str(stderrzolbag))        
        f.write("\n 3. Random forest")
        f.write(str(stderrzolrf))       
#            f.write("\n 4. SVM")
#            f.write(str(stderrzolsvm))
        f.close();
            
            
    def analysisDriver(self):
        
        
        ratios1 = [0.025, 0.05, 0.125, 0.25]
        words1 = [1000]       
        f = open(self.file,"a+")
        f.write("\n_____________________________________________________________");
        f.write("\n Analysis 1")
        f.close();
        self.changeRatios(ratios1, words1)
        
#        ratios2 = [0.25]
#        words2 = [200, 500, 1000, 1500]
#        f = open(self.file,"a+")
#        f.write("\n_____________________________________________________________");
#        f.write("\n Analysis 2")
#        f.close();
#        self.changeWords(ratios2, words2)
        
        
#        depths = [5,10,15,20]
#        f = open(self.file,"a+")
#        f.write("\n_____________________________________________________________");
#        f.write("\n Analysis 3")
#        f.close();
#        self.changeDepth(depths)
        
#        numTrees = [10,25,50,100]
##        numTrees = [25]
#        f = open(self.file,"a+")  
#        f.write("\n_____________________________________________________________");
#        f.write("\n Analysis 4")
#        f.close();        
#        self.changeNumTrees(numTrees)
        
        
        