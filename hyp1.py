# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:41:56 2017

@author: Sriranjitha
"""

from scipy import stats
import numpy as np


# Hyp 1 : Dt vs svm
dt = [0.32399999999999995, 0.31950000000000001, 0.23499999999999996, 0.215]
svm = [0.29450000000000004, 0.16500000000000001, 0.095499999999999974, 0.073499999999999982]
ratios = [0.025, 0.05, 0.125, 0.25]
#dt = np.array(dt)
#svm = np.array(svm)
f = open("analysis.txt","a+")
f.write("\n\n_________________________________________");
f.write("\nAnalysis Qn 1");
f.write("\n\nTSS \t p value");
for i in range(len(dt)):
    t,p = stats.ttest_rel(dt[i],svm[i])
    print t,p
    f.write("\n"+str(ratios[i])+"\t"+str(p));
f.close();

# Hyp 2 :  Dt vs svm
dt = [0.23849999999999999, 0.22800000000000004, 0.24100000000000002, 0.20300000000000001]
svm = [0.156, 0.087499999999999981, 0.078999999999999987, 0.072999999999999982]
#dt = np.array(dt)
#svm = np.array(svm)
words = [200,500,1000,1500]
f = open("analysis.txt","a+")
f.write("\n\n_________________________________________");
f.write("\nAnalysis Qn 2");
f.write("\n\nNo. of words \t p value");
for i in range(len(dt)):
    t,p = stats.ttest_rel(dt[i],svm[i])
    print t,p
    f.write("\n"+str(words[i])+"\t"+str(p));
f.close();

# Hyp 3 :  Dt vs rf
dt = [0.26949999999999996, 0.22500000000000001, 0.19400000000000001, 0.20950000000000002]
rf = [0.2495, 0.1565, 0.12050000000000001, 0.098499999999999976]
#dt = np.array(dt)
#rf = np.array(svm)
f = open("analysis.txt","a+")
f.write("\n\n_________________________________________");
f.write("\nAnalysis Qn 3");
f.write("\n\nDepth \t p value");
depths = [5,10,15,20]
for i in range(len(dt)):
    t,p = stats.ttest_rel(dt[i],rf[i])
    print t,p
    f.write("\n"+str(depths[i])+"\t"+str(p));
f.close();

       
# Hyp 4 :  Dt vs rf
dt = [0.215, 0.215, 0.215, 0.215]
rf = [0.19600000000000004, 0.17050000000000004, 0.17850000000000002, 0.157]
#dt = np.array(dt)
#rf = np.array(svm)
f = open("analysis.txt","a+")
f.write("\n\n_________________________________________");
f.write("\nAnalysis Qn 4");
f.write("\n\nNo. of trees \t p value");
depths = [5,10,15,20]
for i in range(len(dt)):
    t,p = stats.ttest_rel(dt[i],rf[i])
    print t,p
    f.write("\n"+str(depths[i])+"\t"+str(p));
f.close();
      