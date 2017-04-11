# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 23:38:41 2017

@author: Sriranjitha
"""

import matplotlib.pyplot as plt
import numpy as np

# Make an array of x values
x = [200,500,1000,1500]
# Make an array of y values for each x value

avgzoldt  = [0.23849999999999999, 0.22800000000000004, 0.24100000000000002, 0.20300000000000001]
avgzolbag = [0.17650000000000005, 0.15150000000000002, 0.15100000000000002, 0.16100000000000003]
avgzolrf  = [0.17450000000000002, 0.15450000000000003, 0.13450000000000001, 0.12599999999999997]
avgzolsvm = [0.156, 0.087499999999999981, 0.078999999999999987, 0.072999999999999982]

fig = plt.figure()


#STANDARD ERROR ZERO ONE LOSS
avgstderrdt  = [0.0075183109805328019, 0.017919263377717286, 0.019959959919799437, 0.0098030607465219716]
avgstderrbag = [0.0071780916684032314, 0.007715244649393823, 0.0089106677639781869, 0.0086833173384369654]
avgstderrrf  = [0.010640723659601347, 0.010873132023478792, 0.013480541532149217, 0.01187013058057914]
avgstderrsvm = [0.0082704292512541331, 0.0051112620750652172, 0.0052820450584976954, 0.0044271887242357238]

plt.title('Analysis 2')
plt.xlim(200,1600)
plt.xlabel('number of words')
plt.ylabel('zero-one loss')
#plt.legend(handles = [dtplot[0],bagplot[0],  rfplot[0], svmplot[0]])
plt.errorbar(x = x, y = avgzoldt, yerr = avgstderrdt,  capsize=2, label = 'DT mean')
plt.errorbar(x = x, y = avgzolbag, yerr = avgstderrbag, capsize=2, label = 'BAG mean')
plt.errorbar(x = x, y = avgzolrf, yerr = avgstderrrf,  capsize=2, label = 'RF mean')
plt.errorbar(x = x, y = avgzolsvm, yerr = avgstderrsvm,  capsize=2, label = 'SVM mean')
plt.legend(loc='upper right')
# show the plot on the screen
plt.show()


fig.savefig('analysis2.png')
