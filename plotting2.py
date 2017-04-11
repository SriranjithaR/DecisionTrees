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

## use pylab to plot x and y
#dtplot  = plt.plot(x, avgzoldt,  'b', label = 'DT Mean')
#bagplot = plt.plot(x, avgzolbag, 'c', label = 'BAG Mean')
#rfplot  = plt.plot(x, avgzolrf,  'k', label = 'RF Mean')
#svmplot = plt.plot(x, avgzolsvm, 'r', label = 'SVM Mean')

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
plt.errorbar(x = x, y = avgzoldt, yerr = avgstderrdt,  capthick=2, label = 'DT mean')
plt.errorbar(x = x, y = avgzolbag, yerr = avgstderrbag, capthick=2, label = 'BAG mean')
plt.errorbar(x = x, y = avgzolrf, yerr = avgstderrrf,  capthick=2, label = 'RF mean')
plt.errorbar(x = x, y = avgzolsvm, yerr = avgstderrsvm,  capthick=2, label = 'SVM mean')
plt.legend(loc='upper right')
# show the plot on the screen
plt.show()


fig.savefig('analysis2.png')



# # Make an array of x values
# x2 = [10, 50, 250, 500, 1000, 4000]
# # Make an array of y values for each x value
# avg2 = [0.33160000000000001, 0.23180000000000001, 0.13340000000000002, 0.090599999999999972, 0.061700000000000012, 0.053400000000000024]
# std2 = [0.050329315512929441, 0.026029982712249353, 0.017692936443677192, 0.013395521639712294, 0.0093706990134140612, 0.012126005112979274]
# avg_error2 = [0.47859999999999997, 0.4849, 0.47970000000000007, 0.4803, 0.47799999999999992, 0.47809999999999997]
# std_error2 = [0.0078128099938498501, 0.0058043087443725846, 0.01049809506529638, 0.013476275449841485, 0.008921883209278194, 0.0059236812878479591]
#
# # use pylab to plot x and y
# avgplot = plt.plot(x2, avg2, 'b', label = 'Mean')
# stdplot = plt.plot(x2, std2, 'g', label = 'Standard Deviation')
# avgerrorplot = plt.plot(x2, avg_error2, 'r', label = 'Mean error')
# stderrorplot = plt.plot(x2, std_error2, 'y', label = 'Standard Deviation in error')
# plt.title('4(b)')
# plt.xlabel('zero-one loss')
# plt.ylabel('feature size')
# plt.legend(handles = [avgplot[0], stdplot[0], avgerrorplot[0], stderrorplot[0]])
# # show the plot on the screen
# plt.show()
