# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 23:38:41 2017

@author: Sriranjitha
"""

import matplotlib.pyplot as plt

# Make an array of x values
x = [5,10,15,20]
# Make an array of y values for each x value

avgzoldt  = [0.26949999999999996, 0.22500000000000001, 0.19400000000000001, 0.20950000000000002]
avgzolbag = [0.159, 0.17800000000000002, 0.17549999999999999, 0.16650000000000001]
avgzolrf  = [0.2495, 0.1565, 0.12050000000000001, 0.098499999999999976]

fig = plt.figure()


#STANDARD ERROR ZERO ONE LOSS
avgstderrdt  = [0.016844138446355753, 0.012186057606953939, 0.011995832609702415, 0.0092046184059959776]
avgstderrbag = [0.011287160847617966, 0.0093327380762560745, 0.014619336510252439, 0.014054358754493212]
avgstderrrf  = [0.034084087196226916, 0.017190840584450778, 0.010896100219803416, 0.0066351337590134594]

plt.title('Analysis 3')
plt.xlim(0,25)
plt.xlabel('tree depth')
plt.ylabel('zero-one loss')

plt.errorbar(x = x, y = avgzoldt, yerr = avgstderrdt,  capthick=2, label = 'DT mean')
plt.errorbar(x = x, y = avgzolbag, yerr = avgstderrbag, capthick=2, label = 'BAG mean')
plt.errorbar(x = x, y = avgzolrf, yerr = avgstderrrf,  capthick=2, label = 'RF mean')
plt.legend(loc='upper right')
# show the plot on the screen
plt.show()


fig.savefig('analysis3 .png')

