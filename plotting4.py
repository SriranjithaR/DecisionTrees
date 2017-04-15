# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:22:27 2017

@author: Sriranjitha
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 23:38:41 2017

@author: Sriranjitha
"""

import matplotlib.pyplot as plt

# Make an array of x values
x = [10,25,50,100]
# Make an array of y values for each x value

avgzoldt = [0.215, 0.215, 0.215, 0.215]
avgzolbag = [0.18050000000000002, 0.16600000000000001, 0.17050000000000001, 0.15850000000000003]
avgzolrf = [0.22999999999999998, 0.1875, 0.15299999999999997, 0.15349999999999997]
#avgzolrf = [0.19600000000000004, 0.17050000000000004, 0.17850000000000002, 0.157]
#avgzolbag  = [0.23300000000000001, 0.1615, 0.18350000000000002, 0.1595]

fig = plt.figure()


#STANDARD ERROR ZERO ONE LOSS
avgstderrrf = [0.01216141439142668, 0.0096033848199475972, 0.010584186317332094, 0.0092520268049763009]
avgstderrbag  = [0.013605146085213493, 0.022978794572387821, 0.016659081607339584, 0.024662218067319091]
avgstderrdt = [0.014017845768876177, 0.014017845768876177, 0.014017845768876177, 0.014017845768876177]

plt.title('Analysis 4')
plt.xlim(0,125)
plt.xlabel('number of trees')
plt.ylabel('zero-one loss')

plt.errorbar(x = x, y = avgzolbag, yerr = avgstderrbag, capsize=2, label = 'BAG mean')
plt.errorbar(x = x, y = avgzolrf, yerr = avgstderrrf,  capsize=2, label = 'RF mean')
plt.errorbar(x = x, y = avgzoldt, yerr = avgstderrdt,  capsize=2, label = 'Dt mean')
plt.legend(loc='upper right')
# show the plot on the screen
plt.show()


fig.savefig('analysis4.png')

