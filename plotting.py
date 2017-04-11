import matplotlib.pyplot as plt
import numpy as np

# Make an array of x values
x = [0.025, 0.05, 0.125, 0.25]
# Make an array of y values for each x value

avgzoldt = [0.32399999999999995, 0.31950000000000001, 0.23499999999999996, 0.215]
avgzolbag= [0.3165, 0.272105263158, 0.19650000000000004, 0.20200000000000001]
avgzolrf = [0.29749999999999999, 0.24749999999999997, 0.1565, 0.17250000000000001]
avgzolsvm = [0.29450000000000004, 0.16800000000000001, 0.090999999999999984, 0.08249999999999999]

fig = plt.figure()

#STANDARD ERROR ZERO ONE LOSS
avgstderrdt = [0.019297668252926305, 0.014974144382902147, 0.016309506430300089, 0.014017845768876177]
avgstderrbag= [0.016677080080157915, 0.018401766219577943, 0.014125331854508762, 0.0072886898685566182]
avgstderrrf = [0.023717609491683608, 0.019341664871463361, 0.01539561625918235, 0.017481418706729729]
avgstderrsvm = [0.031169295789285966, 0.011384199576606163, 0.007169379331573968, 0.006754628043053143]

plt.title('Analysis 1')
plt.xlim(0.0,0.3)
plt.xlabel('training set size in %')
plt.ylabel('zero-one loss')
#plt.legend(handles = [dtplot[0], bagplot[0], rfplot[0]])
plt.errorbar(x = x, y = avgzoldt, yerr = avgstderrdt, capsize=2, label = 'DT mean')
plt.errorbar(x = x, y = avgzolbag, yerr = avgstderrbag,  capsize=2, label = 'BAG mean')
plt.errorbar(x = x, y = avgzolrf, yerr = avgstderrrf,  capsize=2, label = 'RF mean')
plt.errorbar(x = x, y = avgzolsvm, yerr = avgstderrsvm,  capsize=2, label = 'SVM mean')
# show the plot on the screen
plt.legend(loc='upper right')
plt.show()


fig.savefig('analysis1.png')

