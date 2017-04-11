import matplotlib.pyplot as plt
import numpy as np

# Make an array of x values
x = [0.025, 0.05, 0.125, 0.25]
# Make an array of y values for each x value

avgzoldt = [0.32399999999999995, 0.31950000000000001, 0.23499999999999996, 0.215]
avgzolbag= [0.3165, 0.272105263158, 0.19650000000000004, 0.20200000000000001]
avgzolrf = [0.29749999999999999, 0.24749999999999997, 0.1565, 0.17250000000000001]

fig = plt.figure()

# use pylab to plot x and y
dtplot = plt.plot(x, avgzoldt,  'b', label = 'DT Mean')
bagplot = plt.plot(x, avgzolbag,'g', label = 'BAG Mean')
rfplot = plt.plot(x, avgzolrf,  'y', label = 'RF Mean')

#STANDARD ERROR ZERO ONE LOSS
avgstderrdt = [0.019297668252926305, 0.014974144382902147, 0.016309506430300089, 0.014017845768876177]
avgstderrbag= [0.016677080080157915, 0.018401766219577943, 0.014125331854508762, 0.0072886898685566182]
avgstderrrf = [0.023717609491683608, 0.019341664871463361, 0.01539561625918235, 0.017481418706729729]


plt.title('Analysis 1')
plt.xlim(0.0,0.3)
plt.xlabel('training set size in %')
plt.ylabel('zero-one loss')
plt.legend(handles = [dtplot[0], bagplot[0], rfplot[0]])
plt.errorbar(x = x, y = avgzoldt, yerr = avgstderrdt, ecolor='b', capthick=2)
plt.errorbar(x = x, y = avgzolbag, yerr = avgstderrbag, ecolor='g', capthick=2)
plt.errorbar(x = x, y = avgzolrf, yerr = avgstderrrf, ecolor='y', capthick=2)
# show the plot on the screen
plt.show()


fig.savefig('analysis1.png')



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
