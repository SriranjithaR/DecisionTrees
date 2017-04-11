# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:11:35 2017

@author: Sriranjitha
"""
import numpy as np
import matplotlib.pyplot as plt
def plot_results_training_set(LR_results, NBC_results, SVM_results):
    x = np.arange(LR_results.shape[0])
    n = LR_results.shape[1]
    x_labels = np.array([0.01, 0.03, 0.05, 0.08, 0.1, 0.15]) * 2000 
    plt.errorbar(x, LR_results.mean(axis=1), yerr=LR_results.std(axis=1)/np.sqrt(n),label='LR')
    plt.errorbar(x, NBC_results.mean(axis=1), yerr=NBC_results.std(axis=1)/np.sqrt(n),label='NBC')
    plt.errorbar(x, SVM_results.mean(axis=1), yerr=SVM_results.std(axis=1)/np.sqrt(n),label='SVM')
    plt.xticks(x, x_labels)
    plt.legend(loc='upper right')
    plt.xlabel("Training set size")
    plt.ylabel("Zero-one loss")  
    plt.savefig("plot_1.pdf", bbox_inches='tight')