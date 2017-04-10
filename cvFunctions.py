# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 02:06:00 2017

@author: Sriranjitha
"""
import numpy as np
from random import sample

def getTrainData(trainfv, ratio):
    trainfv = np.array(trainfv)
    lt = len(trainfv) #length of data
    l = 2000
    f = float(l)*(float(ratio))  #number of elements you need
    indices = sample(range(lt),int(f))
    
    train_data = trainfv[indices]
    return train_data
