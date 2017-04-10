# -*- coding: utf-8 -*-
"""
Created on Fri Apr 07 23:02:07 2017

@author: Sriranjitha
"""
import numpy as np
d = [[1,0,0],[0,0,3],[0,0,1]]

f = 0;

d = np.array(d)


l = d[d[:,f] == 0]
r = d[d[:,f] == 1]
print l
print r

a = [1,1,1,1,0,0,1,1,0,0,0,0,0,0,0]

print max(set(a),key=a.count)

b = [0,1,2,3,4]

c = b
#b.pop(3)
#print b
#print c

c = np.array(b)
c = np.delete(c,3)

print c.tolist()
c = c.tolist()

print c
print b

x = [1,2,3,4,11,12,13,14,21,22,23,24,31,32,33,34]
y = [1,2,5,6]

z = [x is y]
print z

def callfn(b, index):
    c = np.array(b)
    
    if(len(b)>0):
        c = np.delete(c,len(b)/2)

    else:
        return
        #    print c.tolist()
    c = c.tolist()
    print c
    
    if(index == 2 or index == -1):
        return 
        
    callfn(c, index-1)
    callfn(c, index+1)
    
#callfn(x,0)

arr = [1,2,3,4,5,6,7,8,9,10,11,12]
arr = np.array(arr).reshape(3,4)
print "arr : ",arr
r = np.random.choice(np.arange(3),3)
print r
print np.take(arr,r,0)

print "_________________"
q = np.random.choice(np.array(x),len(x))
print q.tolist()

 

print d[:,d.shape[1]-1]
print d[:,-1]

print np.bincount(q).argmax()


print "+++++++++++++++++++++++++"

sarr = [0.9,0.5,1,0,0,.5]
print np.around(sarr)

x1 = 0.0
if(x1 == 0):
    print "YEP"

f = open("temp.txt","a+")
f.write("hey"+ str(sarr))
f.close();