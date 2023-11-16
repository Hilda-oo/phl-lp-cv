# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 15:04:16 2021

@author: Xinzhuo Hu, Liangchao Zhu
"""

'''
basic vectorized operations for np.array
'''

import numpy as np

## distance from grid points *pm* TO the point *a*
def dist(a,pm):
    ap = np.zeros(pm.shape)
    ap[:,:,:,0] = pm[:,:,:,0]-a[0]
    ap[:,:,:,1] = pm[:,:,:,1]-a[1]
    ap[:,:,:,2] = pm[:,:,:,2]-a[2]
    
    ll=(ap[:,:,:,0]**2 + ap[:,:,:,1]**2 + ap[:,:,:,2]**2)**0.5
    return ll

# https://blog.csdn.net/angelazy/article/details/38489293
## cos(ap,ab)
def calculater(a, b, p):
    ap = np.subtract(p, a)
    ab = np.subtract(b, a)
    r = np.dot(ap, ab) / (np.linalg.norm(ab) ** 2)
    return r

# v1: start point of the line segment, 1x3 array
# v2: end point of the line segment, 1x3 array
# p: query point, 1x3 array
# p到v1v2的距离
def distance2line(v1, v2, p):
    n = np.subtract(v2, v1)
    m = np.subtract(v1, p)
    d = np.linalg.norm(np.cross(n, m)) / np.linalg.norm(n)
    return d

# a: start point of the line segment, 1x3 array
# b: end point of the line segment,  1x3 array
# p: query point, (resolution+1)**3 x 3 array
def calculatervector(a, b, pm):
    #ap:a到pm中每个点的向量
    ap = np.subtract(pm, a)
    #ab:a到b的向量
    ab = np.subtract(b, a)
    #r:ap在ab方向上的投影长度
    r = np.dot(ap, ab) / (np.linalg.norm(ab) ** 2)
    return r

# m: h x h x h x 3 array
# p: p-norm
def myvectorizationNorm(m, p):
    h = m.shape[0]
    temp_sum = np.zeros([h, h, h], dtype=float)
    for i in range(m.shape[3]):
        temp_sum = temp_sum + (m[:, :, :, i] ** p)
    mynorm = np.sqrt(temp_sum)
    return mynorm

# v1: start point of the line segment, 1x3 array
# v2: end point of the line segment, 1x3 array
# p: query matrix, h x h x h x 3 array
def dist2linevectorization(v1, v2, p):
    n = np.subtract(v2, v1)
    m = np.subtract(v1, p)
    d = myvectorizationNorm(np.cross(n, m), 2) / np.linalg.norm(n)
    return d