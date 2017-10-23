#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 16:01:54 2017

@author: harilsatra
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import sys

def generateCluster(dist_mat,clusters,no_genes,cluster_count,max_cluster):
    if max_cluster == no_genes:
        return
    
    single_link = sys.maxsize
    cluster1 = -1
    cluster2 = -1
    
    #Find the two most similar clusters (least distance) from the distance matrix
    for i in range(0,cluster_count):
        for j in range(i+1,cluster_count):
            if single_link > dist_mat[i][j]:
                single_link = dist_mat[i][j]
                cluster1 = i
                cluster2 = j
    
    new_cluster = []
    new_cluster.append(dist_mat[cluster1][no_genes])
    new_cluster.append(dist_mat[cluster2][no_genes])
    
    clusters[cluster_count] = new_cluster
            

#Read file and store into a 2d array
with open("iyer.txt") as textFile:
    lines = [line.split() for line in textFile]
    
#Convert 2d array into np array    
input_data = np.asarray(lines)

points = input_data[:,2:len(input_data[0])]
points = np.mat(points,dtype=float)

ground_truth = input_data[:,1]

no_genes = len(input_data)
no_attr = np.shape(points)[1]

dist_mat = euclidean_distances(points,points)
print(type(dist_mat))
dist_mat = np.concatenate((dist_mat,np.zeros((no_genes,1))),1)

for i in range(0,no_genes):
    dist_mat[i][no_genes] = i+1
                
print(dist_mat)

clusters = {}
generateCluster(dist_mat,clusters,no_genes,no_genes+1)