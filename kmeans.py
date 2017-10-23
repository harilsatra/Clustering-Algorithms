#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 18:10:47 2017

@author: harilsatra
"""

import numpy as np
import random
from scipy.spatial import distance

def isEqual(centroids,new_centroids):
    for i in range(len(centroids)):
        if np.linalg.norm(new_centroids[i]-centroids[i]) != 0:
            return False;
    return True;

#Read file and store into a 2d array
with open("cho.txt") as textFile:
    lines = [line.split() for line in textFile]
    
#Convert 2d array into np array    
input_data = np.asarray(lines)

points = input_data[:,2:len(input_data[0])]
points = np.mat(points,dtype=float)

ground_truth = input_data[:,1]
#ground_truth = np.mat(ground_truth,dtype=float)
#print(ground_truth)

no_genes = len(input_data)
no_attr = np.shape(points)[1]
k = 5
centroids = []
random_set = set()
while len(random_set)<k:
    no_random = random.randint(0,no_genes)
    if no_random not in random_set:
        centroids.append(points[no_random].tolist())
        #print(no_random)
    random_set.add(no_random)

#print(np.asmatrix(np.array(centroids)))

while True:
    dist = np.zeros((no_genes,k))
    
    for i in range(0,no_genes):
        for j in range(0,k):
            dist[i][j] = np.linalg.norm(points[i]-centroids[j])
    
    clusters = np.argmin(dist,axis=1)
    
    new_centroids = np.zeros((k,no_attr))
    no_points = np.zeros(k)
    
    for i in range(0,no_genes):
        new_centroids[clusters[i]] += np.ravel(points[i])
        no_points[clusters[i]] += 1
        
    
    #print(new_centroids)
    #print(no_points)
    
    for i in range(0,k):
        for j in range(0,no_attr):
            new_centroids[i][j] /= no_points[i]
    
    if isEqual(centroids,new_centroids):
        break
    else:
        centroids = new_centroids
    #print(centroids)

#print(np.shape(clusters))
cluster_matrix = np.zeros((no_genes,no_genes))
for i in range(0,no_genes):
    for j in range(0,no_genes):
        if clusters[i] == clusters[j]:
            cluster_matrix[i][j] = 1
                          
truth_matrix = np.zeros((no_genes,no_genes))
for i in range(0,no_genes):
    for j in range(0,no_genes):
        if ground_truth[i] == ground_truth[j]:
            truth_matrix[i][j] = 1

agree_disagree = np.zeros((2,2))
#print(agree_disagree)
count = 0
for i in range(0,no_genes):
    for j in range(0,no_genes):
        if truth_matrix[i][j] ==  cluster_matrix[i][j]:
            if  truth_matrix[i][j] == 1:
                agree_disagree[0][0] += 1
            else:
                agree_disagree[1][1] += 1
        else:
            if truth_matrix[i][j] == 1:
                agree_disagree[0][1] += 1
            else:
                agree_disagree[1][0] += 1

#print(agree_disagree)

#jaccard = agree_disagree[0][0]/(agree_disagree[0][0]+agree_disagree[1][0]+agree_disagree[0][1])
rand = (agree_disagree[0][0]+agree_disagree[1][1])/(agree_disagree[0][0]+agree_disagree[1][0]+agree_disagree[0][1]+agree_disagree[1][1])
print(rand)