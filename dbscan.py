#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 19:44:26 2017

@author: harilsatra
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

noise = []
def dbScan(points, eps, min_pts, dist_mat, cluster, visited):
    global noise
    c=0
    for i in range(0,len(points)):
        if visited[i][0] == 0:
            visited[i][0] = 1
            neighbors = regionQuery(i,eps,dist_mat,len(points))
            if len(neighbors) < min_pts:
                noise.append(i)
            else:
                c = c+1
                expandCluster(i, neighbors, c, eps, min_pts, cluster, visited,no_genes)
            
def expandCluster(point, neighbors, c, eps, min_pts, cluster, visited,no_genes):
    cluster[point] = c
    while len(neighbors)>0:
        i = neighbors.pop()
        if visited[i][0] == 0:
            visited[i][0] = 1
            new_neighbors = regionQuery(i,eps,dist_mat,no_genes)
            if len(new_neighbors) >= min_pts:
                neighbors.update(new_neighbors)
        if cluster[i][0] == 0:
            cluster[i][0] = c
    return
                
            
                   
def regionQuery(point, eps, dist_mat,no_genes):
    neighbors = set()
    for i in range(0, no_genes):
        if dist_mat[point][i] <= eps:
            neighbors.add(i)
    return neighbors





#Read file and store into a 2d array
with open("cho.txt") as textFile:
    lines = [line.split() for line in textFile]
    
#Convert 2d array into np array    
input_data = np.asarray(lines)

points = input_data[:,2:len(input_data[0])]
points = np.mat(points,dtype=float)

ground_truth = input_data[:,1]

no_genes = len(input_data)
no_attr = np.shape(points)[1]

dist_mat = euclidean_distances(points,points)
cluster = np.zeros((no_genes,1))
visited = np.zeros((no_genes,1))
    

eps=1.1
min_pts=4
dbScan(points, eps, min_pts, dist_mat, cluster, visited)

#print(np.shape(clusters))
cluster_matrix = np.zeros((no_genes,no_genes))
for i in range(0,no_genes):
    for j in range(0,no_genes):
        if cluster[i] == cluster[j]:
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
count = set()
for i in range(0,len(cluster)):
    count.add(cluster[i][0])
print(len(count))
