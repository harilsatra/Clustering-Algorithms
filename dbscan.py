#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 19:44:26 2017

@author: harilsatra
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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
    return c
            
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
               
                   
def regionQuery(point, eps, dist_mat,no_genes):
    neighbors = set()
    for i in range(0, no_genes):
        if dist_mat[point][i] <= eps:
            neighbors.add(i)
    return neighbors

def calcExtIndex(cluster):
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



    jaccard = agree_disagree[0][0]/(agree_disagree[0][0]+agree_disagree[1][0]+agree_disagree[0][1])
    rand = (agree_disagree[0][0]+agree_disagree[1][1])/(agree_disagree[0][0]+agree_disagree[1][0]+agree_disagree[0][1]+agree_disagree[1][1])

    return jaccard,rand

def generatePlot(clusterList, count_clusters):
    pca = PCA(n_components=2)
    pca.fit(points)
    reducedPoints = pca.transform(points)
    
    labels = []
    for i in range(0,count_clusters+1):
        labels.append(i)

    colors = [plt.cm.jet(float(i)/(max(labels))) for i in labels]
    
    for i, l in enumerate(labels):
       # plot_list = [(plt.plot(plot_dict_x[l],plot_dict_y[l],'+', c = plt.cm.jet(float(i)/(len(labels))), label = str(l)))]
       x = [reducedPoints[j][0] for j in range(len(reducedPoints)) if int(clusterList[j]) == (l)]
       y = [reducedPoints[j][1] for j in range(len(reducedPoints)) if int(clusterList[j]) == (l)]
       plt.plot(x, y,'wo', c= colors[i], label = str(l), markersize=9, alpha=0.75)
       
      
    
    fig_size = plt.rcParams["figure.figsize"]
    # Set figure width to 12 and height to 9
    fig_size[0] = 12
    fig_size[1] = 9
    plt.rcParams["figure.figsize"] = fig_size
    
    #plt.scatter(reducedPoints[:, 0], reducedPoints[:, 1], c=clusterList, alpha=0.5)
    plt.title('Density Based Clustering')
    plt.legend(numpoints=1)
    plt.grid(True)
    plt.show()


#Read file and store into a 2d array
filename = input("Enter the filename with extension: ")
with open(filename) as textFile:
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
    

eps=input("Enter the value of neighborhood radius(eps): ")
eps = float(eps)
min_pts=input("Enter the minimum points in neighborhood: ")
min_pts = int(min_pts)
c = dbScan(points, eps, min_pts, dist_mat, cluster, visited)
jaccard, rand = calcExtIndex(cluster)

generatePlot(cluster,c)

print("JACCARD = ",jaccard)
print("RAND = ",rand)

