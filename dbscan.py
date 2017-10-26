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
# Function to begin density based scanning
def dbScan(points, eps, min_pts, dist_mat, cluster, visited):
    global noise
    c=0
    # Visit all the ponits that are not visited
    for i in range(0,len(points)):
        if visited[i][0] == 0:
            visited[i][0] = 1
            neighbors = regionQuery(i,eps,dist_mat,len(points))
            # If neighbors are less than minimum pts mark the point as noise
            if len(neighbors) < min_pts:
                noise.append(i)
            # If neighbors are greater than minimum pts mark the point into the next cluster.
            else:
                c = c+1
                expandCluster(i, neighbors, c, eps, min_pts, cluster, visited,no_genes)
    return c

# Function to expand the cluster           
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
               
# Function to return the neighbors of a point with radius as eps.
def regionQuery(point, eps, dist_mat,no_genes):
    neighbors = set()
    for i in range(0, no_genes):
        if dist_mat[point][i] <= eps:
            neighbors.add(i)
    return neighbors

# Function to generate the external index.
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
    
    """ Generate the agree_disagree matrix as follows:
    A pair of data object (Oi,Oj) falls into one of the following categories 
    M11 :  Cij=1 and Pij=1;  (agree)
    M00 : Cij=0 and Pij=0;   (agree)
    M10 :  Cij=1 and Pij=0;  (disagree)
    M01 :  Cij=0 and Pij=1;  (disagree)
    ----------------------------------------------------------------------------------
    |                                   |             CLUSTERING RESULT              |
    ----------------------------------------------------------------------------------
    |                                   | Same Cluster     |     Different Cluster   |
    ----------------------------------------------------------------------------------
    |              | Same Cluster       |      M11         |           M10           |
    | GROUND TRUTH |                    |                  |                         |
    |              | Different Cluster  |      M01         |           M00           |
    ----------------------------------------------------------------------------------
    """
    agree_disagree = np.zeros((2,2))
    
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

# Function to generate the plot
def generatePlot(clusterList, count_clusters):
    pca = PCA(n_components=2)
    pca.fit(points)
    reducedPoints = pca.transform(points)
    
    labels = []
    for i in range(0,count_clusters+1):
        labels.append(i)

    colors = [plt.cm.jet(float(i)/(max(labels))) for i in labels]
    
    for i, l in enumerate(labels):
       x = [reducedPoints[j][0] for j in range(len(reducedPoints)) if int(clusterList[j]) == (l)]
       y = [reducedPoints[j][1] for j in range(len(reducedPoints)) if int(clusterList[j]) == (l)]
       plt.plot(x, y,'wo', c= colors[i], label = str(l), markersize=9, alpha=0.75)
       
      
    
    fig_size = plt.rcParams["figure.figsize"]
    # Set figure width to 12 and height to 9
    fig_size[0] = 12
    fig_size[1] = 9
    plt.rcParams["figure.figsize"] = fig_size
    
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

# Extract the points from input
points = input_data[:,2:len(input_data[0])]
points = np.mat(points,dtype=float)

# Extract the ground truth from the input
ground_truth = input_data[:,1]

# Initialize all the data structures
no_genes = len(input_data)
no_attr = np.shape(points)[1]
cluster = np.zeros((no_genes,1))
visited = np.zeros((no_genes,1))
dist_mat = euclidean_distances(points,points)

# Take input parameters from the user
eps=input("Enter the value of neighborhood radius(eps): ")
eps = float(eps)
min_pts=input("Enter the minimum points in neighborhood: ")
min_pts = int(min_pts)

# Call the dbscan function to begin density based clustering
c = dbScan(points, eps, min_pts, dist_mat, cluster, visited)

# Call the calExtIndex() function to obtain the jaccard and rand index.
# The input to the function will be the cluster assigned to each gene id.
jaccard, rand = calcExtIndex(cluster)

# Call the generatePlot function to display the plot
# The input to the function will be the cluster assigned to each gene id and no of clusters.
generatePlot(cluster,c)

# Display the Jaccard and Rand index.
print("JACCARD = ",jaccard)
print("RAND = ",rand)