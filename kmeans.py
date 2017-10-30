#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 18:10:47 2017

@author: harilsatra
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Function to check if the old centroids and new  centroids are equal
def isEqual(centroids,new_centroids):
    for i in range(len(centroids)):
        if np.linalg.norm(new_centroids[i]-centroids[i]) != 0:
            return False;
    return True;

# Generate Incidence Matrices from the ground truth and the cluster results that we get.
def calcExtIndex(clusters):
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

    return jaccard, rand


# Function to reduce the dimensions of the data using PCA and plot the clusters
def generatePlot(clusterList, count_clusters):
    pca = PCA(n_components=2)
    pca.fit(points)
    reducedPoints = pca.transform(points)
    
    labels = []
    for i in range(0,count_clusters):
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
    
    plt.title('K Means Clustering')
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

no_genes = len(input_data)
no_attr = np.shape(points)[1]

#Ask the user to input the number of clusters
k = input("Enter the number of clusters: ")
k = int(k)

#Ask the user to input the number of iterations
iterations = int(input("Enter the number of iterations: "))

#Ask the user to input the k initial centers
centroids = []
for i in range(k):
    initial_center = input("Enter the initial center no "+str(i+1)+": ")
    if int(initial_center) < 1 or int(initial_center) > no_genes:
        print()
        print("ERROR: Please enter in range 1 to "+str(no_genes))
        initial_center = input("Enter the initial center no "+str(i+1)+": ")
    centroids.append(points[int(initial_center)-1].tolist())

# Calculate new centroids and loop until new centroids are not equal to the old centroid
# or until the max number of iterations have been processed.
while iterations != 0:
    
    dist = np.zeros((no_genes,k))
    
    for i in range(0,no_genes):
        for j in range(0,k):
            dist[i][j] = np.linalg.norm(points[i]-centroids[j])
    
    clusters = np.argmin(dist,axis=1)
    #print([i for i in clusters])
    #print()
    new_centroids = np.zeros((k,no_attr))
    no_points = np.zeros(k)
    
    for i in range(0,no_genes):
        new_centroids[clusters[i]] += np.ravel(points[i])
        no_points[clusters[i]] += 1
        
    
    for i in range(0,k):
        for j in range(0,no_attr):
            new_centroids[i][j] /= no_points[i]
    
    if isEqual(centroids,new_centroids):
        break
    else:
        centroids = new_centroids
    iterations -= 1
    
# Print all the external indexs.
jaccard , rand = calcExtIndex(clusters)
generatePlot(clusters,k)
print("JACCARD: ", jaccard)
print("RAND: " ,rand)
#print("ITERATION: ",iteration)
    