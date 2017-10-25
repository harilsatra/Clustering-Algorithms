
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys

def generateClusters(dist_mat, clusters, no_clusters):
    
    while(no_clusters != len(clusters)):
        cluster1 = -1
        cluster2 = -1
        single_link = sys.maxsize
        for i in range(0,len(dist_mat)):
            for j in range(i+1,len(dist_mat)):
                if single_link > dist_mat[i][j]:
                    single_link = dist_mat[i][j]
                    cluster1 = i
                    cluster2 = j
                    
        for i in range((cluster1+1),len(dist_mat)):
            dist_mat[cluster1][i] = min(dist_mat[cluster1][i],dist_mat[cluster2][i])
            
        for i in range(0,cluster1):
            dist_mat[i][cluster1] = min(dist_mat[i][cluster1],dist_mat[i][cluster2])
            
        dist_mat = np.delete(dist_mat,cluster2,0)    
        dist_mat = np.delete(dist_mat,cluster2,1)     
        clusters[cluster1].extend(clusters[cluster2])
        clusters.pop(cluster2)
       

def calcExtIndex(clusters, ground_truth):

    clusterList = np.zeros(no_genes)
    
    for i in range(0, len(clusters)):
        for j in clusters[i]:
            clusterList[j] = i+1
    
    cluster_matrix = np.zeros((no_genes,no_genes))
    
    for i in range(0,no_genes):
        for j in range(0,no_genes):
            if clusterList[i] == clusterList[j]:
                cluster_matrix[i][j] = 1
                              
    truth_matrix = np.zeros((no_genes,no_genes))
    for i in range(0,no_genes):
        for j in range(0,no_genes):
            if ground_truth[i] == ground_truth[j]:
                truth_matrix[i][j] = 1
                    
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
    
    return jaccard,rand, clusterList


def generatePlot(clusterList):
    pca = PCA(n_components=2)
    pca.fit(points)
    reducedPoints = pca.transform(points)
    
    fig_size = plt.rcParams["figure.figsize"]
     
    # Set figure width to 12 and height to 9
    fig_size[0] = 12
    fig_size[1] = 9
    plt.rcParams["figure.figsize"] = fig_size
    
    plt.scatter(reducedPoints[:, 0], reducedPoints[:, 1], c=clusterList, alpha=0.5)
    plt.title('Hierarchical Agglomerative Clustering')
    plt.grid(True)
    plt.show()
    
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

no_clusters = 5

clusters = []

for i in range(0, no_genes):
    tempList = []
    tempList.append(i)
    clusters.append(tempList)
    
generateClusters(dist_mat, clusters, no_clusters)

jaccard, rand, clusterList = calcExtIndex(clusters, ground_truth)


generatePlot(clusterList)

print("Jaccard = ",jaccard)
print("Rand = ",rand)
