
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys

# generate clusters by merging 2 closest clusters and updating the distance matrix until the desired number of clusters are formed
def generateClusters(dist_mat, clusters, no_clusters):
    
    while(no_clusters != len(clusters)):
        cluster1 = -1
        cluster2 = -1
        single_link = sys.maxsize
        # find the 2 clusters with minimun distance 
        for i in range(0,len(dist_mat)):
            for j in range(i+1,len(dist_mat)):
                if single_link > dist_mat[i][j]:
                    single_link = dist_mat[i][j]
                    cluster1 = i
                    cluster2 = j
        #update distance matrix using single link(min) approach            
        for i in range((cluster1+1),len(dist_mat)):
            dist_mat[cluster1][i] = min(dist_mat[cluster1][i],dist_mat[cluster2][i])
            
        for i in range(0,cluster1):
            dist_mat[i][cluster1] = min(dist_mat[i][cluster1],dist_mat[i][cluster2])
            
        dist_mat = np.delete(dist_mat,cluster2,0)    
        dist_mat = np.delete(dist_mat,cluster2,1)     
        
        #merge the 2 closest clusters
        clusters[cluster1].extend(clusters[cluster2])
        clusters.pop(cluster2)
       
# Generate Incidence Matrices from the ground truth and the cluster results that we get.
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

    """ 
    Generate the agree_disagree matrix as follows:
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
    
    return jaccard,rand, clusterList

# Function to reduce the dimensions of the data using PCA and plot the clusters
def generatePlot(clusterList, count_clusters):
    pca = PCA(n_components=2)
    pca.fit(points)
    reducedPoints = pca.transform(points)
    
    labels = []
    for i in range(1,count_clusters+1):
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
dist_mat = euclidean_distances(points,points)

no_clusters = input("Enter the no of clusters: ")
no_clusters = int(no_clusters)

clusters = []

for i in range(0, no_genes):
    tempList = []
    tempList.append(i)
    clusters.append(tempList)
    
generateClusters(dist_mat, clusters, no_clusters)

# Print all the external indexs.
jaccard, rand, clusterList = calcExtIndex(clusters, ground_truth)

generatePlot(clusterList,no_clusters)

print("Jaccard = ",jaccard)
print("Rand = ",rand)
