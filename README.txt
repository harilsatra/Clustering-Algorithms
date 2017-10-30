***********************
***  Dataset format ***
***********************

Each row represents a gene:
1) the first column is gene_id.
2) the second column is the ground truth clusters. You can compare it with your results. "-1" means outliers.
3) the rest columns represent gene's expression values (attributes).

******************************
****  K Means  Clustering ****
******************************

Steps to execute kmeans.py:
1) Save the kmeans.py file into a local directory.
2) Save the input file in the same directory.
3) Run the keamns.py file using an ide or through the console (python kmeans.py).
4) The script will prompt you to enter the file name. Enter the input file name.
5) The script will then prompt you to enter the number of clusters.
6) The script will then prompt you to enter the number of iterations.
7) The script will then prompt you to enter the initial cluster centers.
6) The jaccard index and rand index will be displayed and the plot of the cluster assignment will also be displayed.

**************************************************
****  Hierarchical Agglomerative Clustering   ****
**************************************************

Steps to execute hac.py:
1) Save the hac.py file into a local directory.
2) Save the input file in the same directory.
3) Run the hac.py file using an ide or through the console (python hac.py).
4) The script will prompt you to enter the file name. Enter the input file name.
5) The script will then prompt you to enter the number of clusters.
6) The jaccard index and rand index will be displayed and the plot of the cluster assignment will also be displayed.

******************************
****         DBSCAN 	  ****
******************************

Steps to execute dbscan:
1) Save the dbscan.py file into a local directory.
2) Save the input file in the same directory.
3) Run the dbscan.py file using an ide or through the console (python dbscan.py).
4) The script will prompt you to enter the file name. Enter the input file name.
5) The script will then prompt you to enter the eps (minimum radius of neighborhood) and the minimum nuber of points there should be in the neighborhood.
6) The jaccard index and rand index will be displayed and the plot of the cluster assignment will also be displayed.

**************************************************
****       Map-Reduce K Means Clustering 	  ****
**************************************************

Prerequisites:
1) Hadoop is installed and running.
	a. Command to start hadoop: start-hadoop.sh
	b. Command to stop hadoop: stop-hadoop.sh

Note: Make sure the ~/.bashrc file is updated with the paths of JAVA_HOME and HADOOP_CLASSPATH

Steps to execute the map-reduce kmeans:
1) Save the input file in. local directory
2) Make a new directory on hadfs for the input file:

	hdfs dfs -mkdir -p ~/input/

3) Copy the local input file to the hdfs directory input/:

	hdfs dfs -put <Enter path to the input file> ~/input

	Ex: If ~/genes/cho.txt is the path of the input file the command would be as follows:

	hdfs dfs -put ~/genes/cho.txt ~/input

4) Make sure you copied the input file to the right place:

	hdfs dfs -ls ~/input

5) Save the Kmeans.java file in a local directory.
6) Navigate to that directory on the terminal (Command line interface)
7) Run the following command to compile Kmeans.java:

	hadoop com.sun.tools.javac.Main Kmeans.java

	Note: If you get an error at this point, it is likely either your java version, JAVA_HOME, or HADOOP_CLASSPATH that aren’t configured correctly.

8) Run the following command to create a jar:

	jar cf kmeans.jar Kmeans*.class

9) Command to run the MR job:

	hadoop jar kmeans.jar Kmeans ~/input/<Enter input filename that you added in step 3> ~/output <Enter number of clusters>

	Ex: If the file that you added in step 3 is cho.txt and you want to find result for 5 clusters:

	hadoop jar kmeans.jar Kmeans ~/input/cho.txt ~/output 5

10) The jaccard index will be desplayed on the console along with the following information for each cluster:

	a. CLuster Id
	b. Centroid for that Cluster Id
	c. Genes that belong to that Cluster Id