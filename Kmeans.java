import java.io.IOException;
import java.util.StringTokenizer;
import java.util.*;
import java.io.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class KMeans{
  // Initialize all the class variables
  public static ArrayList<ArrayList<Double>> points = new ArrayList<ArrayList<Double>>();
  public static ArrayList<ArrayList<Double>> centroid = new ArrayList<ArrayList<Double>>();
  public static ArrayList<ArrayList<Double>> dist = new ArrayList<ArrayList<Double>>();
  public static ArrayList<Integer> ground_truth = new ArrayList<Integer>();
  public static ArrayList<Integer> cluster_map = new ArrayList<Integer>();
  public static ArrayList<Integer> new_cluster_map = new ArrayList<Integer>();
  public static int k = 0;
  public static int no_genes = 0;
  
  // Function to read the data and choose the inital clusters from all the points.
  public static void readData(String filepath){
    try {
      Path pt = new Path(filepath);
      FileSystem fs = FileSystem.get(new Configuration());
      BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(pt)));
      String sCurrentLine;
      while ((sCurrentLine = br.readLine()) != null) {
	String[] line_split = sCurrentLine.split("\t"); 
	ArrayList<Double> row = new ArrayList<Double>();
	ground_truth.add(Integer.valueOf(line_split[1]));
	for(int j=2; j<line_split.length; j++){
	  row.add(Double.valueOf(line_split[j].trim()));
	}
         points.add(row);
         cluster_map.add(0);
      }
      no_genes = points.size();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
  
  // Randomly select k points from the data as the initial clusters.
  public static void generateInitialCenters(){
//  	int[] centroid_input = {68, 287, 300, 370, 403, 212, 221, 409, 93, 63};
//  	for(int i=0; i<k; i++){
//  		centroid.add(points.get(centroid_input[i]));
//  	}

     Random rand = new Random();
     HashSet<Integer> init_clust = new HashSet<Integer>();
     System.out.println("INITIAL CENTROIDS CHOSEN: ");
     while(init_clust.size()<k){
       int no_random = rand.nextInt(no_genes);
       if(!init_clust.contains(no_random)){
 	centroid.add(points.get(no_random));
 	System.out.println(no_random);
       }
       init_clust.add(no_random);
     }
  }
  
  //Generate the distance matrix between each point and all of the k clusters (shape:(no_genes,k))
  public static void generateDist(){
    for(int i=0; i<no_genes; i++){
      ArrayList<Double> temp2 = new ArrayList<Double>();
      for(int j=0; j<k; j++){
        temp2.add(j,euclid(points.get(i),centroid.get(j)));
      }
      dist.add(i,temp2);
    }
  }
  
  //Calculate the euclidean distance between two points a and b
  public static double euclid(ArrayList<Double> a,ArrayList<Double> b){
    double sum = 0;
    for(int i=0; i<a.size(); i++){
      sum = sum + Math.pow(a.get(i)-b.get(i),2);
    }
    return Math.sqrt(sum);
  }
  
  //Assign cluster to a gene by finding the argmin in the corresponding row of the distance matrix.
  public static int assignCluster(ArrayList<Double> row,int id){
    double min = Double.MAX_VALUE;
    int index = -1;
    for(int i=0; i<k; i++){
      if(dist.get(id).get(i) < min){
	min = dist.get(id).get(i);
	index = i;
      }
    }
    new_cluster_map.add(id,index);
      
    return index;
  }
  
  //Function to calculate the jaccard index.
  public static double jaccard(){
    int[][] cluster_matrix = new int[no_genes][no_genes];
    int[][] truth_matrix = new int[no_genes][no_genes];

    for(int i=0; i<points.size(); i++){
      for(int j=0; j<points.size();j++){
        if(Double.compare(cluster_map.get(i),cluster_map.get(j)) == 0){
          cluster_matrix[i][j] = 1;
        }
      }
    }

    for(int i=0; i<points.size(); i++){
      for(int j=0; j<points.size();j++){
        if(Double.compare(ground_truth.get(i),ground_truth.get(j))==0){
          truth_matrix[i][j] = 1;
        }
      }
    }
    /* Generate the agree_disagree matrix as follows:
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
    */
    int[][] agree_disagree = new int[2][2];
    for(int i=0; i<points.size(); i++){
      for(int j=0; j<points.size();j++){
        if(cluster_matrix[i][j] == truth_matrix[i][j]){
          if(cluster_matrix[i][j] == 1){
            agree_disagree[0][0] += 1; 
          }
          else{
            agree_disagree[1][1] += 1;
          }
        }
        else{
          if(cluster_matrix[i][j] == 1){
            agree_disagree[1][0] += 1; 
          }
          else{
            agree_disagree[0][1] += 1;
          }
        }
      }
    }
    return (double)(agree_disagree[0][0]*1.0)/(agree_disagree[0][0]+agree_disagree[1][0]+agree_disagree[0][1]);
  }
  
  // The Mapper class which contains the map function
  public static class ClusterMapper
       extends Mapper<Object, Text, IntWritable, IntWritable>{

    private Text word = new Text();
    
    // The map function which takes each row of the input file as the input.
    public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {

      String row = value.toString();

      String[] row_split = row.split("\t");

      ArrayList<Double> temp = new ArrayList<Double>();
      for(int j=2; j<row_split.length; j++){
        temp.add(Double.valueOf(row_split[j].trim()));
      }
      int gene_id = Integer.parseInt(row_split[0])-1;

      int cluster = assignCluster(temp,gene_id);
      IntWritable cluster_no = new IntWritable(cluster);
      int index = gene_id;
      IntWritable id = new IntWritable(index);
      //Emit (Cluster_id,Gene_id) to the reducer
      context.write(cluster_no,id);
    }
   }
   
   // The Reduce class which contains the reduce function
   public static class ClusterReducer
	extends Reducer<IntWritable,IntWritable,IntWritable,Text> {
    private Text result = new Text();
    
    // The reduce function which takes output emitted by the mapper as the input, combined as per the key on its way from the mapper.
    public void reduce(IntWritable key, Iterable<IntWritable> values, Context context
		      ) throws IOException, InterruptedException {
		      
      ArrayList<Double> temp_centroid = new ArrayList<Double>();
      for(int i=0; i<points.get(0).size();i++){
        temp_centroid.add(i,(double)0);
      }
      
      int count = 0;
      
      /* Iterate over the Iterable<IntWritable> which contains the ids of the gene which belong to the same cluster
      where the cluster id is the key of the key,value pair obtained by the reducer. While iterating update the new
      centroid of the cluster by adding individual dimensions.
      */
      for(IntWritable val:values){
        for(int i=0; i<points.get(val.get()).size();i++){
          temp_centroid.set(i,temp_centroid.get(i)+points.get(val.get()).get(i));
        }
        count++;
      }
      
      // Calculate the final centroid by dividing the sum of individual dimensions with the number of genes in that cluster.
      for(int i=0; i<temp_centroid.size(); i++){
        temp_centroid.set(i,temp_centroid.get(i)/count);
      }
      
      /* Add this new cluster centroid to the new_centroid data structure to later on compare it with
      the old centroids for convergence. */
      centroid.set(key.get(),temp_centroid);
      result.set(temp_centroid.toString());

      context.write(key,result);
    }
  }
  
  public static void main(String[] args) throws Exception{
    String filepath = args[0];
    // Set the value of k as per the input.
    k = Integer.valueOf(args[2]);
    // Read the file once and populate the data structures.
    readData(filepath);
    // Select the initial centroids.
    generateInitialCenters();
    int count = 1;
    // Loop until the cluser assignment does not change.
    while(true){
      // Generate the dist matrix of all the genes with the cluster centroids.
      generateDist();
      Configuration conf = new Configuration();
      FileSystem hdfs = FileSystem.get(conf);
      Path output = new Path(args[1]);
      Job job = Job.getInstance(conf, "kMeans");
      job.setJarByClass(KMeans.class);
      job.setMapperClass(ClusterMapper.class);
      job.setReducerClass(ClusterReducer.class);
      job.setOutputKeyClass(IntWritable.class);
      job.setOutputValueClass(IntWritable.class);
      FileInputFormat.addInputPath(job, new Path(args[0]));
      // Check if there is already an output folder generated.
      if(hdfs.exists(output)){
	// If the output directory already exists, delete it.
	hdfs.delete(output,true);
      }
      FileOutputFormat.setOutputPath(job, output);
      job.waitForCompletion(true);
      // Check if the cluster assignment does not change after new centroids are computed.
      if(cluster_map.equals(new_cluster_map)){
	// If it has not changed calculate and print the jaccard coefficient and break from the loop.
	System.out.println("JACCARD: "+jaccard());
	System.out.println();
	System.out.println("CENTROIDS: ");
	for(int i=0; i<k; i++){
	  System.out.println("Cluster "+(i+1));
	  System.out.println(centroid.get(i).toString());
	}
	PrintWriter writer = new PrintWriter("mapping.txt", "UTF-8");
        for(int i=0; i<cluster_map.size(); i++){
	  writer.println(i+" "+cluster_map.get(i));
        }
        writer.close();
	break;
      }
      else{
	// If the cluster assignment has changed update the old cluster map with the new one and continue.
	cluster_map = new ArrayList<Integer>(new_cluster_map);
	new_cluster_map.clear();
	dist.clear();
      }
    }
    
  }
}