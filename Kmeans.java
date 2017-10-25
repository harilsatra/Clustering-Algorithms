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

public class Kmeans{
  // Initialize all the class variables
  public static ArrayList<ArrayList<Float>> points = new ArrayList<ArrayList<Float>>();
  public static ArrayList<ArrayList<Float>> centroid = new ArrayList<ArrayList<Float>>();
  public static ArrayList<ArrayList<Float>> new_centroid = new ArrayList<ArrayList<Float>>();
  public static ArrayList<Float> ground_truth = new ArrayList<Float>();
  public static ArrayList<Float> cluster_map = new ArrayList<Float>();
  public static ArrayList<ArrayList<Float>> dist = new ArrayList<ArrayList<Float>>();
  public static int k = 0;
  public static int no_genes = 0;
  
  // Function to read the data and choose the inital clusters from all the points.
  public static void readData(String filepath){
    try {
      // Read data from the file and generate the points matrix.
      Path pt = new Path(filepath);
      FileSystem fs = FileSystem.get(new Configuration());
      BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(pt)));
      String sCurrentLine;
      while ((sCurrentLine = br.readLine()) != null) {
         String[] line_split = sCurrentLine.split("\t"); 
         ArrayList<Float> row = new ArrayList<Float>();
         ground_truth.add(Float.valueOf(line_split[1]));
         for(int j=2; j<line_split.length; j++){
           row.add(Float.valueOf(line_split[j].trim()));
         }
         points.add(row);
      }
      no_genes = points.size();
      // Randomly select k points from the data as the initial clusters.
      Random rand = new Random();
      HashSet<Integer> init_clust = new HashSet<Integer>();
      while(init_clust.size()<k){
        int no_random = rand.nextInt(no_genes);
        if(!init_clust.contains(no_random)){
          centroid.add(points.get(no_random));
        }
        init_clust.add(no_random);
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
  
  //Generate the distance matrix between each point and all of the k clusters (shape:(no_genes,k))
  public static void generateDist(){
    for(int i=0; i<no_genes; i++){
      ArrayList<Float> temp2 = new ArrayList<Float>();
      for(int j=0; j<k; j++){
        temp2.add(j,euclid(points.get(i),centroid.get(j)));
      }
      dist.add(i,temp2);
    }
  }
  
  //Calculate the euclidean distance between two points a and b
  public static float euclid(ArrayList<Float> a,ArrayList<Float> b){
    double sum = 0;
    for(int i=0; i<a.size(); i++){
      sum = sum + Math.pow(a.get(i)-b.get(i),2);
    }
    return (float)Math.sqrt(sum);
  }
  
  //Assign cluster to a gene by finding the argmin in the corresponding row of the distance matrix.
  public static int assignCluster(ArrayList<Float> row){
    float min = Float.MAX_VALUE;
    int id = points.indexOf(row);
    int index = -1;
    if(id != -1){
      for(int i=0; i<k; i++){
        if(dist.get(id).get(i) < min){
          min = dist.get(id).get(i);
          index = i;
        }
      }
      cluster_map.add(id,(float)index);
    }
    return index;
  }
  
  //Function to calculate the jaccard index.
  public static double jaccard(){
    int[][] cluster_matrix = new int[no_genes][no_genes];
    int[][] truth_matrix = new int[no_genes][no_genes];

    // Populate the cluster_matrix which contains information whether each point is in the same cluster with all the points based on the generated clusters.
    for(int i=0; i<points.size(); i++){
      for(int j=0; j<points.size();j++){
        if(Float.compare(cluster_map.get(i),cluster_map.get(j)) == 0){
          cluster_matrix[i][j] = 1;
        }
      }
    }

    // Populate the truth_matrix which contains information whether each point is in the same cluster with all the points based on the true clusters.
    for(int i=0; i<points.size(); i++){
      for(int j=0; j<points.size();j++){
        if(Float.compare(ground_truth.get(i),ground_truth.get(j))==0){
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
    // jaccard = M11 / (M11 + M01 + M10)
    return (agree_disagree[0][0]*1.0)/(agree_disagree[0][0]+agree_disagree[1][0]+agree_disagree[0][1]);
  }
  
  // The Mapper class which contains the map function
  public static class ClusterMapper
       extends Mapper<Object, Text, IntWritable, IntWritable>{

    private Text word = new Text();

    // The map function which takes each row of the input file as the input.
    public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {
      // Store the input of the map funtion which will be a single line
      String row = value.toString();
      // Split this line into tokens
      String[] row_split = row.split("\t");
      // Extract the attributes/dimensions of the input point.
      ArrayList<Float> temp = new ArrayList<Float>();
      for(int j=2; j<row_split.length; j++){
        temp.add(Float.valueOf(row_split[j]));
      }
      // Find the cluster to which this point belongs
      int cluster = assignCluster(temp);

      //Emit (Cluster_id,Gene_id) to the reducer
      IntWritable cluster_no = new IntWritable(cluster);
      int index = points.indexOf(temp);
      IntWritable id = new IntWritable(index);
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
      // Initialize the new centroid of the cluster to 0.
      ArrayList<Float> temp_centroid = new ArrayList<Float>();
      for(int i=0; i<points.get(0).size();i++){
        temp_centroid.add(i,(float)0);
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
      new_centroid.add(key.get(),temp_centroid);
      result.set(temp_centroid.toString());

      // Emit (cluster_id,corresponding_centroid)
      context.write(key,result);
    }
  }
  
  public static void main(String[] args) throws Exception {
    String filepath = args[0];
    // Set the value of k as per the input.
    k = Integer.valueOf(args[2]);
    // Read the file once and populate the data structures. Also select the initial centroids.
    readData(filepath);
    // Loop till the centroids do not converge.
    while(true){
      // Generate the dist matrix of all the genes with the cluster centroids.
      generateDist();
      Configuration conf = new Configuration();
      FileSystem hdfs = FileSystem.get(conf);
      Path output = new Path(args[1]);
      Job job = Job.getInstance(conf, "kmeans");
      job.setJarByClass(Kmeans.class);
      job.setMapperClass(ClusterMapper.class);
      //job.setCombinerClass(ClusterReducer.class);
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

      // Check if the old centroids are the same as the new computed centroids.
      if(centroid.equals(new_centroid)){
        // If they are the same, calculate the jaccard index and break out of the infinite loop.
        System.out.println("JACCARD: "+jaccard());
        System.out.println();
        // Display the Cluster id, Centroid and all the genes in that cluster. Do this for all the final clusters.
        for(int i=0; i<k; i++){
          System.out.println("CLUSTER ID: "+i);
          System.out. println("CENTROID: "+new_centroid.get(i).toString());
          for(int j=0; j<cluster_map.size(); j++){
            if(Float.compare(cluster_map.get(j),(float)i)==0){
              System.out.print(j+" ");
            }
          }
          System.out.println("\n");
        }
        break;
      }
      else{ // else update the centroids with new centroids and continue with the lopp.
        centroid = new ArrayList<ArrayList<Float>>(new_centroid);
        new_centroid.clear();
        cluster_map.clear();
      }
    }
  }
}