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
  
  // Function to read the data and choose the inital clusters from all the points
  public static void readData(String filename){
    try {
      BufferedReader br = new BufferedReader(new FileReader(filename));
      String sCurrentLine;
      while ((sCurrentLine = br.readLine()) != null) {
	//System.out.println(sCurrentLine);
	String[] line_split = sCurrentLine.split("\t"); 
	ArrayList<Float> row = new ArrayList<Float>();
	ground_truth.add(Float.valueOf(line_split[1]));
	for(int j=2; j<line_split.length; j++){
	  row.add(Float.valueOf(line_split[j].trim()));
	  //System.out.print(row.get(j-2)+" ");
	}
	//System.out.println();
	points.add(row);
      }
      no_genes = points.size();
      //points.add(row);
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
  
  //Generate the distance matrix
  public static void generateDist(){
    for(int i=0; i<no_genes; i++){
      ArrayList<Float> temp2 = new ArrayList<Float>();
      for(int j=0; j<k; j++){
	temp2.add(j,euclid(points.get(i),centroid.get(j)));
      }
      dist.add(i,temp2);
    }
  }
  
  //Calculate the euclidean distance between two points
  public static float euclid(ArrayList<Float> a,ArrayList<Float> b){
    double sum = 0;
    for(int i=0; i<a.size(); i++){
      sum = sum + Math.pow(a.get(i)-b.get(i),2);
    }
    return (float)Math.sqrt(sum);
  }
  
  //Assign cluster to a gene
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
  
  public static double jaccard(){
    //ArrayList<ArrayList<Integer>> cluster_matrix = new ArrayList<ArrayList<Integer>>();
    //ArrayList<ArrayList<Integer>> truth_matrix = new ArrayList<ArrayList<Integer>>();
    int[][] cluster_matrix = new int[no_genes][no_genes];
    int[][] truth_matrix = new int[no_genes][no_genes];
    for(int i=0; i<points.size(); i++){
      for(int j=0; j<points.size();j++){
	if(Float.compare(cluster_map.get(i),cluster_map.get(j)) == 0){
	  cluster_matrix[i][j] = 1;
	}
      }
    }
    for(int i=0; i<points.size(); i++){
      for(int j=0; j<points.size();j++){
	if(Float.compare(ground_truth.get(i),ground_truth.get(j))==0){
	  truth_matrix[i][j] = 1;
	}
      }
    }
    
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
    return (agree_disagree[0][0]*1.0)/(agree_disagree[0][0]+agree_disagree[1][0]+agree_disagree[0][1]);
  }
  
  public static class ClusterMapper
       extends Mapper<Object, Text, IntWritable, IntWritable>{

    private Text word = new Text();

    public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {
      String row = value.toString();
      String[] row_split = row.split("\t");
      ArrayList<Float> temp = new ArrayList<Float>();
      for(int j=2; j<row_split.length; j++){
	temp.add(Float.valueOf(row_split[j]));
      }
      int cluster = assignCluster(temp);
      IntWritable cluster_no = new IntWritable(cluster);
      int index = points.indexOf(temp);
      IntWritable id = new IntWritable(index);
      context.write(cluster_no,id);
    }
   }
  
  public static class ClusterReducer
	extends Reducer<IntWritable,IntWritable,IntWritable,Text> {
    private Text result = new Text();
    
    public void reduce(IntWritable key, Iterable<IntWritable> values, Context context
		      ) throws IOException, InterruptedException {
      //StringBuffer res = new StringBuffer("");
      ArrayList<Float> temp_centroid = new ArrayList<Float>();
      for(int i=0; i<points.get(0).size();i++){
	temp_centroid.add(i,(float)0);
      }
      int count = 0;
      for(IntWritable val:values){
	for(int i=0; i<points.get(val.get()).size();i++){
	  temp_centroid.set(i,temp_centroid.get(i)+points.get(val.get()).get(i));
	}
	count++;
      }
      for(int i=0; i<temp_centroid.size(); i++){
	temp_centroid.set(i,temp_centroid.get(i)/count);
      }
      new_centroid.add(key.get(),temp_centroid);
      result.set(temp_centroid.toString());
      context.write(key,result);
    }
  }
  
  public static void main(String[] args) throws Exception {
    String filepath = args[0];
    k = Integer.valueOf(args[2]);
    readData(filepath);
    while(true){
    //System.out.println("---------------------------------------------------------------------------------------------------------------------------------");
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
      if(hdfs.exists(output)){
	hdfs.delete(output,true);
      }
      FileOutputFormat.setOutputPath(job, output);
      job.waitForCompletion(true);
      if(centroid.equals(new_centroid)){
	System.out.println(jaccard());
	break;
      }
      else{
	centroid = new ArrayList<ArrayList<Float>>(new_centroid);
	new_centroid.clear();
	cluster_map.clear();
      }
    }
  }
}