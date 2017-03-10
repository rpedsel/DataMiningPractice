// Two phase matrix multiplication in Hadoop MapReduce
// Template file for homework #1 - INF 553 - Spring 2017
// - Wensheng Wu

import java.io.IOException;

// add your import statement here if needed
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.AbstractMap;
import java.util.AbstractMap.SimpleEntry;
// you can only import packages from java.*;

import org.apache.hadoop.conf.Configuration;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.LongWritable;

import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class TwoPhase {

    // mapper for processing entries of matrix A
    public static class PhaseOneMapperA 
	extends Mapper<LongWritable, Text, Text, Text> {
	
	private Text outKey = new Text();
	private Text outVal = new Text();

	public void map(LongWritable key, Text value, Context context)
	    throws IOException, InterruptedException {
	    
	    // fill in your code
	    // A(I,K,V)
	    String[] parts = value.toString().split(",");
	    outKey.set(parts[1]);
	    String s = "A,"+parts[0]+","+parts[2];
	    outVal.set(s);
	    context.write(outKey,outVal);
	}
    }

    // mapper for processing entries of matrix B
    public static class PhaseOneMapperB
	extends Mapper<LongWritable, Text, Text, Text> {
	
	private Text outKey = new Text();
	private Text outVal = new Text();

	public void map(LongWritable key, Text value, Context context)
	    throws IOException, InterruptedException {
	    
	    // fill in your code
	    // B(K,j,W)
	    String[] parts = value.toString().split(",");
	    outKey.set(parts[0]);
	    String s = "B,"+parts[1]+","+parts[2];
	    outVal.set(s);
	    context.write(outKey,outVal);
	}
    }

    public static class PhaseOneReducer
	extends Reducer<Text, Text, Text, Text> {

	private Text outKey = new Text();
	private Text outVal = new Text();

	public void reduce(Text key, Iterable<Text> values, Context context) 
	    throws IOException, InterruptedException {
	    
	    // fill in your code
	    List<Map.Entry<String,Integer>> arA = new ArrayList<>();
            List<Map.Entry<String,Integer>> arB = new ArrayList<>();
	    for (Text val : values){
	        String[] parts = val.toString().split(",");
		int vorw = Integer.parseInt(parts[2]);
		Map.Entry<String,Integer> pair=new SimpleEntry<>(parts[1],vorw);
		if (parts[0].equals("A")){
		    arA.add(pair);
		}
	        else
		    arB.add(pair);
	    }
	    for (Map.Entry<String,Integer> a : arA){
	    	for (Map.Entry<String,Integer> b : arB){
		    String s = a.getKey()+","+b.getKey();
		    outKey.set(s);
		    int vw = a.getValue()*b.getValue();
		    outVal.set(String.valueOf(vw));
		    context.write(outKey,outVal);
		}
	    }
	}
    }

    public static class PhaseTwoMapper 
	extends Mapper<Text, Text, Text, Text> {
	
	private Text outKey = new Text();
	private Text outVal = new Text();

	public void map(Text key, Text value, Context context)
	    throws IOException, InterruptedException {

	    // fill in your code
	    outKey.set(key);
	    outVal.set(value);
	    context.write(outKey,outVal);
	}
    }

    public static class PhaseTwoReducer 
	extends Reducer<Text, Text, Text, Text> {
	
	private Text outKey = new Text();
	private Text outVal = new Text();

	public void reduce(Text key, Iterable<Text> values, Context context)
	    throws IOException, InterruptedException {
	 
	    // fill in your code
	    int sum = 0;
	    for (Text val : values){
	    	sum += Integer.parseInt(val.toString());
     	    }
	    outKey.set(key);
	    outVal.set(String.valueOf(sum));
	    context.write(outKey,outVal);
	}
    }


    public static void main(String[] args) throws Exception {
	Configuration conf = new Configuration();

	Job jobOne = Job.getInstance(conf, "phase one");

	jobOne.setJarByClass(TwoPhase.class);

	jobOne.setOutputKeyClass(Text.class);
	jobOne.setOutputValueClass(Text.class);

	jobOne.setReducerClass(PhaseOneReducer.class);

	MultipleInputs.addInputPath(jobOne,
				    new Path(args[0]),
				    TextInputFormat.class,
				    PhaseOneMapperA.class);

	MultipleInputs.addInputPath(jobOne,
				    new Path(args[1]),
				    TextInputFormat.class,
				    PhaseOneMapperB.class);

	Path tempDir = new Path("temp");

	FileOutputFormat.setOutputPath(jobOne, tempDir);
	jobOne.waitForCompletion(true);


	// job two
	Job jobTwo = Job.getInstance(conf, "phase two");
	

	jobTwo.setJarByClass(TwoPhase.class);

	jobTwo.setOutputKeyClass(Text.class);
	jobTwo.setOutputValueClass(Text.class);

	jobTwo.setMapperClass(PhaseTwoMapper.class);
	jobTwo.setReducerClass(PhaseTwoReducer.class);

	jobTwo.setInputFormatClass(KeyValueTextInputFormat.class);

	FileInputFormat.setInputPaths(jobTwo, tempDir);
	FileOutputFormat.setOutputPath(jobTwo, new Path(args[2]));
	
	jobTwo.waitForCompletion(true);
	
	FileSystem.get(conf).delete(tempDir, true);
	
    }
}
