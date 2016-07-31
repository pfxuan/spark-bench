/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileAsBinaryOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.LinearDataGenerator;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;

/**
 * @author minli
 */
public class LinearRegressionDataGen {
  public static void main(String[] args) {
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF);
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN);
    if (args.length < 5) {
      System.out.println("usage: <output> <nExamples> <nFeatures> <eps> <intercept> <numPar>");
      System.exit(0);
    }
    String output = args[0];
    int nExamples = Integer.parseInt(args[1]);
    int nFeatures = Integer.parseInt(args[2]);
    double eps = Double.parseDouble(args[3]);
    double intercepts = Double.parseDouble(args[4]);
    int numPar = (args.length > 5) ? Integer.parseInt(args[5]) : System.getProperty("spark.default.parallelism") != null ? Integer.parseInt(System.getProperty("spark.default.parallelism")) : 2;

    SparkConf conf = new SparkConf().setAppName("Liner Regression data generation");
    SparkContext sc = new SparkContext(conf);
    RDD<LabeledPoint> data = LinearDataGenerator.generateLinearRDD(sc, nExamples,
        nFeatures, eps, numPar, intercepts);
    data.toJavaRDD()
        .mapToPair(r -> new Tuple2<>(new DoubleWritable(r.label()), new DoubleArrayWritable(ArrayUtils.toObject(r.features().toArray()))))
        .saveAsNewAPIHadoopFile(output, DoubleWritable.class, DoubleArrayWritable.class, SequenceFileOutputFormat.class);
  }
}
