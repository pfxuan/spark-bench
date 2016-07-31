/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * @author minli
 */

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.log4j.Logger;
import org.apache.log4j.Level;

import java.util.Arrays;

import scala.Tuple2;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
//import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.SparkConf;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;
import org.apache.log4j.Logger;
import org.apache.log4j.Level;
import org.apache.spark.storage.StorageLevel;


public class LogisticRegressionApp {

  public static void main(String[] args) {
    if (args.length < 3) {
      System.out.println("usage: <input> <output>  <maxIterations> <StorageLevel> ");
      System.exit(0);
    }
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN);
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF);
    String input = args[0];
    String output = args[1];
    int numIterations = Integer.parseInt(args[2]);
    String sl = args[3];

    SparkConf conf = new SparkConf().setAppName("LogisticRegressionApp Example");
    JavaSparkContext sc = new JavaSparkContext(conf);
    conf.registerKryoClasses(new Class<?>[]{LogisticRegressionApp.class});

    // Load and parse data
    long start = System.currentTimeMillis();
    JavaPairRDD<DoubleWritable, DoubleArrayWritable> data = sc.newAPIHadoopFile(input, SequenceFileInputFormat.class, DoubleWritable.class, DoubleArrayWritable.class, new Configuration());
    JavaRDD<LabeledPoint> parsedData = data.map(r -> new LabeledPoint(r._1.get(), Vectors.dense(r._2.toPrimitiveArray())));
    RDD<LabeledPoint> parsedRDD_Data = JavaRDD.toRDD(parsedData);

    if (sl.equals("MEMORY_AND_DISK_SER"))
      parsedRDD_Data.persist(StorageLevel.MEMORY_AND_DISK_SER());
    else if (sl.equals("MEMORY_ONLY"))
      parsedRDD_Data.persist(StorageLevel.MEMORY_ONLY());
    else
      parsedRDD_Data.persist(StorageLevel.MEMORY_AND_DISK());

    double loadTime = (double) (System.currentTimeMillis() - start) / 1000.0;

    // Building the model
    start = System.currentTimeMillis();
    final LogisticRegressionModel model
        = LogisticRegressionWithSGD.train(parsedRDD_Data, numIterations);
    double trainingTime = (double) (System.currentTimeMillis() - start) / 1000.0;

    // Evaluate model on training examples and compute training error
    start = System.currentTimeMillis();
    JavaRDD<Tuple2<Double, Double>> valuesAndPreds = parsedData.map(
        new Function<LabeledPoint, Tuple2<Double, Double>>() {
          public Tuple2<Double, Double> call(LabeledPoint point) {
            double prediction = model.predict(point.features());
            return new Tuple2<Double, Double>(prediction, point.label());
          }
        }
    );
    Double MSE = new JavaDoubleRDD(valuesAndPreds.map(
        new Function<Tuple2<Double, Double>, Object>() {
          public Object call(Tuple2<Double, Double> pair) {
            return Math.pow(pair._1() - pair._2(), 2.0);
          }
        }
    ).rdd()).mean();
    double testTime = (double) (System.currentTimeMillis() - start) / 1000.0;

    System.out.printf("{\"loadTime\":%.3f,\"trainingTime\":%.3f,\"testTime\":%.3f}\n", loadTime, trainingTime, testTime);
    System.out.println("training Mean Squared Error = " + MSE);

    sc.stop();
  }
}
