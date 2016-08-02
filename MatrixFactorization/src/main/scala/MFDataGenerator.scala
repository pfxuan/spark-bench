/*
 * (C) Copyright IBM Corp. 2015 
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at 
 *
 *  http://www.apache.org/licenses/LICENSE-2.0 
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.apache.hadoop.io.{ArrayPrimitiveWritable, DoubleWritable}
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.language.postfixOps
import scala.util.Random
import org.jblas.DoubleMatrix
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

 /**
  *
  * Generate RDD(s) containing data for Matrix Factorization.
  *
  * This method samples training entries according to the oversampling factor
  * 'trainSampFact', which is a multiplicative factor of the number of
  * degrees of freedom of the matrix: rank*(m+n-rank).
  *
  * It optionally samples entries for a testing matrix using
  * 'testSampFact', the percentage of the number of training entries
  * to use for testing.
  *
  * This method takes the following inputs:
  * outputPath     (String) Directory to save output.
  * m              (Int) Number of rows in data matrix.
  * n              (Int) Number of columns in data matrix.
  * rank           (Int) Underlying rank of data matrix.
  * trainSampFact  (Double) Oversampling factor.
  * noise          (Boolean) Whether to add gaussian noise to training data.
  * sigma          (Double) Standard deviation of added gaussian noise.
  * test           (Boolean) Whether to create testing RDD.
  * testSampFact   (Double) Percentage of training data to use as test data.
  * numPar         (Int) Number of partitions of input data file
  */

object MFDataGenerator {
  def main(args: Array[String]) {
    if (args.length < 1) {
      println("Usage: MFDataGenerator " +
        "<outputDir> [m] [n] [rank] [trainSampFact] [noise] [sigma] [test] [testSampFact] [numPar]")
      System.exit(1)
    }
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val outputPath: String = args(0)
    val m: Int = if (args.length > 1) args(1).toInt else 100
    val n: Int = if (args.length > 2) args(2).toInt else 100
    val rank: Int = if (args.length > 3) args(3).toInt else 10
    val trainSampFact: Double = if (args.length > 4) args(4).toDouble else 1.0
    val noise: Boolean = if (args.length > 5) args(5).toBoolean else false
    val sigma: Double = if (args.length > 6) args(6).toDouble else 0.1
    val test: Boolean = if (args.length > 7) args(7).toBoolean else false
    val testSampFact: Double = if (args.length > 8) args(8).toDouble else 0.1
    val defPar = if (System.getProperty("spark.default.parallelism") == null) 2 else System.getProperty("spark.default.parallelism").toInt
    val numPar: Int = if (args.length > 9) args(9).toInt else defPar

    val conf = new SparkConf().setAppName("MFDataGenerator")
    val sc = new SparkContext(conf)

    val A = DoubleMatrix.randn(m, rank)
    val B = DoubleMatrix.randn(rank, n)
    val z = 1 / scala.math.sqrt(scala.math.sqrt(rank))
    A.mmuli(z)
    B.mmuli(z)
    val fullData = A.mmul(B)

    val df = rank * (m + n - rank)
    val sampSize = scala.math.min(scala.math.round(trainSampFact * df), scala.math.round(.99 * m * n)).toInt
    val rand = new Random()
    val mn = m * n

    val my_rdd = sc.makeRDD(1 to mn, numPar)
    my_rdd.cache()
    val trainData: RDD[(Int, Int, Double)] = my_rdd
      .map(x => (fullData.indexRows(x - 1), fullData.indexColumns(x - 1), fullData.get(x - 1)))

    // optionally add gaussian noise
    if (noise) {
      trainData.map(x => (x._1, x._2, x._3 + rand.nextGaussian * sigma))
    }
    //trainData.map(x => x._1 + "," + x._2 + "," + x._3).saveAsTextFile(outputPath)
    trainData.map(v => (new DoubleWritable(v._3), new ArrayPrimitiveWritable(Array(v._1, v._2))))

    sc.stop()
  }
}
