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
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.mllib.util.KMeansDataGenerator
import org.apache.spark.sql.SparkSession

object KmeansGenML {
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN);
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF);
    if (args.length < 5) {
      println("usage: <output> <numPoints> <numClusters> <dimenstion> <scaling factor> [numpar]")
      System.exit(0)
    }
    // Creates a SparkSession.
    val spark = SparkSession
      .builder
      .appName(s"${this.getClass.getSimpleName}")
      .getOrCreate()
    import spark.implicits._

    val output = args(0)
    val numPoint = args(1).toInt
    val numCluster = args(2).toInt
    val numDim = args(3).toInt
    val scaling = args(4).toDouble
    val defPar = if (System.getProperty("spark.default.parallelism") == null) 2 else System.getProperty("spark.default.parallelism").toInt
    val numPar = if (args.length > 5) args(5).toInt else defPar

    val data = KMeansDataGenerator.generateKMeansRDD(spark.sparkContext, numPoint, numCluster, numDim, scaling, numPar)
    data.map(new Tuple1(_)).toDF("features").write.parquet(output)
    //data.map(Tuple1.apply).toDF("features").write.parquet(output)
    //spark.createDataFrame[Array[Double]](data).toDF().write.parquet(output)

    spark.stop();
  }
}
