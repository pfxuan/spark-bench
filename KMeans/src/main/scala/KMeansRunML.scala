/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.SparkSession

/**
 * An example demonstrating k-means clustering.
 * Run with
 * {{{
 * bin/run-example ml.KMeansExample
 * }}}
 */
object KMeansRunML {

  def main(args: Array[String]): Unit = {
    if (args.length < 4) {
      println("usage: <input> <output> <numClusters> <maxIterations> <runs> - optional")
      System.exit(0)
    }
    val input = args(0)
    val output = args(1)
    val K = args(2).toInt
    val maxIterations = args(3).toInt

    // Creates a SparkSession.
    val spark = SparkSession
      .builder
      .appName(s"${this.getClass.getSimpleName}")
      .getOrCreate()

    // Loads data.
    val dataset = spark.read.parquet(input)

    // Trains a k-means model.
    val kmeans = new KMeans().setK(K).setMaxIter(maxIterations).setInitMode("k-means||").setSeed(127L)
    val model = kmeans.fit(dataset.map(new DenseVector(_)))

    // Evaluate clustering by computing Within Set Sum of Squared Errors.
    val WSSSE = model.computeCost(dataset)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    // Shows the result.
    // println("Cluster Centers: ")
    // model.clusterCenters.foreach(println)
    // $example off$

    spark.stop()
  }
}