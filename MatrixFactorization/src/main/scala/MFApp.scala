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

import org.apache.hadoop.io.{ArrayPrimitiveWritable, DoubleWritable}
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}

object MFApp {
  def main(args: Array[String]): Unit = {
    if (args.length < 5) {
      System.out.println("usage: <input> <output> <rank> <maxIterations> <lambda> <storageLevel>")
      System.exit(0)
    }
    val input: String = args(0)
    val output: String = args(1)
    val rank: Int = args(2).toInt
    val numIterations: Int = args(3).toInt
    val lambda: Double = args(4).toDouble
    val storage_level: String = args(5)

    val conf = new SparkConf().setAppName("Matrix FactorizationModel App")
    val sc = new SparkContext(conf)

    // Load and parse the data
    val data = sc.newAPIHadoopFile[DoubleWritable, ArrayPrimitiveWritable, SequenceFileInputFormat[DoubleWritable, ArrayPrimitiveWritable]](input)
    val ratings = data.map(s => new Rating(s._2.get().asInstanceOf[Array[Int]](0), s._2.get().asInstanceOf[Array[Int]](1), s._1.get()))
    // ratings.cache()

    // Build the recommendation model using ALS
    val model = ALS.train(ratings, rank, numIterations, lambda)

    // Evaluate the model on rating data
    val usersProducts = ratings.map { case Rating(user, product, rate) =>
      (user, product)
    }
    val predictions =
      model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }
    val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions)
    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()

    println("Mean Squared Error = " + MSE)

    // Save and load model
    // model.save(sc, "target/tmp/myCollaborativeFilter")
    // val sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")

    sc.stop()
  }
}