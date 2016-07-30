package kmeans_min.src.main.scala

import org.apache.hadoop.io.{ArrayWritable, DoubleWritable}

/**
  * Created by pxuan on 6/26/16.
  */
class DoubleArrayWritable extends ArrayWritable(classOf[DoubleWritable])