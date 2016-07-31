import org.apache.hadoop.io.{ArrayWritable, DoubleWritable}

class DoubleArrayWritable extends ArrayWritable(classOf[DoubleWritable]) {

  def toDoubleArray(): Array[Double] = get().map(e => e.asInstanceOf[DoubleWritable].get())

}