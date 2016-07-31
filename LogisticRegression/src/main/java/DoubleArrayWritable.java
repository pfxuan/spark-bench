import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Writable;

/**
 * Created by pxuan on 7/30/16.
 */
public class DoubleArrayWritable extends ArrayWritable {

  public DoubleArrayWritable() {
    super(DoubleWritable.class);
  }

  public DoubleArrayWritable(Double[] array) {
    super(DoubleWritable.class);
    DoubleWritable[] ret = new DoubleWritable[array.length];
    for (int i = 0; i < array.length; i++) {
      ret[i] = new DoubleWritable(array[i]);
    }
    this.set(ret);
  }

  public double[] toPrimitiveArray() {
    Writable[] records = get();
    double[] ret = new double[records.length];
    for (int i = 0; i < records.length; i++) {
      ret[i] = ((DoubleWritable) records[i]).get();
    }
    return ret;
  }
}
