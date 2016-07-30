import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.DoubleWritable;

/**
 * Created by pxuan on 7/30/16.
 */
public class DoubleArrayWritable extends ArrayWritable {

  public DoubleArrayWritable(Class<DoubleWritable> valueClass) {
    super(valueClass);
  }
}
