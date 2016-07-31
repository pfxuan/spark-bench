import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Writable;

import java.util.Arrays;
import java.util.stream.Collectors;

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
}
