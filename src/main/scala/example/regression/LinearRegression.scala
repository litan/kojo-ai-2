package example.regression

import org.tensorflow.framework.optimizers.GradientDescent
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import org.tensorflow.types.TFloat32
import org.tensorflow.{Graph, Session}

import java.util.Random
import scala.util.Using

object LinearRegression {
  /**
   * Amount of data points.
   */
  private val N: Int = 10

  /**
   * This value is used to fill the Y placeholder in prediction.
   */
  val LEARNING_RATE: Float = 0.1f
  val WEIGHT_VARIABLE_NAME: String = "weight"
  val BIAS_VARIABLE_NAME: String = "bias"

  def main(args: Array[String]): Unit = { // Prepare the data
    val xValues = Array(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f)
    val yValues = new Array[Float](N)
    val rnd = new Random(42)
    for (i <- yValues.indices) {
      yValues(i) = (10 * xValues(i) + 2 + 0.1 * (rnd.nextDouble - 0.5)).toFloat
    }
    Using(new Graph) { graph =>
      val tf = Ops.create(graph)
      // Define placeholders
      val xData = tf.placeholder(classOf[TFloat32], Placeholder.shape(Shape.scalar))
      val yData = tf.placeholder(classOf[TFloat32], Placeholder.shape(Shape.scalar))

      // Define variables
      val weight = tf.withName(WEIGHT_VARIABLE_NAME).variable(tf.constant(1f))
      val bias = tf.withName(BIAS_VARIABLE_NAME).variable(tf.constant(1f))

      // Define the model function weight*x + bias
      val mul = tf.math.mul(xData, weight)
      val yPredicted = tf.math.add(mul, bias)

      // Define loss function MSE
      val sum = tf.math.pow(tf.math.sub(yPredicted, yData), tf.constant(2f))
      val mse = tf.math.div(sum, tf.constant(2f * N))

      // Back-propagate gradients to variables for training
      val optimizer = new GradientDescent(graph, LEARNING_RATE)
      val minimize = optimizer.minimize(mse)

      Using(new Session(graph)) { session =>
        // Initialize graph variables
        session.run(tf.init)
        // Train the model on data
        for (i <- xValues.indices) {
          val y = yValues(i)
          val x = xValues(i)
          val xTensor = TFloat32.scalarOf(x)
          val yTensor = TFloat32.scalarOf(y)
          session.runner.addTarget(minimize).feed(xData.asOutput, xTensor).feed(yData.asOutput, yTensor).run
          System.out.println("Training phase")
          System.out.println("x is " + x + " y is " + y)
        }

        // Extract linear regression model weight and bias values
        val tensorList = session.runner.fetch(WEIGHT_VARIABLE_NAME).fetch(BIAS_VARIABLE_NAME).run
        val weightValue = tensorList.get(0).asInstanceOf[TFloat32]
        val biasValue = tensorList.get(1).asInstanceOf[TFloat32]
        System.out.println("Weight is " + weightValue.getFloat())
        System.out.println("Bias is " + biasValue.getFloat())

        // Let's predict y for x = 10f
        val x = 10f
        var predictedY = 0f
        val xTensor = TFloat32.scalarOf(x)
        val yPredictedTensor = session.runner.feed(xData.asOutput, xTensor).fetch(yPredicted).run.get(0).asInstanceOf[TFloat32]
        predictedY = yPredictedTensor.getFloat()
        xTensor.close(); yPredictedTensor.close()
        System.out.println("Predicted value: " + predictedY)
      }
    }
  }
}
