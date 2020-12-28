package example.regression

import net.kogics.kojo.plot._
import net.kogics.kojo.preprocess.StandardScaler
import org.knowm.xchart.SwingWrapper
import org.tensorflow.framework.optimizers.GradientDescent
import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.buffer.DataBuffers
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import org.tensorflow.types.TFloat32
import org.tensorflow.{Graph, Session}

object LinearRegression3 {

  def main(args: Array[String]): Unit = {
    val m = 3
    val c = 10
    val xData0 = Array.tabulate(20)(e => (e + 1.0))
    val yData0 = xData0 map (_ * m + c + math.random() * 10 - 5)
    val normalizer = new StandardScaler()

    val chart = scatterChart("Regression Data", "X", "Y", xData0, yData0)
    chart.getStyler.setLegendVisible(true)
    val chartWin = new SwingWrapper(chart).displayChart()

    val xData = normalizer.fitTransform(xData0)
    val yData = yData0

    val xDataf = xData.map(_.toFloat)
    val yDataf = yData.map(_.toFloat)

    val model = new Model
    model.train(xDataf, yDataf)
    val yPreds = model.predict(xDataf)
    addLineToChart(chart, Some("model"), xData0, yPreds.map(_.toDouble))
    chartWin.repaint()
  }

  class Model {
    val LEARNING_RATE: Float = 0.1f
    val WEIGHT_VARIABLE_NAME: String = "weight"
    val BIAS_VARIABLE_NAME: String = "bias"

    val graph = new Graph
    val tf = Ops.create(graph)
    val session = new Session(graph)

    // Define variables
    val weight = tf.withName(WEIGHT_VARIABLE_NAME).variable(tf.constant(1f))
    val bias = tf.withName(BIAS_VARIABLE_NAME).variable(tf.constant(1f))

    def train(xValues: Array[Float], yValues: Array[Float]): Unit = {
      val N = xValues.length
      // Define placeholders
      val xData = tf.placeholder(TFloat32.DTYPE, Placeholder.shape(Shape.of(-1, 1)))
      val yData = tf.placeholder(TFloat32.DTYPE, Placeholder.shape(Shape.of(-1, 1)))

      // Define the model function weight*x + bias
      val mul = tf.math.mul(xData, weight)
      val yPredicted = tf.math.add(mul, bias)

      // Define loss function MSE
      val sum = tf.math.pow(tf.math.sub(yPredicted, yData), tf.constant(2f))
      val mse = tf.math.div(sum, tf.constant(2f * N))

      // Back-propagate gradients to variables for training
      val optimizer = new GradientDescent(graph, LEARNING_RATE)
      val minimize = optimizer.minimize(mse)

      // Initialize graph variables
      session.run(tf.init)

      // Train the model on data
      for (_ <- 1 to 40) {
        val xTensor = TFloat32.tensorOf(Shape.of(N, 1), DataBuffers.of(xValues, true, false))
        val yTensor = TFloat32.tensorOf(Shape.of(N, 1), DataBuffers.of(yValues, true, false))
        session.runner.addTarget(minimize).feed(xData.asOutput, xTensor).feed(yData.asOutput, yTensor).run
        xTensor.close(); yTensor.close()
      }
    }

    def predict(xValues: Array[Float]): Array[Float] = {
      // Define placeholders
      val xData = tf.placeholder(TFloat32.DTYPE, Placeholder.shape(Shape.of(-1, 1)))

      // Define the model function weight*x + bias
      val mul = tf.math.mul(xData, weight)
      val yPredicted = tf.math.add(mul, bias)

      val xTensor = TFloat32.tensorOf(Shape.of(xValues.length, 1), DataBuffers.of(xValues, true, false))
      val yPredictedTensor = session.runner.feed(xData.asOutput, xTensor).fetch(yPredicted).run.get(0).expect(TFloat32.DTYPE)
      val predictedY = new Array[Float](xValues.length)
      val predictedYBuffer = yPredictedTensor.rawData().asFloats()
      predictedYBuffer.read(predictedY)
      predictedY
    }
  }
}
