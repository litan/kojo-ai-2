// #include /nn.kojo
// #include /plot.kojo

cleari()
clearOutput()

val m = 10
val c = 3
val xData = Array.tabulate(20)(e => (e + 1.0))
val yData = xData map (_ * m + c + randomDouble(-2.5, 2.5))

val chart = scatterChart("Regression Data", "X", "Y", xData, yData)
chart.getStyler.setLegendVisible(true)
drawChart(chart)

val xDataf = xData.map(_.toFloat)
val yDataf = yData.map(_.toFloat)

val model = new Model
model.train(xDataf, yDataf)
val yPreds = model.predict(xDataf)
addLineToChart(chart, Some("model"), xData, yPreds.map(_.toDouble))
drawChart(chart)
model.close()

class Model {
    val LEARNING_RATE: Float = 0.1f
    val WEIGHT_VARIABLE_NAME: String = "weight"
    val BIAS_VARIABLE_NAME: String = "bias"

    val graph = new Graph
    val tf = Ops.create(graph)
    val session = new Session(graph)

    // Define variables
    val weight = tf.variable(tf.constant(1f))
    val bias = tf.variable(tf.constant(1f))

    def train(xValues: Array[Float], yValues: Array[Float]): Unit = {
        val N = xValues.length
        // Define placeholders
        val xData = tf.placeholder(classOf[TFloat32], Placeholder.shape(Shape.scalar))
        val yData = tf.placeholder(classOf[TFloat32], Placeholder.shape(Shape.scalar))

        // Define the model function weight*x + bias
        val mul = tf.math.mul(xData, weight)
        val yPredicted = tf.math.add(mul, bias)

        // Define loss function MSE
        val sum = tf.math.pow(tf.math.sub(yPredicted, yData), tf.constant(2f))
        val mse = tf.math.div(sum, tf.constant(2f * N))

        // Back-propagate gradients to variables for training
        val optimizer = new GradientDescent(graph, LEARNING_RATE)
        val minimize = optimizer.minimize(mse)

        // Train the model on data
        for (epoch <- 1 to 150) {
            for (i <- xValues.indices) {
                val y = yValues(i)
                val x = xValues(i)
                val xTensor = TFloat32.scalarOf(x)
                val yTensor = TFloat32.scalarOf(y)
                session.runner
                    .addTarget(minimize)
                    .feed(xData.asOutput, xTensor)
                    .feed(yData.asOutput, yTensor)
                    .run
                xTensor.close(); yTensor.close()
            }
        }

        val wb = session.runner.fetch(weight).fetch(bias).run
        val weightValue = wb.get(0).asInstanceOf[TFloat32]
        val biasValue = wb.get(1).asInstanceOf[TFloat32]

        println("Weight is " + weightValue.getFloat())
        println("Bias is " + biasValue.getFloat())
    }

    def predict(xValues: Array[Float]): Array[Float] = {
        // Define placeholders
        val xData = tf.placeholder(classOf[TFloat32], Placeholder.shape(Shape.scalar))

        // Define the model function weight*x + bias
        val mul = tf.math.mul(xData, weight)
        val yPredicted = tf.math.add(mul, bias)

        xValues.map { x =>
            val xTensor = TFloat32.scalarOf(x)
            val yPredictedTensor = session.runner
            .feed(xData.asOutput, xTensor)
            .fetch(yPredicted)
            .run
            .get(0).asInstanceOf[TFloat32]
            
            val predictedY = yPredictedTensor.getFloat()
            xTensor.close(); yPredictedTensor.close()
            predictedY
        }
    }

    def close() {
        session.close()
        graph.close()
    }
}