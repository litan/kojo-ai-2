package example

import org.knowm.xchart.SwingWrapper
import net.kogics.kojo.plot._

object Plot {
  def main(args: Array[String]): Unit = {
    val xs = Array(1.0, 2, 3, 4)
    val ys = xs.map(e => 3 * e + 2)
    val xs2 = Array(1.0, 2, 3, 4)
    val ys2 = xs2 map (e => 4 * e + 1)
    val chart = scatterChart("A line chart", "xs", "ys", xs, ys)
    addPointsToChart(chart, Some("second line"), xs2, ys2)

    //    val xs = List("Maruti", "Renault", "Honda")
    //    val ys = List(10, 12, 9)
    //    //    val chart = barChart("Brand Volume", "Brand", "Volume", xs, ys)
    //    val chart = pieChart("Brand Volume", xs, ys)
    //
    //    val rgen = new util.Random()
    //    val xs = (1 to 1000).map(_ => rgen.nextGaussian).toArray
    //    val chart = histogram("Random normal", "Bins", "Counts", xs, 10)

    new SwingWrapper(chart).displayChart()
  }
}
