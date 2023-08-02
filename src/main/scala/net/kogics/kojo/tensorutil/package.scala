package net.kogics.kojo

import java.awt.image.BufferedImage

import org.tensorflow.ndarray.buffer.DataBuffers
import org.tensorflow.ndarray.Shape
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TUint8

package object tensorutil {

  def imgToTensorF(image: BufferedImage, divFactor: Float): TFloat32 = {
    import java.nio.ByteBuffer
    val h = image.getHeight
    val w = image.getWidth
    val imgBuffer = ByteBuffer.allocate(h * w * 3 * 4)

    for (y <- 0 until h) {
      for (x <- 0 until w) {
        val pixel = image.getRGB(x, y)
        val red = (pixel >> 16) & 0xff
        val green = (pixel >> 8) & 0xff
        val blue = pixel & 0xff
        imgBuffer.putFloat(red.toFloat / divFactor)
        imgBuffer.putFloat(green.toFloat / divFactor)
        imgBuffer.putFloat(blue.toFloat / divFactor)
      }
    }
    imgBuffer.flip()
    val shape = Shape.of(1, image.getHeight, image.getWidth, 3)
    val db = DataBuffers.of(imgBuffer).asFloats()
    val t2 = TFloat32.tensorOf(shape, db)
    t2
  }

  def imgToTensorI(image: BufferedImage): TUint8 = {
    import java.nio.ByteBuffer
    val h = image.getHeight
    val w = image.getWidth
    val imgBuffer = ByteBuffer.allocate(h * w * 3)

    for (y <- 0 until h) {
      for (x <- 0 until w) {
        val pixel = image.getRGB(x, y)
        val red = (pixel >> 16) & 0xff
        val green = (pixel >> 8) & 0xff
        val blue = pixel & 0xff
        imgBuffer.put(red.toByte)
        imgBuffer.put(green.toByte)
        imgBuffer.put(blue.toByte)
      }
    }
    imgBuffer.flip()
    val shape = Shape.of(1, image.getHeight, image.getWidth, 3)
    val db = DataBuffers.of(imgBuffer)
    val t2 = TUint8.tensorOf(shape, db)
    t2
  }

  def clip(v: Int, min: Int, max: Int) = math.max(0, math.min(v, 255))

  def tensorFToImg(tensor: TFloat32, mulFactor: Float): BufferedImage = {
    val data = tensor.asRawTensor().data().asFloats()
    val h = tensor.shape.get(1).toInt
    val w = tensor.shape.get(2).toInt
    val img = new BufferedImage(w.toInt, h.toInt, BufferedImage.TYPE_INT_RGB)
    var index = 0

    for (y <- 0 until h) {
      for (x <- 0 until w) {
        val alpha = 0
        val r = clip((data.getFloat(index) * mulFactor).toInt, 0, 255); index += 1
        val g = clip((data.getFloat(index) * mulFactor).toInt, 0, 255); index += 1
        val b = clip((data.getFloat(index) * mulFactor).toInt, 0, 255); index += 1
        val rgb = alpha << 24 | r << 16 | g << 8 | b
        img.setRGB(x, y, rgb)
      }
    }
    img
  }

  def tensorIToImg(tensor: TUint8): BufferedImage = {
    val data = tensor.asRawTensor().data()
    val h = tensor.shape.get(1).toInt
    val w = tensor.shape.get(2).toInt
    val img = new BufferedImage(w.toInt, h.toInt, BufferedImage.TYPE_INT_RGB)
    var index = 0

    for (y <- 0 until h) {
      for (x <- 0 until w) {
        val alpha = 0
        import java.lang.Byte.toUnsignedInt
        val r = toUnsignedInt(data.getByte(index)); index += 1
        val g = toUnsignedInt(data.getByte(index)); index += 1
        val b = toUnsignedInt(data.getByte(index)); index += 1
        val rgb = alpha << 24 | r << 16 | g << 8 | b
        img.setRGB(x, y, rgb)
      }
    }
    img
  }
}
