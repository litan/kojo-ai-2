package net.kogics.kojo

import java.awt.image.BufferedImage

import org.tensorflow.ndarray.buffer.DataBuffers
import org.tensorflow.ndarray.Shape
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TUint8

package object tensorutil {
  type RgbFloatScalar = (Float, Float, Float) => (Float, Float, Float)
  type RgbIntScalar = (Int, Int, Int) => (Int, Int, Int)

  def imgToTensorF(
      image: BufferedImage,
      oScaler: Option[RgbFloatScalar] = None
  ): TFloat32 = {
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
        val (r, g, b) = oScaler match {
          case Some(scaler) => scaler(red.toFloat, green.toFloat, blue.toFloat)
          case None         => (red.toFloat, green.toFloat, blue.toFloat)
        }
        imgBuffer.putFloat(r); imgBuffer.putFloat(g); imgBuffer.putFloat(b)
      }
    }
    imgBuffer.flip()
    val shape = Shape.of(1, image.getHeight, image.getWidth, 3)
    val db = DataBuffers.of(imgBuffer).asFloats()
    val t2 = TFloat32.tensorOf(shape, db)
    t2
  }

  def imgToTensorI(image: BufferedImage, oScaler: Option[RgbIntScalar] = None): TUint8 = {
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
        val (r, g, b) = oScaler match {
          case Some(scaler) => scaler(red, green, blue)
          case None         => (red, green, blue)
        }
        imgBuffer.put(r.toByte)
        imgBuffer.put(g.toByte)
        imgBuffer.put(b.toByte)
      }
    }
    imgBuffer.flip()
    val shape = Shape.of(1, image.getHeight, image.getWidth, 3)
    val db = DataBuffers.of(imgBuffer)
    val t2 = TUint8.tensorOf(shape, db)
    t2
  }

  def clip(v: Int, min: Int, max: Int) = math.max(0, math.min(v, 255))

  def tensorFToImg(tensor: TFloat32, oScaler: Option[RgbFloatScalar] = None): BufferedImage = {
    val data = tensor.asRawTensor().data().asFloats()
    val h = tensor.shape.get(1).toInt
    val w = tensor.shape.get(2).toInt
    val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    var index = 0

    for (y <- 0 until h) {
      for (x <- 0 until w) {
        val red = data.getFloat(index); index += 1
        val green = data.getFloat(index); index += 1
        val blue = data.getFloat(index); index += 1

        val (r0, g0, b0) = oScaler match {
          case Some(scaler) => scaler(red, green, blue)
          case None         => (red, green, blue)
        }

        val r = clip(r0.toInt, 0, 255)
        val g = clip(g0.toInt, 0, 255)
        val b = clip(b0.toInt, 0, 255)
        val rgb = r << 16 | g << 8 | b
        img.setRGB(x, y, rgb)
      }
    }
    img
  }

  def tensorIToImg(tensor: TUint8, oScaler: Option[RgbIntScalar] = None): BufferedImage = {
    val data = tensor.asRawTensor().data()
    val h = tensor.shape.get(1).toInt
    val w = tensor.shape.get(2).toInt
    val img = new BufferedImage(w.toInt, h.toInt, BufferedImage.TYPE_INT_RGB)
    var index = 0

    import java.lang.Byte.toUnsignedInt
    for (y <- 0 until h) {
      for (x <- 0 until w) {
        val red = toUnsignedInt(data.getByte(index)); index += 1
        val green = toUnsignedInt(data.getByte(index)); index += 1
        val blue = toUnsignedInt(data.getByte(index)); index += 1

        val (r0, g0, b0) = oScaler match {
          case Some(scaler) => scaler(red, green, blue)
          case None         => (red, green, blue)
        }

        val r = clip(r0, 0, 255)
        val g = clip(g0, 0, 255)
        val b = clip(b0, 0, 255)
        val rgb = r << 16 | g << 8 | b
        img.setRGB(x, y, rgb)
      }
    }
    img
  }
}
