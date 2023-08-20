package net.kogics.kojo

import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import java.awt.image.DataBufferInt
import java.lang.Byte.toUnsignedInt

import org.tensorflow.ndarray.buffer.DataBuffers
import org.tensorflow.ndarray.Shape
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TUint8

package object tensorutil {
  type PixelScalar = (Float, Float, Float) => (Float, Float, Float)

  trait ImageReader {
    def getRGB(x: Int, y: Int): Int
  }

  class ImageReader_INT_RGB(image: BufferedImage) extends ImageReader {
    val imgWidth = image.getWidth
    val imgPixels = image.getRaster.getDataBuffer.asInstanceOf[DataBufferInt].getData

    def getRGB(x: Int, y: Int): Int = {
      imgPixels(x + y * imgWidth)
    }
  }

  class ImageReader_3BYTE_BGR(image: BufferedImage) extends ImageReader {
    val imgWidth = image.getWidth
    val imgPixels = image.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData

    def getRGB(x: Int, y: Int): Int = {
      val idx = (x + y * imgWidth) * 3
      val b = toUnsignedInt(imgPixels(idx + 0))
      val g = toUnsignedInt(imgPixels(idx + 1))
      val r = toUnsignedInt(imgPixels(idx + 2))
      r << 16 | g << 8 | b
    }
  }

  def imgToTensorF(
      image: BufferedImage,
      oScaler: Option[PixelScalar] = None
  ): TFloat32 = {
    val imgHeight = image.getHeight
    val imgWidth = image.getWidth
    val imgr: ImageReader =
      if (image.getType == BufferedImage.TYPE_INT_RGB) new ImageReader_INT_RGB(image)
      else if (image.getType == BufferedImage.TYPE_3BYTE_BGR) new ImageReader_3BYTE_BGR(image)
      else throw new RuntimeException("Unknown Image Type")

    val shape = Shape.of(1, imgHeight, imgWidth, 3)

    TFloat32.tensorOf(
      shape,
      (tensor: TFloat32) => {
        for (y <- 0 until imgHeight) {
          for (x <- 0 until imgWidth) {
            val pixel = imgr.getRGB(x, y)
            val red = (pixel >> 16) & 0xff
            val green = (pixel >> 8) & 0xff
            val blue = pixel & 0xff
            val (r, g, b) = oScaler match {
              case Some(scaler) => scaler(red.toFloat, green.toFloat, blue.toFloat)
              case None         => (red.toFloat, green.toFloat, blue.toFloat)
            }
            tensor.setFloat(r, 0, y, x, 0)
            tensor.setFloat(g, 0, y, x, 1)
            tensor.setFloat(b, 0, y, x, 2)
          }
        }
      }
    )
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

  def tensorFToImg(tensor: TFloat32, oScaler: Option[PixelScalar] = None): BufferedImage = {
    val data = tensor.asRawTensor().data().asFloats()
    val h = tensor.shape.get(1).toInt
    val w = tensor.shape.get(2).toInt
    val img = new BufferedImage(w.toInt, h.toInt, BufferedImage.TYPE_INT_RGB)
    val imgRaster = img.getRaster
    var index = 0

    val pixels = new Array[Int](w)
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
        pixels(x) = rgb
      }
      imgRaster.setDataElements(0, y, w, 1, pixels)
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
