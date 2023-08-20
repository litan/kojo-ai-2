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
  type PixelFloatScalar = (Float, Float, Float) => (Float, Float, Float)
  type PixelIntScalar = (Int, Int, Int) => (Int, Int, Int)

  trait ImageReader {
    def getRGB(x: Int, y: Int): Int
  }

  object ImageReader {
    def create(image: BufferedImage): ImageReader = image.getType match {
      case BufferedImage.TYPE_INT_RGB   => new ImageReader_INT_RGB(image)
      case BufferedImage.TYPE_3BYTE_BGR => new ImageReader_3BYTE_BGR(image)
      case _                            => throw new RuntimeException("Unknown Image Type")
    }
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
      oScaler: Option[PixelFloatScalar] = None
  ): TFloat32 = {
    val imgHeight = image.getHeight
    val imgWidth = image.getWidth
    val imgReader = ImageReader.create(image)

    val shape = Shape.of(1, imgHeight, imgWidth, 3)

    TFloat32.tensorOf(
      shape,
      (tensor: TFloat32) => {
        for (y <- 0 until imgHeight) {
          for (x <- 0 until imgWidth) {
            val pixel = imgReader.getRGB(x, y)
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

  def imgToTensorI(
      image: BufferedImage,
      oScaler: Option[PixelIntScalar] = None
  ): TUint8 = {
    val imgHeight = image.getHeight
    val imgWidth = image.getWidth
    val imgReader = ImageReader.create(image)

    val shape = Shape.of(1, imgHeight, imgWidth, 3)

    TUint8.tensorOf(
      shape,
      (tensor: TUint8) => {
        for (y <- 0 until imgHeight) {
          for (x <- 0 until imgWidth) {
            val pixel = imgReader.getRGB(x, y)
            val red = (pixel >> 16) & 0xff
            val green = (pixel >> 8) & 0xff
            val blue = pixel & 0xff
            val (r, g, b) = oScaler match {
              case Some(scaler) => scaler(red, green, blue)
              case None         => (red, green, blue)
            }
            tensor.setByte(r.toByte, 0, y, x, 0)
            tensor.setByte(g.toByte, 0, y, x, 1)
            tensor.setByte(b.toByte, 0, y, x, 2)
          }
        }
      }
    )
  }

  def clip(v: Int, min: Int, max: Int) = math.max(0, math.min(v, 255))

  def tensorFToImg(tensor: TFloat32, oScaler: Option[PixelFloatScalar] = None): BufferedImage = {
    val data = tensor.asRawTensor().data().asFloats()
    val h = tensor.shape.get(1).toInt
    val w = tensor.shape.get(2).toInt
    val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
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

  def tensorIToImg(tensor: TUint8, oScaler: Option[PixelIntScalar] = None): BufferedImage = {
    val data = tensor.asRawTensor().data()
    val h = tensor.shape.get(1).toInt
    val w = tensor.shape.get(2).toInt
    val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    val imgRaster = img.getRaster
    var index = 0

    val pixels = new Array[Int](w)
    for (y <- 0 until h) {
      for (x <- 0 until w) {
        val red = toUnsignedInt(data.getByte(index)); index += 1
        val green = toUnsignedInt(data.getByte(index)); index += 1
        val blue = toUnsignedInt(data.getByte(index)); index += 1

        val (r0, g0, b0) = oScaler match {
          case Some(scaler) => scaler(red, green, blue)
          case None => (red, green, blue)
        }

        val r = clip(r0, 0, 255)
        val g = clip(g0, 0, 255)
        val b = clip(b0, 0, 255)
        val rgb = r << 16 | g << 8 | b
        pixels(x) = rgb
      }
      imgRaster.setDataElements(0, y, w, 1, pixels)
    }
    img
  }
}
