import java.awt.image.BufferedImage
import java.io.{ BufferedInputStream, File, FileInputStream }
import java.nio.FloatBuffer
import java.util

import javax.imageio.ImageIO
import org.tensorflow.{ SavedModelBundle, Tensor }
import org.tensorflow.types.{ TFloat32, TUint8 }
import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.buffer.DataBuffers

class NeuralStyleFilter(model: String, style: String, alpha: Float) extends ImageOp {
    def imgToTensor2(image: BufferedImage): Tensor[TFloat32] = {
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
                imgBuffer.putFloat(red)
                imgBuffer.putFloat(green)
                imgBuffer.putFloat(blue)
            }
        }
        imgBuffer.flip()
        val shape = Shape.of(1, image.getHeight, image.getWidth, 3)
        val db = DataBuffers.of(imgBuffer).asFloats()
        val t2 = TFloat32.tensorOf(shape, db)
        t2
    }

    def tensorToImg(tensor: Tensor[TFloat32]): BufferedImage = {
        val data = tensor.rawData().asFloats()
        val h = tensor.shape.size(1).toInt
        val w = tensor.shape.size(2).toInt
        val img = new BufferedImage(w.toInt, h.toInt, BufferedImage.TYPE_INT_RGB)
        var index = 0
        def clip(v: Int, min: Int, max: Int) = math.max(0, math.min(v, 255))

        for (y <- 0 until h) {
            for (x <- 0 until w) {
                val alpha = 0
                val r = clip(data.getFloat(index).toInt, 0, 255); index += 1
                val g = clip(data.getFloat(index).toInt, 0, 255); index += 1
                val b = clip(data.getFloat(index).toInt, 0, 255); index += 1
                val rgb = alpha << 24 | r << 16 | g << 8 | b
                img.setRGB(x, y, rgb)
            }
        }
        img
    }

    def removeAlphaChannel(img: BufferedImage, color: Color = white): BufferedImage = {
        if (!img.getColorModel().hasAlpha()) {
            return img
        }

        val target = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB)
        val g = target.createGraphics()
        g.setColor(color)
        g.fillRect(0, 0, img.getWidth(), img.getHeight())
        g.drawImage(img, 0, 0, null)
        g.dispose()
        target;
    }

    val styleImage = image(style)
    val styleTensor = imgToTensor2(removeAlphaChannel(styleImage, white))

    def filter(src: BufferedImage) = {
        println("Loading model")
        val savedModel = SavedModelBundle.load(model)
        val args = new util.HashMap[String, Tensor[_]]()
        val inputTensor = imgToTensor2(removeAlphaChannel(src, white))
        println(s"Style Input: $inputTensor")
        args.put("args_0", inputTensor)
        args.put("args_0_1", styleTensor)
        args.put("args_0_2", TFloat32.scalarOf(alpha))
        val out = savedModel.call(args).get("output_1").asInstanceOf[Tensor[TFloat32]]
        println(s"Style Input: $out")
        savedModel.close()
        tensorToImg(out)
    }
}
