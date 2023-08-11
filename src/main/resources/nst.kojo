import java.awt.image.BufferedImage
import java.util.HashMap

import scala.util.Using

import org.tensorflow.types.TFloat32
import org.tensorflow.{ SavedModelBundle, Tensor }

import net.kogics.kojo.nst._
import net.kogics.kojo.tensorutil._

class NeuralStyleFilter(savedModelFile: String, styleImageFile: String, alpha: Float) extends ImageOp {
    val scaleFactor = 1f
    val styleImage = image(styleImageFile)
    val styleTensor = imgToTensorF(removeAlphaChannel(styleImage, white), scaleFactor)

    def filter(src: BufferedImage) = {
        Using.Manager { use =>
            val model = use(SavedModelBundle.load(savedModelFile))
            val args = new HashMap[String, Tensor]()
            val inputTensor = imgToTensorF(removeAlphaChannel(src, white), scaleFactor)
            args.put("args_0", inputTensor)
            args.put("args_0_1", styleTensor)
            args.put("args_0_2", TFloat32.scalarOf(alpha))
            val out = use(model.call(args).get("output_1").get.asInstanceOf[TFloat32])
            tensorFToImg(out, scaleFactor)
        }.get
    }
}

class NeuralStyleFilter2(savedModelFile: String, styleImageFile: String) extends ImageOp {
    val scaleFactor = 255f
    val styleImage = image(styleImageFile)
    val styleTensor = modifyStyle(imgToTensorF(removeAlphaChannel(styleImage, white), scaleFactor))

    def modifyStyle(t: TFloat32): TFloat32 = t

    def filter(src: BufferedImage) = {
        Using.Manager { use =>
            val model = use(SavedModelBundle.load(savedModelFile))
            val args = new HashMap[String, Tensor]()
            val inputTensor = imgToTensorF(removeAlphaChannel(src, white), scaleFactor)
            args.put("placeholder", inputTensor)
            args.put("placeholder_1", styleTensor)
            val out = use(model.call(args).get("output_0").get.asInstanceOf[TFloat32])
            tensorFToImg(out, scaleFactor)
        }.get
    }
}

class NeuralStyleFilter2b(savedModelFile: String, styleImageFile: String, styleBlur: Int)
    extends NeuralStyleFilter2(savedModelFile: String, styleImageFile: String) {
    require(styleBlur > 2 && styleBlur < 13, "style blur should be between 3 and 12 (included)")
    override def modifyStyle(t: TFloat32): TFloat32 = {
        val tf = org.tensorflow.op.Ops.create()
        import java.util.Arrays
        tf.nn.avgPool(
            tf.constant(t),
            Arrays.asList(1, 3, 3, 1),
            Arrays.asList(1, 1, 1, 1),
            "SAME"
        ).asTensor()
    }
}

