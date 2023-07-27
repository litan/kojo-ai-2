import java.awt.image.BufferedImage
import java.util.HashMap

import scala.util.Using

import org.tensorflow.types.TFloat32
import org.tensorflow.{ SavedModelBundle, Tensor }

import net.kogics.kojo.nst._
import net.kogics.kojo.tensorutil._

class NeuralStyleFilter(savedModelFile: String, styleImageFile: String, alpha: Float) extends ImageOp {
    val styleImage = image(styleImageFile)
    val styleTensor = imgToTensorF(removeAlphaChannel(styleImage, white))

    def filter(src: BufferedImage) = {
        Using.Manager { use =>
            val model = use(SavedModelBundle.load(savedModelFile))
            val args = new HashMap[String, Tensor]()
            val inputTensor = imgToTensorF(removeAlphaChannel(src, white))
            args.put("args_0", inputTensor)
            args.put("args_0_1", styleTensor)
            args.put("args_0_2", TFloat32.scalarOf(alpha))
            val out = use(model.call(args).get("output_1").get.asInstanceOf[TFloat32])
            tensorFToImg(out)
        }.get
    }
}
