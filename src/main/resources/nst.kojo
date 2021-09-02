import java.awt.image.BufferedImage
import java.util

import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.buffer.DataBuffers
import org.tensorflow.types.TFloat32
import org.tensorflow.{ SavedModelBundle, Tensor }

import net.kogics.kojo.nst._
import net.kogics.kojo.tensorutil._

class NeuralStyleFilter(savedModel: String, style: String, alpha: Float) extends ImageOp {
    val styleImage = image(style)
    val styleTensor = imgToTensorF(removeAlphaChannel(styleImage, white))

    def filter(src: BufferedImage) = {
        val model = SavedModelBundle.load(savedModel)
        val args = new util.HashMap[String, Tensor]()
        val inputTensor = imgToTensorF(removeAlphaChannel(src, white))
        args.put("args_0", inputTensor)
        args.put("args_0_1", styleTensor)
        args.put("args_0_2", TFloat32.scalarOf(alpha))
        val out = model.call(args).get("output_1").asInstanceOf[TFloat32]
        model.close()
        val ret = tensorFToImg(out)
        out.close()
        ret
    }
}
