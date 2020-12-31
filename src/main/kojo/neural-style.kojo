import java.awt.image.BufferedImage
import java.util

import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.buffer.DataBuffers
import org.tensorflow.types.TFloat32
import org.tensorflow.{SavedModelBundle, Tensor}

import net.kogics.kojo.nst._
import net.kogics.kojo.tensorutil._

class NeuralStyleFilter(model: String, style: String, alpha: Float) extends ImageOp {
    val styleImage = image(style)
    val styleTensor = imgToTensorF(removeAlphaChannel(styleImage, white))

    def filter(src: BufferedImage) = {
        println("Loading model")
        val savedModel = SavedModelBundle.load(model)
        val args = new util.HashMap[String, Tensor[_]]()
        val inputTensor = imgToTensorF(removeAlphaChannel(src, white))
        println(s"Style Input: $inputTensor")
        args.put("args_0", inputTensor)
        args.put("args_0_1", styleTensor)
        args.put("args_0_2", TFloat32.scalarOf(alpha))
        val out = savedModel.call(args).get("output_1").asInstanceOf[Tensor[TFloat32]]
        println(s"Style Output: $out")
        savedModel.close()
        val ret = tensorFToImg(out)
        out.close()
        ret
    }
}
