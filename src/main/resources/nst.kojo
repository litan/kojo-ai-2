import java.awt.image.BufferedImage
import java.util
import org.tensorflow.Tensor
import org.tensorflow.SavedModelBundle
import org.tensorflow.types.TFloat32
import net.kogics.kojo.nst._

class NeuralStyleFilter(model: String, style: String, alpha: Float) extends ImageOp {
    val styleImage = image(style)
    val styleTensor = imgToTensor(removeAlphaChannel(styleImage, white))

    def filter(src: BufferedImage) = {
      val savedModel = SavedModelBundle.load(model)
      val args = new util.HashMap[String, Tensor[_]]()
      val inputTensor = imgToTensor(removeAlphaChannel(src, white))
      args.put("args_0", inputTensor)
      args.put("args_0_1", styleTensor)
      args.put("args_0_2", TFloat32.scalarOf(alpha))
      val out = savedModel.call(args).get("output_1").asInstanceOf[Tensor[TFloat32]]
      savedModel.close()
      tensorToImg(out)
    }
}
