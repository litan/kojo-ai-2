import java.awt.image.BufferedImage
import java.util
import org.tensorflow.Tensor
import org.tensorflow.SavedModelBundle
import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.buffer.DataBuffers
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TUint8
import net.kogics.kojo.tensorutil._

// you need to change the following locations based on where you downloaded and extracted
// the kojo-ai repository and the object detection saved-model
val kojoAiRoot = "/home/lalit/work/kojo-ai-2"
val savedModel = "/home/lalit/work/object-det/models/ssdlite_mobilenet_v2_coco_2018_05_09/saved_model"

val labelsFile = scala.io.Source.fromFile(s"$kojoAiRoot/src/main/kojo/mscoco-labels.txt")
val labels = HashMap.empty[Int, String]
labelsFile.getLines.zipWithIndex.foreach {
    case (line, idx) =>
        labels.put(idx + 1, line.trim)
}
labelsFile.close()

case class DetectionOutput(boxes: TFloat32, scores: TFloat32, classes: TFloat32, num: TFloat32)

def drawBox(src: BufferedImage, box: ArrayBuffer[Float], label: String) {
    val w = src.getWidth
    val h = src.getHeight
    val xmin = w * box(1)
    val xmax = w * box(3)
    val ymin1 = h * box(0)
    val ymax = h - ymin1
    val ymax1 = h * box(2)
    val ymin = h - ymax1
    val bbox = Picture.rectangle(xmax - xmin, ymax - ymin)
    val bbox2 = Picture.rectangle(xmax - xmin, ymax - ymin)
    val lbl = Picture.text(label)
    val lbl2 = Picture.text(label)
    draw(bbox2, bbox, lbl2, lbl)
    bbox.setPosition(xmin, ymin)
    bbox2.setPosition(xmin, ymin)
    bbox.setPenColor(ColorMaker.hsl(60, 0.91, 0.68))
    bbox2.setPenColor(darkGray)
    bbox.setPenThickness(4)
    bbox2.setPenThickness(6)
    lbl.setPosition(xmin, ymin)
    lbl.setPenColor(yellow)
    lbl2.setPosition(xmin + 1, ymin - 1)
    lbl2.setPenColor(black)
}

def drawBoxes(detectionOutput: DetectionOutput) {
    val num = detectionOutput.num.getFloat().toInt
    for (i <- 0 until detectionOutput.boxes.shape.size(1).toInt) {
        val score = detectionOutput.scores.getFloat(0, i)
        if (score > 0.3) {
            println(score)
            val box = ArrayBuffer.empty[Float]
            detectionOutput.boxes.get(0, i).scalars.forEach { x =>
                box.append(x.getFloat())
            }
            val code = detectionOutput.classes.getFloat(0, i).toInt
            val label = labels.getOrElse(code, s"Unknown code - $code")
            drawBox(src, box, label)
        }
    }
}

cleari()
clearOutput()
setBackground(white)

val src = image(s"$kojoAiRoot/images/elephants-pixabay.jpg")

val pic = Picture.image(src)
draw(pic)

val model = SavedModelBundle.load(savedModel)
val args = new util.HashMap[String, Tensor]()
val inputTensor = imgToTensorI(src)
args.put("inputs", inputTensor)
val out = model.call(args)
val boxes = out.get("detection_boxes").get.asInstanceOf[TFloat32]
val classes = out.get("detection_classes").get.asInstanceOf[TFloat32]
val scores = out.get("detection_scores").get.asInstanceOf[TFloat32]
val num = out.get("num_detections").get.asInstanceOf[TFloat32]
model.close()

val detection = DetectionOutput(boxes, scores, classes, num)
drawBoxes(detection)
