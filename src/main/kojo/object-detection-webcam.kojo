import java.awt.image.BufferedImage
import java.util
import org.tensorflow.Tensor
import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.buffer.DataBuffers
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TUint8
import com.github.sarxos.webcam.Webcam
import org.tensorflow.SavedModelBundle

val fps = 5
cleari()
val cb = canvasBounds
val webcam = Webcam.getDefault()
println(webcam)
val views = webcam.getViewSizes
val idx = views.length - 1
val selectedView = views(idx)
webcam.setViewSize(selectedView)
webcam.open()

zoom(1, selectedView.getWidth / 2, selectedView.getHeight / 2)

var stop = false
val stopBtn = picStackCentered(
    fillColor(red) -> Picture.rectangle(100, 50),
    penColor(black) -> Picture.text("Stop")
)
stopBtn.onMouseClick { (x, y) =>
    stop = true
}
stopBtn.setPosition(-110, -60)
draw(stopBtn)

var pics = ArrayBuffer.empty[Picture]

val model = "/home/lalit/work/object-det/models/ssdlite_mobilenet_v2_coco_2018_05_09/saved_model"
//val model = "/home/lalit/work/object-det/models/ssd_inception_v2_coco_2017_11_17/saved_model"
val savedModel = SavedModelBundle.load(model)

val labelsFile = scala.io.Source.fromFile("/home/lalit/work/kojo-ai-2/src/main/kojo/mscoco-labels.txt")
val labels = HashMap.empty[Int, String]
labelsFile.getLines.zipWithIndex.foreach {
    case (line, idx) =>
        labels.put(idx + 1, line.trim)
}
labelsFile.close()

try {
    val delay = 1.0 / fps
    repeatWhile(!stop) {
        // get image
        val frame = webcam.getImage();
        val pics2 = detectAndDraw(frame)
        pics.foreach { pic =>
            pic.erase()
        }
        pics = pics2
        pause(delay)
    }
}
finally {
    webcam.close()
    savedModel.close()
}

def imgToTensorI(image: BufferedImage): Tensor[TUint8] = {
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

case class DetectionOutput(boxes: Tensor[TFloat32], scores: Tensor[TFloat32], classes: Tensor[TFloat32], num: Tensor[TFloat32])

def drawBox(src: BufferedImage, box: ArrayBuffer[Float], label: String, pics2: ArrayBuffer[Picture]) {
def drawBoxes(detectionOutput: DetectionOutput, src: BufferedImage, pics2: ArrayBuffer[Picture]) {
    val num = detectionOutput.num.data.getFloat().toInt
    //    println(s"Objects detected: $num")
    //    println(detectionOutput)
    for (i <- 0 until detectionOutput.boxes.shape.size(1).toInt) {
        val score = detectionOutput.scores.data.getFloat(0, i)
        if (score > 0.3) {
            //            println(score)
            val box = ArrayBuffer.empty[Float]
            detectionOutput.boxes.data.get(0, i).scalars.forEach { x =>
                box.append(x.getFloat())
            }
            val code = detectionOutput.classes.data.getFloat(0, i).toInt
            val label = labels.getOrElse(code, s"Unknown code - $code")
            drawBox(src, box, label, pics2)
        }
    }
}

def detectAndDraw(src: BufferedImage): ArrayBuffer[Picture] = {
    val pics2 = ArrayBuffer.empty[Picture]
    val args = new util.HashMap[String, Tensor[_]]()
    val inputTensor = imgToTensorI(src)
    args.put("inputs", inputTensor)
    val out = savedModel.call(args)
    val boxes = out.get("detection_boxes").asInstanceOf[Tensor[TFloat32]]
    val classes = out.get("detection_classes").asInstanceOf[Tensor[TFloat32]]
    val scores = out.get("detection_scores").asInstanceOf[Tensor[TFloat32]]
    val num = out.get("num_detections").asInstanceOf[Tensor[TFloat32]]

    val detection = DetectionOutput(boxes, scores, classes, num)
    val pic = Picture.image(src)
    pics2.append(pic)
    draw(pic)
    drawBoxes(detection, src, pics2)
    pics2
}
