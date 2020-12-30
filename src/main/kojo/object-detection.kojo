import java.awt.image.BufferedImage
import java.util
import org.tensorflow.Tensor
import org.tensorflow.SavedModelBundle
import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.buffer.DataBuffers
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TUint8

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

def drawBox(src: BufferedImage, box: ArrayBuffer[Float]) {
    val w = src.getWidth
    val h = src.getHeight
    val xmin = w * box(1)
    val xmax = w * box(3)
    val ymin1 = h * box(0)
    val ymax = h - ymin1
    val ymax1 = h * box(2)
    val ymin = h - ymax1
    PictureDraw.rectangle(xmin, ymin, xmax - xmin, ymax - ymin)
}

def drawBoxes(detectionOutput: DetectionOutput) {
    val num = detectionOutput.num.data.getFloat().toInt
    println(s"Objects detected: $num")
    println(detectionOutput)
    for (i <- 0 until detectionOutput.boxes.shape.size(1).toInt) {
        val score = detectionOutput.scores.data.getFloat(0, i)
        if (score > 0.5) {
            println(score)
            val box = ArrayBuffer.empty[Float]
            detectionOutput.boxes.data.get(0, i).scalars.forEach { x =>
                box.append(x.getFloat())
            }
            drawBox(src, box)
            println(detectionOutput.classes.data.getFloat(0, i))
            println("---")
        }
    }
}

cleari()
clearOutput()

val model = "/home/lalit/work/object-det/models/ssdlite_mobilenet_v2_coco_2018_05_09/saved_model"
val src = image("/home/lalit/Downloads/cells.jpg")
val savedModel = SavedModelBundle.load(model)
val args = new util.HashMap[String, Tensor[_]]()
val inputTensor = imgToTensorI(src)
args.put("inputs", inputTensor)
val out = savedModel.call(args)
val boxes = out.get("detection_boxes").asInstanceOf[Tensor[TFloat32]]
val classes = out.get("detection_classes").asInstanceOf[Tensor[TFloat32]]
val scores = out.get("detection_scores").asInstanceOf[Tensor[TFloat32]]
val num = out.get("num_detections").asInstanceOf[Tensor[TFloat32]]
savedModel.close()

val detection = DetectionOutput(boxes, scores, classes, num)
val pic = Picture.image(src)
draw(pic)
drawBoxes(detection)
