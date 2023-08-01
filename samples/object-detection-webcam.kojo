import java.awt.image.BufferedImage
import java.util
import org.tensorflow.Tensor
import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.buffer.DataBuffers
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TUint8
import org.tensorflow.SavedModelBundle
import net.kogics.kojo.tensorutil._

import org.bytedeco.javacv._
import org.bytedeco.opencv.global.opencv_core.CV_32F
import org.bytedeco.opencv.global.opencv_dnn.{ readNetFromCaffe, blobFromImage }
import org.bytedeco.opencv.opencv_core.{ Rect, Mat, Size, Scalar, Point }
import org.bytedeco.javacpp.indexer.FloatIndexer
import org.bytedeco.opencv.global.opencv_imgproc.rectangle
import org.bytedeco.opencv.global.opencv_imgproc.resize
import org.bytedeco.opencv.global.opencv_imgcodecs.imread

val kojoAiRoot = "/home/lalit/work/kojo-ai-2"
val savedModel = "/home/lalit/work/object-det/models/ssdlite_mobilenet_v2_coco_2018_05_09/saved_model"

val fps = 5
cleari()
val cb = canvasBounds

var pics = ArrayBuffer.empty[Picture]

val model = SavedModelBundle.load(savedModel)

val labelsFile = scala.io.Source.fromFile(s"$kojoAiRoot/samples/mscoco-labels.txt")
val labels = HashMap.empty[Int, String]
labelsFile.getLines.zipWithIndex.foreach {
    case (line, idx) =>
        labels.put(idx + 1, line.trim)
}
labelsFile.close()

var lastFrameTime = epochTimeMillis

def detectSequence(grabber: FrameGrabber): Unit = {
    val delay = 1000.0 / fps

    grabber.start()
    try {
        var frame = grabber.grab()
        while (frame != null) {
            frame = grabber.grab()
            val currTime = epochTimeMillis
            if (currTime - lastFrameTime > delay) {
                val bufImg = Java2DFrameUtils.toBufferedImage(frame)
                if (bufImg != null) {
                    bufImg.getWidth
                    val pics2 = detect(bufImg)
                    pics2.foreach { pic =>
                        pic.moveToFront()
                        pic.translate(-bufImg.getWidth / 2, -bufImg.getHeight / 2)
                        pic.draw()
                    }
                    pics.foreach { pic =>
                        pic.erase()
                    }
                    pics = pics2
                    lastFrameTime = currTime
                }
            }
        }
    }
    catch {
        case _ => // eat up interruption
    }
    finally {
        grabber.stop()
        model.close()
    }
}

case class DetectionOutput(boxes: TFloat32, scores: TFloat32, classes: TFloat32, num: TFloat32)

def detectBox(src: BufferedImage, box: ArrayBuffer[Float], label: String, pics2: ArrayBuffer[Picture]) {
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
    pics2.append(bbox2, bbox, lbl2, lbl)
}

def detectBoxes(detectionOutput: DetectionOutput, src: BufferedImage, pics2: ArrayBuffer[Picture]) {
//    val num = detectionOutput.num.getFloat().toInt
    val foundLabels = HashSet.empty[String]
    for (i <- 0 until detectionOutput.boxes.shape.get(1).toInt) {
        val score = detectionOutput.scores.getFloat(0, i)
        if (score > 0.3) {
            val box = ArrayBuffer.empty[Float]
            detectionOutput.boxes.get(0, i).scalars.forEach { x =>
                box.append(x.getFloat())
            }
            val code = detectionOutput.classes.getFloat(0, i).toInt
            val label = labels.getOrElse(code, s"Unknown code - $code")
            detectBox(src, box, label, pics2)
            foundLabels.add(label)
        }
    }
    checkLabels(foundLabels)
}

def detect(src: BufferedImage): ArrayBuffer[Picture] = {
    val pics2 = ArrayBuffer.empty[Picture]
    val args = new util.HashMap[String, Tensor]()
    val inputTensor = imgToTensorI(src)
    args.put("inputs", inputTensor)
    val out = model.call(args)
    val boxes = out.get("detection_boxes").get.asInstanceOf[TFloat32]
    val classes = out.get("detection_classes").get.asInstanceOf[TFloat32]
    val scores = out.get("detection_scores").get.asInstanceOf[TFloat32]
    val num = out.get("num_detections").get.asInstanceOf[TFloat32]

    val detection = DetectionOutput(boxes, scores, classes, num)
    val pic = Picture.image(src)
    pics2.append(pic)
    detectBoxes(detection, src, pics2)
    pics2
}

val iRadius = 30
val indicator = Picture.circle(iRadius)
indicator.setPenColor(gray)

indicator.setPosition(cb.x + (cb.width - 2 * iRadius) / 2 + iRadius, cb.y + 50)
draw(indicator)

val grabber = new FFmpegFrameGrabber("/dev/video0");
detectSequence(grabber)

def checkLabels(labels: MSet[String]) {
    if (labels.contains("person")) {
        indicator.setFillColor(red)
    }
    else {
        indicator.setFillColor(green)
    }
}
