// #include /nst.kojo

cleari()
clearOutput()
setBackground(white)

val alpha = 0.8f

// you need to change the following locations based on where you downloaded and extracted
// the kojo-ai repository and the style transfer saved-model
val kojoAiRoot = "/home/lalit/work/kojo-ai-2"
val savedModel = "/home/lalit/work/nst/savedmodel/"

val fltr1 = new NeuralStyleFilter(savedModel, s"$kojoAiRoot/images/style/woman_with_hat_matisse_cropped.jpg", alpha)
val fltr2 = new NeuralStyleFilter(savedModel, s"$kojoAiRoot/images/style/sketch_cropped.png", alpha)

val drawing = Picture {
    setPenColor(cm.gray)
    var clr = cm.rgba(255, 0, 0, 127) // start with a semi transparent red color
    setPenThickness(8)
    repeat(18) {
        setFillColor(clr)
        repeat(5) {
            forward(100)
            right(72)
        }
        clr = clr.spin(360 / 18) // change color hue
        right(360 / 18)
    }
}

val pic = effect(fltr1) -> drawing
//val pic = effect(fltr2) * effect(fltr1) -> drawing

draw(pic)
