// #include /nst.kojo

cleari()
clearOutput()

val alpha = 0.8f
val fltr1 = new NeuralStyleFilter("/path/to/savedmodel/", "/path/to/style.jpg", alpha)
// val fltr2 = new NeuralStyleFilter("/path/to/savedmodel/", "/path/to/style2.jpg", alpha)

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
// or apply two filters
// val pic = effect(fltr1) * effect(fltr2) -> drawing

draw(pic)
