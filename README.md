# Kojo-AI
Support for data science, machine learning, and more - within Kojo.

A previous cut of kojo-ai is available at https://github.com/litan/kojo-ai. This is the next cut, based on the official tensorflow-java api

This cut of kojo-ai currently supports neural style transfer (NST). An example showing NST in action is given below:

```scala
// #include /home/lalit/work/kojo-ai-2/src/main/kojo/neural-style.kojo

cleari()
clearOutput()

val alpha = 0.8f
val fltr1 = new NeuralStyleFilter("/path/to/savedmodel/", "/path/to/style.jpg", alpha)
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
draw(pic)
```

