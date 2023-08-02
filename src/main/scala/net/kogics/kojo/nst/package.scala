package net.kogics.kojo

package object nst {
  import java.awt.image.BufferedImage
  import java.awt.Color

  def removeAlphaChannel(img: BufferedImage, color: Color = Color.white): BufferedImage = {
    if (!img.getColorModel().hasAlpha()) {
      return img
    }

    val target = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB)
    val g = target.createGraphics()
    g.setColor(color)
    g.fillRect(0, 0, img.getWidth(), img.getHeight())
    g.drawImage(img, 0, 0, null)
    g.dispose()
    target;
  }
}
