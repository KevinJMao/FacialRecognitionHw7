package utils

import java.awt.image.{BufferedImage, DataBufferByte}

import org.imgscalr.Scalr

/**
 * Generic utility for image manipulation.
 */
object ImageUtil {

  def toGrayscale(image: BufferedImage): BufferedImage = {
    val width = image.getWidth
    val height = image.getHeight
    val grayImg = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
    val g = grayImg.createGraphics()
    g.drawImage(image, 0, 0, null)

    (0 to (width-1)).foreach { x =>
      (0 to (height-1)).foreach { y =>
        val color = grayImg.getRGB(x, y)
        grayImg.setRGB(x, y, color)
      }
    }

    grayImg
  }

  def getNormalizedImagePixels(image: BufferedImage, width: Int, height: Int): Array[Double] = {
    val scaledImage = Scalr.resize(image, Scalr.Method.BALANCED, Scalr.Mode.FIT_TO_HEIGHT, width, height)
    val greyImage: BufferedImage = new ImageNormalizer().getNormalizedValues(scaledImage)

    // convert to grayscale image
    val bytePixels: Array[Byte] = (greyImage.getRaster.getDataBuffer.asInstanceOf[DataBufferByte]).getData

    val doublePixels: Array[Double] = new Array[Double](bytePixels.length)

    (0 to (doublePixels.length-1)).foreach { i =>
      doublePixels(i) = (bytePixels(i) & 255).asInstanceOf[Double]
    }

    doublePixels
  }

  def reconstructImage(imagePixels: Array[Double], width: Int, height: Int): BufferedImage = {
    val meanImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
    val raster = meanImage.getRaster()

    // convert byte array to byte array
    val pixels = new Array[Int](imagePixels.length)
    (0 to (imagePixels.length-1)).foreach { i =>
      pixels(i) = imagePixels(i).toInt
    }
    raster.setPixels(0, 0, width, height, pixels)
    meanImage
  }
}
