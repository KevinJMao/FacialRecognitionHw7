package recognition

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.slf4j.LoggerFactory
import utils.{MatrixHelpers, EigenFaces, ImageUtil}

import scala.util.{Failure, Random, Success, Try}

class FacialRecognition{
  private val logger = LoggerFactory.getLogger(classOf[FacialRecognition])
  private var eigenface : Option[BufferedImage] = None

  private val IMAGE_WIDTH = 180
  private val IMAGE_HEIGHT = 200
  private val RNG = Random

  private val EIGENFACE_FILE = "eigenface"

  private def loadImage(filePath : String): Option[BufferedImage] = loadImage(new File(filePath))

  private def loadImage(file : File) : Option[BufferedImage] = {
    Try{
      ImageIO.read(file)
    } match {
      case Success(image) => Some(image)
      case Failure(e) =>
        logger.error("Exception thrown while loading image: {}", e)
        None
    }
  }

  private def writeEigenface(image : BufferedImage, indexNumber : Int) = {
    Try {
      ImageIO.write(image, "jpg", new File(EIGENFACE_FILE + "_" + indexNumber + ".jpg"))
    } recover {
      case e : Throwable => logger.error("Exception thrown while writing image: {}", e)
    }
  }

  private def selectRandomImage : Option[BufferedImage] = {
    val images = new File("faces").listFiles()
    loadImage(images(RNG.nextInt(images.size)))
  }

  def run() = {
    var trainingFaces = for {
      i <- 1 to 10
    } yield selectRandomImage match {
        case Some(image) => image
        case None =>
          logger.error("Error while loading random image.")
          new BufferedImage(0,0,BufferedImage.TYPE_CUSTOM)
      }

    /* e.g. M = 10000 pixels, N = 50 samples */
    val pixelMatrix = trainingFaces.toArray.map { image =>
      ImageUtil.getNormalizedImagePixels(image, IMAGE_WIDTH, IMAGE_HEIGHT)
    }

    /* M x N */
    val pixelMatrix_2 = new Array2DRowRealMatrix(pixelMatrix).transpose()

    val averagePixels = EigenFaces.computeAverageFace(pixelMatrix)

    /* M x 1 */
    val averagePixelVector = MatrixHelpers.computeMeanVector(pixelMatrix_2)

    val eigenFaces = EigenFaces.computeEigenFaces_2(pixelMatrix_2, averagePixelVector).take(3)

    for((eigenFace, idx) <- eigenFaces.view.zipWithIndex) {
      val eigenFaceImage = ImageUtil.reconstructImage(eigenFace.vector.getColumn(0), IMAGE_WIDTH, IMAGE_HEIGHT)
      writeEigenface(eigenFaceImage, idx)
    }

//    //pixelMatrix = [50 * [36000]]
//    val eigenFaces = EigenFaces.computeEigenFaces(pixelMatrix, averagePixels)
//
//    eigenface = Some(ImageUtil.reconstructImage(eigenFaces.toArray.flatten, IMAGE_WIDTH, IMAGE_HEIGHT))
//
//    eigenface foreach { image => writeEigenface(image) }
  }
}

object FacialRecognition extends App {
  override def main(args: Array[String]): Unit = {
    new FacialRecognition().run
  }
}
