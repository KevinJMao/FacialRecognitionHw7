package recognition

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

import org.apache.commons.math3.linear.{RealMatrix, RealVector, ArrayRealVector, Array2DRowRealMatrix}
import org.slf4j.LoggerFactory
import utils.{MatrixHelpers, EigenFaces, ImageUtil}

import scala.util.{Failure, Random, Success, Try}

class FacialRecognition{
  private val logger = LoggerFactory.getLogger(classOf[FacialRecognition])
  private var eigenface : Option[BufferedImage] = None

  private val IMAGE_WIDTH = 180
  private val IMAGE_HEIGHT = 200
  private val TRAINING_SAMPLE = 20
  private val SELECT_TOP_N_EIGENFACES = 5
  private val MATCH_AGAINST_X_FACES = 10
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
    val trainFaces = for {
      i <- 1 to TRAINING_SAMPLE
    } yield selectRandomImage match {
        case Some(image) => image
        case None =>
          logger.error("Error while loading random training image.")
          new BufferedImage(0,0,BufferedImage.TYPE_CUSTOM)
      }

    /* e.g. M = 10000 pixels, N = 50 samples */
    val trainPixel2DArray = trainFaces.toArray.map { image =>
      ImageUtil.getNormalizedImagePixels(image, IMAGE_WIDTH, IMAGE_HEIGHT)
    }

    /* M x N */
    val trainPixelMatrix = new Array2DRowRealMatrix(trainPixel2DArray).transpose()

    /* M x 1 */
    val trainMeanPixelVector = MatrixHelpers.computeMeanVector(trainPixelMatrix)

    /* An array of the top SELECT_TOP_N_EIGENFACES to use as comparison */
    val eigenFaceArray = EigenFaces.computeEigenFaces(trainPixelMatrix, trainMeanPixelVector).take(SELECT_TOP_N_EIGENFACES)

    /* Generate a reference weight vector to measure test faces against each face class */
    /* Take the training matrix and convert it into an array of vectors */
    /* Normalize each vector against the mean pixel vector */
    val trainNormalizedPixelVectorArray : Array[RealVector] = for(idx <- trainPixelMatrix.getColumnDimension) yield {
      trainPixelMatrix.getColumnVector(idx).subtract(trainMeanPixelVector)
    }

    /* Determine the mean class vector by which to match new faces against */
    val meanClassVector = trainNormalizedPixelVectorArray map { pixelVector =>
      new ArrayRealVector(
        eigenFaceArray map { eigenFace =>
          EigenFaces.computeImageWeightAgainstEigenFace(pixelVector, trainMeanPixelVector, eigenFace)
        })
    } reduce {(current, next) =>
      current.add(next)
    } mapDivide(trainNormalizedPixelVectorArray.length)

    /* Derive the average of all resultant vectors to yield the average class vector */

    /* Take the mean of all vectors belonging to an individual eigenface, this is the face class by which to compare all future mappings */


    /* Select MATCH_AGAINST_X_FACES to match against */
    val testFaces = for {
      i <- 1 to MATCH_AGAINST_X_FACES
    } yield selectRandomImage match {
        case Some(image) => image
        case None =>
          logger.error("Error hwile loading random testing image")
          new BufferedImage(0,0, BufferedImage.TYPE_CUSTOM)
      }

    /* Convert each image into its grayscale intensity values */
    val testPixelVectorArray = testFaces.toArray.map { image =>
      ImageUtil.getNormalizedImagePixels(image, IMAGE_WIDTH, IMAGE_HEIGHT)
    } map { doubleArray =>
      new ArrayRealVector(doubleArray)
    }

    /* Normalize each image against the average pixel vector of the training images */
    /* Omega */
    val testPixelCovarianceVectorArray : Array[RealVector] = testPixelVectorArray map { pixelVector =>
      MatrixHelpers.computePixelCovarianceVector(pixelVector, trainMeanPixelVector)
    }

    val testFaceClassWeightVectorArray = testPixelCovarianceVectorArray map { normalizedPixelVector =>
      new ArrayRealVector(eigenFaceArray map { eigenFace =>
        EigenFaces.computeImageWeightAgainstEigenFace(normalizedPixelVector, trainMeanPixelVector, eigenFace)
      })
    }

    /* The weights form a vector, where each weight represents a contribution of that eigenface image towards building the image */
    testFaceClassWeightVectorArray

    /* Need to stores copies of the eigenface images, the images used to train the eigenfaces, and the test images (divided into matched and unmatched)*/


    /* Write eigenfaces out to file for sanity checking */
    for((eigenFace, idx) <- eigenFaceArray.view.zipWithIndex) {
      val eigenFaceBufferedImage = ImageUtil.reconstructImage(eigenFace.faceMatrix.getColumn(0), IMAGE_WIDTH, IMAGE_HEIGHT)
      writeEigenface(eigenFaceBufferedImage, idx)
    }
  }
}

object FacialRecognition extends App {
  override def main(args: Array[String]): Unit = {
    new FacialRecognition().run
  }
}
