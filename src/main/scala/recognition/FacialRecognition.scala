package recognition

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

import org.apache.commons.math3.linear.{RealMatrix, RealVector, ArrayRealVector, Array2DRowRealMatrix}
import org.slf4j.LoggerFactory
import utils.{EigenFace, MatrixHelpers, EigenFaces, ImageUtil}

import scala.util.{Failure, Random, Success, Try}

class FacialRecognition{
  private val logger = LoggerFactory.getLogger(classOf[FacialRecognition])
  private var eigenface : Option[BufferedImage] = None


  private val TRAINING_SAMPLE = 20
  private val MATCH_AGAINST_X_FACES = 10
  private val MAX_ALLOWABLE_FACE_CLASS_DISTANCE = 3.14
  private val RNG = Random
  private val MAX_PROJECTION_MAGNITUDE = 2.17

  private val EIGENFACE_FILE = "eigenface"

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
    writeImage(image, "images/eigenFaces/" + EIGENFACE_FILE + "_" + indexNumber + ".jpg")
  }

  private def writeImage(image : BufferedImage, location : String) = {
    Try {
      ImageIO.write(image, "jpg", new File(location))
    } recover {
      case e : Throwable => logger.error("Exception thrown while writing image: {}" , e)
    }
  }

  private def selectRandomFaceImage : Option[FaceImage] = {
    val imageFile = new File("faces").listFiles()(RNG.nextInt(new File("faces").listFiles().size))
    loadImage(imageFile) map { image =>
      FaceImage(imageFile.getName, image)
    }
  }

  private def selectNRandomImages(n : Int) : Array[FaceImage] = {
    (for {
      i <- 1 to n
    } yield selectRandomFaceImage match {
        case Some(faceImage) => faceImage
        case None =>
          logger.error("Error while loading random training image.")
          FaceImage("Unavailable Image", new BufferedImage(0, 0, BufferedImage.TYPE_CUSTOM))
      }).toArray
  }

  private def emptyDirectory(topLevelDirectory : File) : Int = {
    var counter = 0
    topLevelDirectory.listFiles().foreach {
      case file : File if file.isDirectory => counter += emptyDirectory(file)
      case other : File =>
        other.delete()
        counter += 1
    }
    counter
  }

  def run() = {
    /* Our set of training faces */
    val trainFaceImages : Array[FaceImage] = selectNRandomImages(TRAINING_SAMPLE)

    /* An array of the top SELECT_TOP_N_EIGENFACES to use as comparison (u_i) */
    val eigenFaceArray : Array[EigenFace] = EigenFaces.computeEigenFaces(trainFaceImages.toArray)

    /* A matrix of the pixel intensity values of each image. Each column corresponds to a single image. (Set of all _GAMMA_) */
    val trainPixelMatrix : Array2DRowRealMatrix = EigenFaces.convertImagesToPixelMatrix(trainFaceImages)

    /* The mean column-wise vector of the normalized pixel matrix (PSI) */
    val trainMeanPixelVector : RealVector = MatrixHelpers.computeMeanVector(trainPixelMatrix)

    /* The set of class vectors for each individual image (_OMEGA_k) */
    val trainClassPatternVectors : Array[(FaceImage, RealVector)] = trainFaceImages map { faceImage =>
      (faceImage, EigenFaces.computeFaceClassWeightVector(faceImage, trainMeanPixelVector, eigenFaceArray))
    }
    /* Our set of testing faces */
    val testFaceImages : Array[FaceImage] = selectNRandomImages(MATCH_AGAINST_X_FACES)

    /* The pattern vector for every test face image (_OMEGA_) */
    val testFacePatternVectors : Array[(FaceImage, RealVector)] = testFaceImages map { faceImage =>
      (faceImage, EigenFaces.computeFaceClassWeightVector(faceImage, trainMeanPixelVector, eigenFaceArray))
    }

    /* For each new face image to be identified, calculate its pattern vector _Omega_, the distances _epsilon_i_ to each known class, and the distance _epsilon_ to the face space */
    testFacePatternVectors foreach { testPatternVectorPair =>
      /* The minimum _EPSILON_k we could find */
      val classVectorMagnitude = (trainClassPatternVectors map { trainPatternVectorPair =>
        (testPatternVectorPair._1, MatrixHelpers.computeVectorMagnitude(trainPatternVectorPair._2.subtract(trainPatternVectorPair._2)))
      }).sortBy(_._2).reverse.head

      /* Normalized pixel vector of the test image */
      val normalizedPixelVector = testPatternVectorPair._2.subtract(trainMeanPixelVector
      )
      /* Weighted sum of all of the eigenface vectors */
      val weightedEigenFaceVector = MatrixHelpers.computeWeightedVector(
        eigenFaceArray map {
          eigenFace => eigenFace.faceMatrix.getColumnVector(0)
        } zip testPatternVectorPair._2.toArray)

      /* _EPSILON_ */
      val faceSpaceProjectionMagnitude = MatrixHelpers.computeVectorMagnitude(weightedEigenFaceVector)


      if(faceSpaceProjectionMagnitude < MAX_PROJECTION_MAGNITUDE
        && classVectorMagnitude._2 < MAX_ALLOWABLE_FACE_CLASS_DISTANCE) {
        writeImage(testPatternVectorPair._1.image, "images/testFaces/matched/" + testPatternVectorPair._1.fileName)
      }
      else {
        writeImage(testPatternVectorPair._1.image, "images/testFaces/unmatched/" + testPatternVectorPair._1.fileName)
      }

    }

    /* Need to stores copies of the eigenface images, the images used to train the eigenfaces, and the test images (divided into matched and unmatched)*/


    /* Write eigenfaces out to file for sanity checking */
    for((eigenFace, idx) <- eigenFaceArray.view.zipWithIndex) {
      val eigenFaceBufferedImage = ImageUtil.reconstructImage(eigenFace.faceMatrix.getColumn(0),
        FacialRecognition.IMAGE_WIDTH,
        FacialRecognition.IMAGE_HEIGHT)
      writeEigenface(eigenFaceBufferedImage, idx)
    }
  }
}

case class FaceImage(fileName : String, image : BufferedImage) {
  //pixelMatrix
  def pixelVector : RealVector = new ArrayRealVector(
    ImageUtil.getNormalizedImagePixels(image, FacialRecognition.IMAGE_WIDTH, FacialRecognition.IMAGE_HEIGHT))
}

object FacialRecognition {
  val SELECT_TOP_N_EIGENFACES = 5
  val IMAGE_WIDTH = 180
  val IMAGE_HEIGHT = 200
}
