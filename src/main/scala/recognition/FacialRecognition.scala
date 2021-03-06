package recognition

import java.awt.image.BufferedImage
import java.io.{FileFilter, File}
import javax.imageio.ImageIO

import org.apache.commons.math3.linear.{ArrayRealVector, RealMatrix, RealVector}
import org.slf4j.LoggerFactory
import utils.{EigenFace, EigenFaces, ImageUtil, MatrixHelpers}

import scala.math.sqrt
import scala.util.{Failure, Random, Success, Try}

class FacialRecognition {
  private val logger = LoggerFactory.getLogger(classOf[FacialRecognition])
  private val MAX_ALLOWABLE_FACE_CLASS_DISTANCE = 2e13
  // Corresponds to Epsilon in Turk,Pentland
  private val RNG = Random
  private val MAX_PROJECTION_MAGNITUDE = 5e13
  // Corresponds to Epsilon_k in Turk,Pentland
  private val EIGENFACE_FILE = "eigenface"
  private var eigenface: Option[BufferedImage] = None

  private val TRAINING_FACES_DIRECTORY: String = "trainingFaces"
  private val TESTING_FACES_DIRECTORY: String = "testingFaces"


  def run() = {
    logger.info("Resetting output directories...")
    createDirectoriesIfNotExists()
    resetDirectories(new File("images"))

    /* Our set of training faces */
    logger.info("Retrieving training images from {} directory", TRAINING_FACES_DIRECTORY)
    val trainFaceImages: Array[FaceImage] = retrieveFacesFromDirectory(TRAINING_FACES_DIRECTORY)

    logger.info("Retrieving testing images from {} directory", TESTING_FACES_DIRECTORY)
    /* Our set of testing faces */
    val testFaceImages: Array[FaceImage] = retrieveFacesFromDirectory(TESTING_FACES_DIRECTORY)

    logger.info("Generating Eigefaces...")
    /* An array of the top SELECT_TOP_N_EIGENFACES to use as comparison (u_i) */
    val eigenFaceArray: Array[EigenFace] = EigenFaces.computeEigenFaces(trainFaceImages.toArray)
    writeEigenFacesToFile(eigenFaceArray)


    logger.info("Running recognition algorithm against test faces...")
    /* A matrix of the pixel intensity values of each image. Each column corresponds to a single image. (Set of all _GAMMA_) */
    val trainPixelMatrix: RealMatrix = EigenFaces.convertImagesToPixelMatrix(trainFaceImages)

    /* The mean column-wise vector of the normalized pixel matrix (PSI) */
    val trainMeanPixelVector: RealVector = MatrixHelpers.computeMeanVector(trainPixelMatrix)

    /* The set of class vectors for each individual image (_OMEGA_k) */
    val trainClassPatternVectors: Array[(FaceImage, RealVector)] = trainFaceImages map { faceImage =>
      (faceImage, EigenFaces.computeFaceClassWeightVector(faceImage, trainMeanPixelVector, eigenFaceArray))
    }

    /* The pattern vector for every test face image (_OMEGA_) */
    val testFacePatternVectors: Array[(FaceImage, RealVector)] = testFaceImages map { faceImage =>
      (faceImage, EigenFaces.computeFaceClassWeightVector(faceImage, trainMeanPixelVector, eigenFaceArray))
    }

    /* For each new face image to be identified, calculate its pattern vector _Omega_, the distances _epsilon_i_ to each known class, and the distance _epsilon_ to the face space */
    testFacePatternVectors foreach { testPatternVectorPair =>
      /* The minimum _EPSILON_k we could find */
      val classVectors: Array[(FaceImage, Double)] = (trainClassPatternVectors map { trainPatternVectorPair =>
        (trainPatternVectorPair._1, MatrixHelpers.computeVectorMagnitude(testPatternVectorPair._2.subtract(trainPatternVectorPair._2)))
      }).sortBy(_._2)
      val classVectorMagnitude = classVectors.head

      /* Normalized pixel vector of the test image */
      val testFaceImagePixelVector = new ArrayRealVector(ImageUtil.getNormalizedImagePixels(testPatternVectorPair._1.image, FacialRecognition.IMAGE_WIDTH, FacialRecognition.IMAGE_HEIGHT))
      val testNormalizedImagePixelVector = testFaceImagePixelVector.subtract(trainMeanPixelVector)

      /* Weighted sum of all of the eigenface vectors */
      val weightedEigenFaceVector = MatrixHelpers.computeWeightedVector(
        eigenFaceArray map {
          eigenFace => eigenFace.faceMatrix.getColumnVector(0)
        } zip testPatternVectorPair._2.toArray)

      /* _EPSILON_ */
      val faceSpaceProjectionMagnitude =
        MatrixHelpers.computeVectorMagnitude(testNormalizedImagePixelVector.subtract(weightedEigenFaceVector))

      val faceSpaceProjectionDistance = sqrt(faceSpaceProjectionMagnitude)


      if (faceSpaceProjectionDistance < MAX_PROJECTION_MAGNITUDE
        && classVectorMagnitude._2 < MAX_ALLOWABLE_FACE_CLASS_DISTANCE) {
        logger.info("{} resulted in MATCH.", testPatternVectorPair._1.fileName)
        writeImage(testPatternVectorPair._1.image, "images/testFaces/matched/" + testPatternVectorPair._1.fileName)
      }
      else {
        logger.info("{} resulted in NO MATCH.", testPatternVectorPair._1.fileName)
        writeImage(testPatternVectorPair._1.image, "images/testFaces/unmatched/" + testPatternVectorPair._1.fileName)
      }
    }
  }

  def writeFaceImagesToFile(trainFaceImages: Array[FaceImage], folder: String) {
    trainFaceImages foreach { faceImage =>
      writeImage(faceImage.image, folder + faceImage.fileName)
    }
  }

  private def loadImage(file: File): Option[BufferedImage] = {
    Try {
      ImageIO.read(file)
    } match {
      case Success(image) => Some(image)
      case Failure(e) =>
        logger.error("Exception thrown while loading image: {}", e)
        None
    }
  }

  private def writeEigenface(image: BufferedImage, indexNumber: Int) = {
    writeImage(image, "images/eigenFaces/" + EIGENFACE_FILE + "_" + indexNumber + ".jpg")
  }

  private def writeImage(image: BufferedImage, location: String) = {
    Try {
      ImageIO.write(image, "jpg", new File(location))
    } recover {
      case e: Throwable => logger.error("Exception thrown while writing image: {}", e)
    }
  }

  private def selectRandomFaceImage: Option[FaceImage] = {
    val imageFile = new File("faces").listFiles()(RNG.nextInt(new File("faces").listFiles().size))
    loadImage(imageFile) map { image =>
      FaceImage(imageFile.getName, image)
    }
  }

  private def selectNRandomImages(n: Int): Array[FaceImage] = {
    (for {
      i <- 1 to n
    } yield selectRandomFaceImage match {
        case Some(faceImage) => faceImage
        case None =>
          logger.error("Error while loading random training image.")
          FaceImage("Unavailable Image", new BufferedImage(0, 0, BufferedImage.TYPE_CUSTOM))
      }).toArray
  }

  private def retrieveFacesFromDirectory(dir : String) : Array[FaceImage] = {
    (for {
        file <- new File(dir).listFiles(new FileFilter {
          override def accept(pathname: File): Boolean = !pathname.getName.endsWith(".DS_Store")
        })
      } yield loadImage(file).map(image => FaceImage(file.getName, image)) match {
        case Some(faceImage) => faceImage
        case None =>
          logger.error("Error while loading random training image.")
          FaceImage("Unavailable Image", new BufferedImage(0, 0, BufferedImage.TYPE_CUSTOM))
      }).toArray
  }

  private def createDirectoriesIfNotExists() = {
    Seq(TRAINING_FACES_DIRECTORY, TESTING_FACES_DIRECTORY, "images", "images/eigenFaces", "images/testFaces/matched",
      "images/testFaces/unmatched") foreach { path =>
      val dir = new File(path)
      dir.exists match {
        case false => dir.mkdir()
        case _ =>
      }
    }
  }

  private def resetDirectories(topLevelDirectory: File): Int = {
    var counter = 0
    topLevelDirectory.listFiles().foreach {
      case file: File if file.isDirectory => counter += resetDirectories(file)
      case other: File =>
        other.delete()
        counter += 1
    }
    counter
  }

  private def writeEigenFacesToFile(eigenFaceArray: Array[EigenFace]) = {
    /* Write eigenfaces out to file for sanity checking */
    for ((eigenFace, idx) <- eigenFaceArray.view.zipWithIndex) {
      val eigenFaceBufferedImage = ImageUtil.reconstructImage(eigenFace.faceMatrix.getColumn(0),
        FacialRecognition.IMAGE_WIDTH,
        FacialRecognition.IMAGE_HEIGHT)
      writeEigenface(eigenFaceBufferedImage, idx)
    }
  }
}

case class FaceImage(fileName: String, image: BufferedImage) {
  //pixelMatrix
  def pixelVector: RealVector = new ArrayRealVector(
    ImageUtil.getNormalizedImagePixels(image, FacialRecognition.IMAGE_WIDTH, FacialRecognition.IMAGE_HEIGHT))
}

object FacialRecognition {
  val SELECT_TOP_N_EIGENFACES = 10
  val IMAGE_WIDTH = 180
  val IMAGE_HEIGHT = 200
}
