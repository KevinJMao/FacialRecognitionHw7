package utils

import org.apache.commons.math3.linear.{Array2DRowRealMatrix, ArrayRealVector, RealMatrix, RealVector}
import recognition.{FaceImage, FacialRecognition}

/**
 * Helper object for creating EigenFaces from matrices.
 */
object EigenFaces {
  def computeWeightedVector(vectorWeightPairs : Array[(RealVector, Double)]) : RealVector = {
    vectorWeightPairs map {
      pair => pair._1.mapMultiply(pair._2)
    } reduce  { (prev, curr) => prev.add(curr) }
  }

  /* Compute _OMEGA_ */
  def computeFaceClassWeightVector(faceImage : FaceImage,
                                   meanPixelVector : RealVector,
                                   eigenFaces : Array[EigenFace]) : RealVector = {
    val pixelVector = new ArrayRealVector(ImageUtil.getNormalizedImagePixels(faceImage.image,
      FacialRecognition.IMAGE_WIDTH,
      FacialRecognition.IMAGE_HEIGHT))

    val normalizedPixelVector = pixelVector.subtract(meanPixelVector)

    val faceClassWeightVector = eigenFaces map { eigenFace =>
      EigenFaces.computeImageWeightAgainstEigenFace(normalizedPixelVector, meanPixelVector, eigenFace)
    }

    new ArrayRealVector(faceClassWeightVector)
  }

  def computeImageWeightAgainstEigenFace(imagePixelVector: RealVector, trainMeanVector: RealVector, eigenFace: EigenFace): Double = {
    val normalizedImagePixelVector = imagePixelVector.subtract(trainMeanVector)
    val normalizedImagePixelMatrix = new Array2DRowRealMatrix(normalizedImagePixelVector.toArray)

    val resultMatrix = eigenFace.faceMatrix.transpose.multiply(normalizedImagePixelMatrix)
    resultMatrix.getEntry(0, 0)
  }

  def convertImagesToPixelMatrix(faceImages : Array[FaceImage]) : RealMatrix = {
    new Array2DRowRealMatrix(faceImages.toArray.map { face =>
      ImageUtil.getNormalizedImagePixels(face.image, FacialRecognition.IMAGE_WIDTH, FacialRecognition.IMAGE_HEIGHT)
    }).transpose
  }

  def computeEigenFaces(trainFaceImages : Array[FaceImage]) : Array[EigenFace] = {
    val trainPixel2DArray = trainFaceImages.toArray.map { faceImage =>
      ImageUtil.getNormalizedImagePixels(faceImage.image, FacialRecognition.IMAGE_WIDTH, FacialRecognition.IMAGE_HEIGHT)
    }

    val trainPixelMatrix = new Array2DRowRealMatrix(trainPixel2DArray).transpose()

    val trainMeanPixelVector = MatrixHelpers.computeMeanVector(trainPixelMatrix)

    computeEigenFaces(trainPixelMatrix, trainMeanPixelVector).take(FacialRecognition.SELECT_TOP_N_EIGENFACES)
  }

  private def computeEigenFaces(pixelMatrix : RealMatrix, meanColumn : RealVector) : Array[EigenFace] = {

    // (M x N) = (36000 x 50)
    val diffMatrix : RealMatrix = MatrixHelpers.computePixelCovariantMatrix(pixelMatrix, meanColumn)

    // (M x N) = (36000 x 50), each column is an eigenvector
    val eigenFaces : Array[EigenFace] = MatrixHelpers.computeEigenFaces(diffMatrix)

    eigenFaces
  }
}
