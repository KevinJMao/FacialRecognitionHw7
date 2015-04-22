package utils

import org.apache.commons.math3.linear.{Array2DRowRealMatrix, ArrayRealVector, EigenDecomposition, RealMatrix, RealVector}
import recognition.FacialRecognition

import scala.collection.mutable
import scala.math._

object MatrixHelpers {
  def computeWeightedVector(vectorWeightPairs : Array[(RealVector, Double)]) : RealVector = {
    val weightedVectors = vectorWeightPairs map { pair =>
      val vector = pair._1
      val weight = pair._2

      vector.mapMultiply(weight)
    }

    weightedVectors reduce { (prev, curr) => prev.add(curr) }
  }

  def computeMeanVector(pixelMatrix : RealMatrix) : RealVector = {
    val resultVector = new ArrayRealVector(pixelMatrix.getRowDimension)

    for(rowIndex <- 0 until pixelMatrix.getRowDimension) {
      resultVector.setEntry(rowIndex, pixelMatrix.getRow(rowIndex).toSeq.sum / pixelMatrix.getColumnDimension)
    }
    resultVector
  }

  def computePixelCovariantMatrix(pixelMatrix : RealMatrix, meanVector : RealVector) : RealMatrix = {
    val resultMatrix = new Array2DRowRealMatrix(pixelMatrix.getRowDimension, pixelMatrix.getColumnDimension)

    for(colIdx <- 0 until pixelMatrix.getColumnDimension) {
      resultMatrix.setColumnVector(
        colIdx, computePixelCovarianceVector(pixelMatrix.getColumnVector(colIdx), meanVector))
    }
    resultMatrix
  }

  def computePixelCovarianceVector(pixelVector : RealVector, meanVector : RealVector) : RealVector = {
    pixelVector.subtract(meanVector)
  }

  def computeEigenFaces(diffMatrix : RealMatrix) : Array[EigenFace] = {
    val eigenFaceBuffer = mutable.ArrayBuffer[EigenFace]()

    val shortMatrix_L = diffMatrix.transpose().multiply(diffMatrix)

    val eigenDecomposition_L =  new EigenDecomposition(shortMatrix_L)

    (0 until diffMatrix.getColumnDimension).foreach { i =>
      val eigenVectorAsMatrix = new Array2DRowRealMatrix(eigenDecomposition_L.getEigenvector(i).toArray)
      val rescaledEigenVectorAsMatrix = diffMatrix.multiply(eigenVectorAsMatrix)
      val face = EigenFace(eigenDecomposition_L.getRealEigenvalue(i), rescaledEigenVectorAsMatrix)
      eigenFaceBuffer += face
    }

    eigenFaceBuffer.sortBy(_.value).reverse.toArray
  }

  def computeVectorMagnitude(vector : RealVector) : Double = {
    pow(sqrt(vector.toArray.reduce((prev, next) => prev + next * next)), 2)
  }
}

case class EigenFace(value : Double, faceMatrix : RealMatrix) {
  lazy val reconstructedImage = ImageUtil.reconstructImage(faceMatrix.getColumn(0), FacialRecognition.IMAGE_WIDTH,
    FacialRecognition.IMAGE_HEIGHT)
}
