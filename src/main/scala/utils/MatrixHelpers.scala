package utils

import cern.colt.matrix.DoubleMatrix2D
import cern.colt.matrix.impl.DenseDoubleMatrix2D
import cern.colt.matrix.linalg.EigenvalueDecomposition
import org.apache.commons.math3.linear.{Array2DRowRealMatrix, ArrayRealVector, EigenDecomposition, RealMatrix, RealVector}

import scala.collection.mutable

/**
 * Methods for converting image matrices to covariance matrices.
 *
 * @note Adapted from https://github.com/fredang/mahout-eigenface-example.
 */
object MatrixHelpers {
  /**
   * Compute the mean vector of an M X N matrix, returning a M x 1 vector
   * @param pixelMatrix
   * @return
   */
  def computeMeanVector(pixelMatrix : RealMatrix) : RealVector = {
    val resultVector = new ArrayRealVector(pixelMatrix.getRowDimension)

    for(rowIndex <- 0 until pixelMatrix.getRowDimension) {
      resultVector.setEntry(rowIndex, pixelMatrix.getRow(rowIndex).toSeq.sum / pixelMatrix.getColumnDimension)
    }
    resultVector
  }

  def computeDifferenceMatrixPixels(pixelMatrix : RealMatrix, meanVector : RealVector) : RealMatrix = {
    val resultMatrix = new Array2DRowRealMatrix(pixelMatrix.getRowDimension, pixelMatrix.getColumnDimension)

    /* For each sample */
    for(colIdx <- 0 until pixelMatrix.getColumnDimension) {
      resultMatrix.setColumnVector(
        colIdx, computeDifferenceVectorPixels(pixelMatrix.getColumnVector(colIdx), meanVector))
    }
    resultMatrix
  }

  def computeDifferenceVectorPixels(pixelVector : RealVector, meanVector : RealVector) : RealVector = {
    pixelVector.subtract(meanVector)
  }


  def computeEigenFaces(diffMatrix : RealMatrix) : Array[EigenFace] = {
    val eigenFaceBuffer = mutable.ArrayBuffer[EigenFace]()


    // Compute Psi^T * Psi (50 x 50)
    val shortMatrix_L = diffMatrix.transpose().multiply(diffMatrix)

    // Compute EigenValue decomposition
    val eigenDecomposition_L =  new EigenDecomposition(shortMatrix_L)


    // A.premultiply(B) => BA, A.multiply(C) => AC
    (0 until diffMatrix.getColumnDimension).foreach { i =>
      //[10000 x 50] * [50 x 1] = [10000 x 1]
      val eigenVectorAsMatrix = new Array2DRowRealMatrix(eigenDecomposition_L.getEigenvector(i).toArray)
      val rescaledEigenVectorAsMatrix = diffMatrix.multiply(eigenVectorAsMatrix)
      val face = EigenFace(eigenDecomposition_L.getRealEigenvalue(i), rescaledEigenVectorAsMatrix)
      eigenFaceBuffer += face

    }

    eigenFaceBuffer.sortBy(_.value).reverse.toArray
  }
}

case class EigenFace(value : Double, faceMatrix : RealMatrix)
