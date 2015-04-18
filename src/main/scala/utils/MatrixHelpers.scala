package utils

import java.io.File
import javax.imageio.ImageIO
import scala.collection.JavaConversions._
import java.awt.image.{DataBufferByte, BufferedImage}
import cern.colt.matrix.DoubleMatrix2D
import cern.colt.matrix.impl.DenseDoubleMatrix2D
import cern.colt.matrix.linalg.EigenvalueDecomposition

import scala.collection.mutable

/**
 * Methods for converting image matrices to covariance matrices.
 *
 * @note Adapted from https://github.com/fredang/mahout-eigenface-example.
 */
object MatrixHelpers {

  /**
   * Merges a list of arrays of pixels for multiple images into a single matrix.
   * @param matrices
   * @param width
   * @param height
   * @return
   */
  def mergePixelMatrices(matrices: List[Array[Double]], width: Int, height: Int): Array[Array[Double]] = {
    val pixelLength = width * height

    val pixelMatrix = Array.ofDim[Double](pixelLength, matrices.size)

    var matrixNo = 0
    matrices.foreach { pixels =>
      (0 to (pixelLength-1)).foreach { pixelNo =>
        pixelMatrix(pixelNo)(matrixNo) = pixels(pixelNo)
      }
      matrixNo += 1
    }
    pixelMatrix
  }

  /**
   * Computes the mean for each column in a pixel matrix.
   * @param pixelMatrix
   * @return
   */
  def computeMeanColumn(pixelMatrix: Array[Array[Double]]): Array[Double] = {
    val meanColumn = new Array[Double](pixelMatrix.length)
    val columnCount = pixelMatrix(0).length

    (0 to (pixelMatrix.length-1)).foreach { i =>
      var sum: Double = 0.0
      (0 to (columnCount-1)).foreach { j =>
        sum += pixelMatrix(i)(j)
      }
      meanColumn(i) = sum / columnCount
    }
    meanColumn
  }

  /**
   * Computes a difference matrix.
   * @param pixelMatrix
   * @param meanColumn
   * @return
   */
  def computeDifferenceMatrixPixels(pixelMatrix: Array[Array[Double]], meanColumn: Array[Double]): Array[Array[Double]] = {
    val rowCount = pixelMatrix.length
    val columnCount = pixelMatrix(0).length

    val diffMatrixPixels = Array.ofDim[Double](rowCount, columnCount)

    (0 to (pixelMatrix.length-1)).foreach { i =>
      (0 to (columnCount-1)).foreach { j =>
        diffMatrixPixels(i)(j) = pixelMatrix(i)(j) - meanColumn(i)
      }
    }

    diffMatrixPixels
  }

  /**
   * Computes a covariance multi-dimensional array.
   * @param pixelMatrix
   * @return
   */
  def computeCovarianceMatrix(pixelMatrix: Array[Array[Double]], diffMatrixPixels: Array[Array[Double]]): Array[Array[Double]] = {
    val rowCount = pixelMatrix.length
    val columnCount = pixelMatrix(0).length

    val covarianceMatrix = Array.ofDim[Double](columnCount, columnCount)

    (0 to (columnCount-1)).foreach { i =>
      (0 to (columnCount-1)).foreach { j =>
        var sum: Double = 0.0
        (0 to (rowCount-1)).foreach { k =>
          sum += diffMatrixPixels(k)(i) * diffMatrixPixels(k)(j)
        }
        covarianceMatrix(i)(j) = sum
      }
    }

    covarianceMatrix
  }

  /**
   * Computes an Eigenvector matrix from a multi-dimensional covariance array.
   * @param covarianceMatrix
   * @return
   */
  def computeEigenVectors(covarianceMatrix: Array[Array[Double]]): DoubleMatrix2D = {
    val doubleMatrix = new DenseDoubleMatrix2D(covarianceMatrix.length, covarianceMatrix(0).length)
    doubleMatrix.assign(covarianceMatrix)
    val eigenValues = new EigenvalueDecomposition(doubleMatrix)
    eigenValues.getV
  }


  /**
   * Computes an Eigenvector matrix using principal component analysis
   * @param diffMatrix
   * @return
   */

  // diffMatrix = [50 * [36000]
  def shortcutComputeEigenVectors(diffMatrix : Array[Array[Double]]) : DoubleMatrix2D = {
    // Construct L = (PHI_M^T)(PHI_N)
    // (36000 x 1)(1 x 36000) = (1x1)

    val matrix_L = Array.ofDim[Double](diffMatrix.length, diffMatrix.length)

    for(i <- 0 until matrix_L.length){
      for(j <- 0 until matrix_L(i).length) {
        matrix_L(i)(j) = (diffMatrix(i), diffMatrix(j)).zipped.map(_*_).sum
      }
    }

    // eigenVectors_L = [50 x [50]]
    val eigenVectors_L = MatrixHelpers.computeEigenVectors(matrix_L)

    val diff2DMatrix = new DenseDoubleMatrix2D(diffMatrix).viewDice()
    var eigenVectorsFinal = mutable.ArrayBuffer[Array[Double]]()

    for(i <- 0 until eigenVectors_L.rows()) {
      // For each preliminary eigenvector
      val shortEigenVectorArr = Array[Array[Double]](eigenVectors_L.viewRow(i).toArray)
      val shortEigenVector2DMatrix = new DenseDoubleMatrix2D(shortEigenVectorArr).viewDice()
      val fullEigenvector2DMatrix = diff2DMatrix.zMult(shortEigenVector2DMatrix, null)
      eigenVectorsFinal += fullEigenvector2DMatrix.toArray.flatten

    }
    val eigenVectorsFinalArr = eigenVectorsFinal.toArray
    val eigenVectorsFinal2DMatrix = new DenseDoubleMatrix2D(eigenVectorsFinalArr)
    eigenVectorsFinal2DMatrix
  }

}
