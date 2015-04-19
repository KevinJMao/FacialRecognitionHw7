package utils

import org.apache.commons.math3.linear.{RealVector, RealMatrix, Array2DRowRealMatrix}

import scala.collection.JavaConversions._
import cern.colt.matrix.DoubleMatrix2D
import cern.colt.matrix.impl.DenseDoubleMatrix2D

/**
 * Helper object for creating EigenFaces from matrices.
 */
object EigenFaces {

  def computeEigenFaces_2(pixelMatrix : RealMatrix, meanColumn : RealVector) : Array[EigenFace] = {

    // (M x N) = (36000 x 50)
    val diffMatrix : RealMatrix = MatrixHelpers.computeDifferenceMatrixPixels_2(pixelMatrix, meanColumn)

    // (M x N) = (36000 x 50), each column is an eigenvector
    val eigenFaces : Array[EigenFace] = MatrixHelpers.computeEigenFaces(diffMatrix)

    eigenFaces
  }

  /**
   * Calculates a distance score between a mean Pixels/EigenFaces model in comparison to an image subject.
   * @param meanPixels
   * @param eigenFaces
   * @param subjectPixels
   */
  def computeDistance(meanPixels: Array[Double], eigenFaces: DoubleMatrix2D, subjectPixels: Array[Double]): Double = {
    val diffPixels = computeDifferencePixels(subjectPixels, meanPixels)
    val weights = computeWeights(diffPixels, eigenFaces)
    val reconstructedEigenPixels = reconstructImageWithEigenFaces(weights, eigenFaces, meanPixels)
    computeImageDistance(subjectPixels, reconstructedEigenPixels)
  }

  /**
   * Computes the distance between two images.
   * @param pixels1
   * @param pixels2
   */
  private def computeImageDistance(pixels1: Array[Double], pixels2: Array[Double]): Double = {
    var distance = 0.0
    val pixelCount = pixels1.length
    (0 to (pixelCount-1)).foreach { i =>
      var diff = pixels1(i) - pixels2(i)
      distance += diff * diff
    }
    Math.sqrt(distance / pixelCount)
  }

  /**
   * Computes the weights of faces vs. EigenFaces.
   * @param diffImagePixels
   * @param eigenFaces
   */
  private def computeWeights(diffImagePixels: Array[Double], eigenFaces: DoubleMatrix2D): Array[Double] = {
    val pixelCount = eigenFaces.rows()
    val eigenFaceCount = eigenFaces.columns()

    val weights = new Array[Double](eigenFaceCount)
    (0 to (eigenFaceCount-1)).foreach { i=>
      (0 to (pixelCount-1)).foreach { j =>
        weights(i) += diffImagePixels(j) * eigenFaces.get(j,i)
      }
    }
    weights
  }

  /**
   * Computes the difference pixels between a subject image and a mean image.
   * @param subjectPixels
   * @param meanPixels
   */
  private def computeDifferencePixels(subjectPixels: Array[Double], meanPixels: Array[Double]): Array[Double] = {
    val pixelCount = subjectPixels.length
    val diffPixels = new Array[Double](pixelCount)

    (0 to (pixelCount-1)).foreach { i =>
      diffPixels(i) = subjectPixels(i) - meanPixels(i)
    }
    diffPixels
  }

  /**
   * Reconstructs an image using Eigen Faces and weights.
   * @param weights
   * @param eigenFaces
   * @param meanPixels
   */
  private def reconstructImageWithEigenFaces(weights: Array[Double], eigenFaces: DoubleMatrix2D, meanPixels: Array[Double]) = {
    val pixelCount = eigenFaces.rows()
    val eigenFaceCount = eigenFaces.columns()

    // reconstruct image from weight and eigenfaces
    val reconstructedPixels = new Array[Double](pixelCount)
    (0 to (eigenFaceCount-1)).foreach { i =>
      (0 to (pixelCount-1)).foreach { j =>
        reconstructedPixels(j) += weights(i) * eigenFaces.get(j, i)
      }
    }

    // add mean
    (0 to (pixelCount-1)).foreach { i =>
      reconstructedPixels(i) += meanPixels(i)
    }

    var min = Double.MaxValue
    var max = -Double.MaxValue
    (0 to (reconstructedPixels.length-1)).foreach { i =>
      min = Math.min(min, reconstructedPixels(i))
      max = Math.max(max, reconstructedPixels(i))
    }

    val normalizedReconstructedPixels = new Array[Double](pixelCount)
    (0 to (reconstructedPixels.length-1)).foreach { i=>
      normalizedReconstructedPixels(i) = (255.0 * (reconstructedPixels(i) - min)) / (max - min)
    }
    normalizedReconstructedPixels
  }

}
