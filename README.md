# CS 580 Homework 7 - Facial Recognition

A facial recognition program based on Eigenfaces.

# Documentation
* All 153 face images used are from the [Faces94](http://cswww.essex.ac.uk/mv/allfaces/faces94.html) face collection.
* 30 faces are randomly selected from the pool of 153 faces to be used as the training set.
* 30 faces are randomly selected to be used as the testing set.
* Only the top 5 Eigenfaces (largest associated Eigenvector) are selected for use in recognition.
* The maximum allowed distance an image may have from any face class (**&theta;<sub>k</sub>**) is equal to **5 x 10<sup>13</sup>** in order to qualify as a match.
* The maximum allowed distance an image may have to the face space (**&epsilon;**) is equal to **100**.

# Usage
## Compile
```
  ./sbt compile
```
## Run
```
  ./sbt run
```
The program will randomly select a preconfigured number of faces from the `faces/` directory. These faces will
constitute the set of training faces, and are placed under `images/trainingFaces`. A set of Eigenfaces are then
generated from the training set and are placed in `images/eigenFaces`.

Another sample of randomly selected images will be used for testing the algorithm. Test samples that the algorithm
determines to be a match are placed in `images/testFaces/matched`, while unrecognized samples will be placed in
`images/testFaces/unmatched`.
# Authors
[Kevin Mao](https://github.com/KevinJMao)

George Mason University, [kmao@masonlive.gmu.edu](mailto:kmao@masonlive.gmu.edu)

# References
[Eigenfaces for Recognition](http://www.face-rec.org/algorithms/PCA/jcn.pdf), Matthew Turk, Alex Pentland

