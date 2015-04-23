# CS 580 Homework 7 - Facial Recognition

A facial recognition program based on Eigenfaces.

# Documentation


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