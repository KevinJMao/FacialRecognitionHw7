package recognition

object Boot extends App {
  override def main(args: Array[String]): Unit = {
    new FacialRecognition().run
  }
}

//astefa* mag=2.196e26 dist=1.482e13
//bird mag = 2.960e25 dist = 5.44e12
//clouds mag=8.16e25 dist=9.034e12
//kclar mag=2.060e26 dist=1.435e13
//npmitc mag=2.996e25 dist=5.473e12
//rmpugh* mag=7.077e25 dist=8.413e12
//pot mag=2.047e25 dist=4.524e12