package recognition

object Boot extends App {
  override def main(args: Array[String]): Unit = {
    new FacialRecognition().run
  }
}