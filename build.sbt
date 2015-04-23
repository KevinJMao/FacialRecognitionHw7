name := "FacialRecognitionHw7"

version := "1.0"

scalaVersion := "2.10.4"

resolvers += "Sonatype snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/"

libraryDependencies ++= Seq(
  "com.typesafe.akka"             %% "akka-actor"                 % "2.2.3",
  "com.typesafe.akka"             %% "akka-slf4j"                 % "2.2.3",
  "org.imgscalr"                  %  "imgscalr-lib"               % "4.2",
  "com.google.guava"              % "guava"                       % "18.0",
  "org.apache.commons"            % "commons-math3"               % "3.2"
)

initialize := {
  val required = "1.7"
  val current  = sys.props("java.specification.version")
  assert(current == required, s"Unsupported JDK: java.specification.version $current != $required")
}

mainClass in(Compile, run) := Some("recognition.Boot")

val buildSettings = Defaults.defaultSettings ++ Seq(
  javaOptions += "-Xmx8G"
)

addSbtPlugin("com.github.mpeltonen" % "sbt-idea" % "1.7.0-SNAPSHOT")