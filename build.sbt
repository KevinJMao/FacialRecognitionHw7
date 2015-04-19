name := "FacialRecognitionHw7"

version := "1.0"

scalaVersion := "2.10.4"

resolvers += "Sonatype snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/"

libraryDependencies ++= Seq(
  "edu.stanford.nlp"              %  "stanford-corenlp"           % "3.3.1",
  "edu.stanford.nlp"              %  "stanford-corenlp"           % "3.3.1" classifier "models",
  "org.mapdb"                     %  "mapdb"                      % "1.0.6",
  "com.cloudphysics"              %% "jerkson"                    % "0.6.3",
  "com.fasterxml.jackson.module"  %  "jackson-module-scala_2.10"  % "2.4.4",
  "org.fusesource.jansi"          %  "jansi"                      % "1.11",
  "org.apache.spark"              %% "spark-core"                 % "1.1.0",
  "com.typesafe.akka"             %% "akka-actor"                 % "2.2.3",
  "com.typesafe.akka"             %% "akka-slf4j"                 % "2.2.3",
  "org.imgscalr"                  %  "imgscalr-lib"               % "4.2",
  "net.sourceforge.parallelcolt"  %  "parallelcolt"               % "0.10.0",
  "com.google.guava"              % "guava"                       % "18.0",
  "org.apache.commons"            % "commons-math3"               % "3.2"
)

initialize := {
  val required = "1.7"
  val current  = sys.props("java.specification.version")
  assert(current == required, s"Unsupported JDK: java.specification.version $current != $required")
}

mainClass in (Compile, run) := Some("recognition.FacialRecognition")

val buildSettings = Defaults.defaultSettings ++ Seq(
  javaOptions += "-Xmx8G"
)

addSbtPlugin("com.github.mpeltonen" % "sbt-idea" % "1.7.0-SNAPSHOT")