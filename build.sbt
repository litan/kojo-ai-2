name := "kojo-ai-2"

version := "0.5"

scalaVersion := "2.13.6"

scalacOptions := Seq("-feature", "-deprecation")

libraryDependencies ++= Seq(
  "org.tensorflow" % "tensorflow-core-platform" % "0.5.0",
  "org.tensorflow" % "tensorflow-framework" % "0.5.0",
  "org.knowm.xchart" % "xchart" % "3.7.0",
  "org.apache.commons" % "commons-math3" % "3.6.1",
  "com.github.sarxos" % "webcam-capture" % "0.3.12"
)

//Build distribution
val distOutpath             = settingKey[File]("Where to copy all dependencies and kojo")
val buildDist  = taskKey[Unit]("Copy runtime dependencies and built kojo to 'distOutpath'")

lazy val dist = project
  .in(file("."))
  .settings(
    distOutpath              := baseDirectory.value / "dist",
    buildDist   := {
      val allLibs:                List[File]          = (Runtime / dependencyClasspath).value.map(_.data).filter(_.isFile).toList
      val buildArtifact:          File                = (Runtime / packageBin).value
      val jars:                   List[File]          = buildArtifact :: allLibs
      val `mappings src->dest`:   List[(File, File)]  = jars.map(f => (f, distOutpath.value / f.getName))
      val log                                         = streams.value.log
      log.info(s"Copying to ${distOutpath.value}:")
      log.info(s"${`mappings src->dest`.map(f => s" * ${f._1}").mkString("\n")}")
      IO.copy(`mappings src->dest`)
    }
  )
