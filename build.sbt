name := "kojo-ai"

version := "0.5.1"

scalaVersion := "2.13.6"

scalacOptions := Seq("-feature", "-deprecation")

val javacppVersion = "1.5.9"

// Platform classifier for native library dependencies
val platform = org.bytedeco.javacpp.Loader.Detector.getPlatform

// JavaCPP-Preset libraries with native dependencies
val presetLibs = Seq(
  "opencv"   -> "4.7.0",
  "ffmpeg"   -> "6.0",
  "openblas" -> "0.3.23"
).flatMap { case (lib, ver) =>
  Seq(
    "org.bytedeco" % lib % s"$ver-$javacppVersion",
    "org.bytedeco" % lib % s"$ver-$javacppVersion" classifier platform
  )
}

libraryDependencies ++= Seq(
  "org.tensorflow" % "tensorflow-core-platform" % "0.5.0",
  "org.tensorflow" % "tensorflow-framework" % "0.5.0",
  "org.knowm.xchart" % "xchart" % "3.7.0",
  "org.apache.commons" % "commons-math3" % "3.6.1",
  "org.bytedeco" % "javacpp" % javacppVersion,
  "org.bytedeco" % "javacpp" % javacppVersion classifier platform,
  "org.bytedeco" % "javacv" % javacppVersion,
) ++ presetLibs

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
