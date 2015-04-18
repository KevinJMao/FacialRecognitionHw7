package utils

import akka.actor._
import scala.concurrent.duration._
import org.mapdb._
import java.io.File


/**
 * App database backed by disk.
 */
object MapDB {

  /**
   * Storage files.
   */
  private val dbDirectory = new File("data").mkdir()
  private val dbFile = new File("data/faceRecData")

  /**
   * The database context.
   */
  val db = {
    // create the DB
    DBMaker.newFileDB(dbFile)
      .closeOnJvmShutdown()
      .make()
  }
}
