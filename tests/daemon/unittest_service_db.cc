
/**
 * @file        unittest_service_db.cc
 * @date        21 Mar 2023
 * @brief       Unit test for Service DB used by ML Agent
 * @see         https://github.com/nnstreamer/api
 * @author      Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @bug         No known bugs
 */

#include <gio/gio.h>
#include <gtest/gtest.h>

#include "service-db.hh"

/**
 * @brief Negative test for set_pipeline. Empty name or description.
 */
TEST (serviceDB, set_pipeline_n)
{
  MLServiceDB &db = MLServiceDB::getInstance ();
  int gotException = 0;

  try {
    db.set_pipeline ("", "videotestsrc ! fakesink");
  } catch (const std::exception &e) {
    g_critical ("Got Exception: %s", e.what ());
    gotException = 1;
  }
  EXPECT_EQ (gotException, 1);

  gotException = 0;
  try {
    db.set_pipeline ("test_key", "");
  } catch (const std::exception &e) {
    g_critical ("Got Exception: %s", e.what ());
    gotException = 1;
  }
  EXPECT_EQ (gotException, 1);
}

/**
 * @brief Negative test for get_pipeline. Empty name.
 */
TEST (serviceDB, get_pipeline_n)
{
  MLServiceDB &db = MLServiceDB::getInstance ();
  int gotException = 0;

  try {
    std::string pipeline_description;
    db.get_pipeline ("", pipeline_description);
  } catch (const std::exception &e) {
    g_critical ("Got Exception: %s", e.what ());
    gotException = 1;
  }
  EXPECT_EQ (gotException, 1);

}

/**
 * @brief Negative test for delete_pipeline. Empty name.
 */
TEST (serviceDB, delete_pipeline_n)
{
  MLServiceDB &db = MLServiceDB::getInstance ();
  int gotException = 0;

  try {
    db.delete_pipeline ("");
  } catch (const std::exception &e) {
    g_critical ("Got Exception: %s", e.what ());
    gotException = 1;
  }
  EXPECT_EQ (gotException, 1);
}

/**
 * @brief Negative test for set_model. Empty name or model or description.
 */
TEST (serviceDB, set_model_n)
{
  MLServiceDB &db = MLServiceDB::getInstance ();
  int gotException = 0;

  try {
    guint version;
    db.set_model ("", "model", true, "description", &version);
  } catch (const std::exception &e) {
    g_critical ("Got Exception: %s", e.what ());
    gotException = 1;
  }
  EXPECT_EQ (gotException, 1);

  gotException = 0;
  try {
    guint version;
    db.set_model ("test", "", true, "description", &version);
  } catch (const std::exception &e) {
    g_critical ("Got Exception: %s", e.what ());
    gotException = 1;
  }
  EXPECT_EQ (gotException, 1);

  gotException = 0;
  try {
    guint version;
    db.set_model ("test", "model", true, "", &version);
  } catch (const std::exception &e) {
    g_critical ("Got Exception: %s", e.what ());
    gotException = 1;
  }
  EXPECT_EQ (gotException, 1);
}

/**
 * @brief Negative test for get_model. Empty name, invalid version.
 */
TEST (serviceDB, get_model_n)
{
  MLServiceDB &db = MLServiceDB::getInstance ();
  int gotException = 0;

  try {
    std::string model_description;
    db.get_model ("", model_description, 0);
  } catch (const std::exception &e) {
    g_critical ("Got Exception: %s", e.what ());
    gotException = 1;
  }
  EXPECT_EQ (gotException, 1);

  gotException = 0;
  try {
    std::string model_description;
    db.get_model ("test", model_description, -54321);
  } catch (const std::exception &e) {
    g_critical ("Got Exception: %s", e.what ());
    gotException = 1;
  }
  EXPECT_EQ (gotException, 1);
}

/**
 * @brief Negative test for update_model_description. Empty name or description.
 */
TEST (serviceDB, update_model_description_n)
{
  MLServiceDB &db = MLServiceDB::getInstance ();
  int gotException = 0;

  try {
    db.update_model_description ("", 0, "description");
  } catch (const std::exception &e) {
    g_critical ("Got Exception: %s", e.what ());
    gotException = 1;
  }
  EXPECT_EQ (gotException, 1);

  gotException = 0;
  try {
    db.update_model_description ("test", 0, "");
  } catch (const std::exception &e) {
    g_critical ("Got Exception: %s", e.what ());