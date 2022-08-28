/**
 * @file        unittest_capi_inference_nnfw_runtime.cc
 * @date        07 Oct 2019
 * @brief       Unit test for NNFW (ONE) tensor filter plugin with ML API.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <nnstreamer.h>
#include <nnstreamer-single.h>
#include <ml-api-internal.h>
#include <ml-api-inference-internal.h>
#include <ml-api-inference-pipeline-internal.h>

/**
 * @brief Test Fixture class for ML API of the NNFW Inference
 */
class MLAPIInferenceNNFW : public ::testing::Test
{
protected:
  ml_single_h single_h;
  ml_pipeline_h pipeline_h;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_info_h in_res, out_res;
  ml_tensor_dimension in_dim, out_dim, res_dim;
  ml_tensors_data_h input, input2, output;
  const gchar *root_path;
  const gchar *valid_model;

  /**
   * @brief Get the valid model file for NNFW test
   * @return gchar* the path of the model file
   */
  gchar *GetVaildModelFile () {
    gchar *model_file;

    /* nnfw needs a directory with model file and metadata in that directory */
    g_autofree gchar *model_path = g_build_filename (root_path, "tests", "test_models", "models", NULL);

    g_autofree gchar *meta_file = g_build_filename (model_path, "metadata", "MANIFEST", NULL);
    if (!g_file_test (meta_file, G_FILE_TEST_EXISTS)) {
      return NULL;
    }

    model_file = g_build_filename (model_path, "add.tflite", NULL);
    if (!g_file_test (model_file, G_FILE_TEST_EXISTS)) {
      g_free (model_file);
      return NULL;
    }
    return model_file;
  }

protected:
  /**
   * @brief Construct a new MLAPIInferenceNNFW object
   */
  MLAPIInferenceNNFW ()
    : single_h(nullptr), pipeline_h(nullptr), in_info(nullptr), out_info(nullptr),
      in_res(nullp