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
      in_res(nullptr), out_res(nullptr), input(nullptr), input2(nullptr), output(nullptr),
      root_path(nullptr), valid_model(nullptr)
  {
    for (int i = 0; i < ML_TENSOR_RANK_LIMIT; ++i)
      in_dim[i] = out_dim[i] = res_dim[i] = 1;
  }

  /**
   * @brief SetUp method for each test case
   */
  void SetUp () override
  {
    ml_tensors_info_create (&in_info);
    ml_tensors_info_create (&out_info);
    ml_tensors_info_create (&in_res);
    ml_tensors_info_create (&out_res);

    /* supposed to run test in build directory */
    root_path = g_getenv ("MLAPI_SOURCE_ROOT_PATH");
    if (root_path == NULL) {
      root_path = "..";
    }
    valid_model = GetVaildModelFile();
  }

  /**
   * @brief TearDown method for each test case
   */
  void TearDown () override
  {
    if (single_h) {
      ml_single_close (single_h);
      single_h = nullptr;
    }

    if (pipeline_h) {
      ml_pipeline_destroy (pipeline_h);
      pipeline_h = nullptr;
    }

    if (input)
      ml_tensors_data_destroy (input);

    if (input2)
      ml_tensors_data_destroy (input2);

    if (output)
      ml_tensors_data_destroy (output);

    ml_tensors_info_destroy (in_info);
    ml_tensors_info_destroy (out_info);
    ml_tensors_info_destroy (in_res);
    ml_tensors_info_destroy (out_res);
    g_free (const_cast<gchar *>(valid_model));
  }

  /**
   * @brief Signal handler for new data of tensor_sink element
   * @note This handler checks the number of received tensor counts.
   */
  static void
  cb_new_data (const ml_tensors_data_h data, const ml_tensors_info_h info, void *user_data)
  {
    int status;
    float *data_ptr;
    size_t data_size;
    int *checks = (int *)user_data;

    status = ml_tensors_data_get_tensor_data (data, 0, (void **)&data_ptr, &data_size);
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_FLOAT_EQ (*data_ptr, 12.0);

    *checks = *checks + 1;
  }

  /**
   * @brief Signal handler for new data of tensor_sink element and check its shape and payload.
   * @note This handler checks the rank, dimension, and payload of the received tensor.
   */
  static void
  cb_new_data_checker (const ml_tensors_data_h data, const ml_tensors_info_h info, void *user_data)
  {
    unsigned int cnt = 0;
    int status;
    float *data_ptr;
    size_t data_size;
    int *checks = (int *)user_data;
    ml_tensor_dimension out_dim;

    ml_tensors_info_get_count (info, &cnt);
    EXPECT_EQ (cnt, 1U);

    ml_tensors_info_get_tensor_dimension (info, 0, out_dim);
    EXPECT_EQ (out_dim[0], 1001U);
    EXPECT_EQ (out_dim[1], 1U);
    EXPECT_EQ (out_dim[2], 1U);
    EXPECT_EQ (out_dim[3], 1U);

    status = ml_tensors_data_get_tensor_data (data, 0, (void **)&data_ptr, &data_size);
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_EQ (data_size, 1001U);

    *checks = *checks + 1;
  }

  /**
   * @brief Synchronization function for the pipeline execution.
   */
  static void
  wait_for_sink (guint* call_cnt, const guint expected_cnt)
  {
    guint waiting_time = 0U;
    guint unit_time = 1000U * 1000; /* 1 second */
    gboolean done = FALSE;

    while (!done && waiting_time < unit_time * 10) {
      done = (*call_cnt >= expected_cnt);
      waiting_time += unit_time;
      if (!done)
        g_usleep (unit_time);
    }
    ASSERT_TRUE (done);
  }
};

/**
 * @brief Test nnfw subplugin with successful invoke (single ML-API)
 */
TEST_F (MLAPIInferenceNNFW, invoke_single_00)
{
  int status;
  unsigned int count = 0;
  float *data;
  size_t data_size;
  ml_tensor_type_e type = ML_TENSOR_TYPE_UNKNO