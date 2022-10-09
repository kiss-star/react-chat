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
  ml_tensor_type_e type = ML_TENSOR_TYPE_UNKNOWN;

  ASSERT_TRUE (valid_model != nullptr);

  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  status = ml_single_open (
      &single_h, valid_model, in_info, out_info, ML_NNFW_TYPE_NNFW, ML_NNFW_HW_CPU);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* let's ignore timeout (30 sec) */
  status = ml_single_set_timeout (single_h, 30000);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* input tensor in filter */
  status = ml_single_get_input_info (single_h, &in_res);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count (in_res, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (count, 1U);

  status = ml_tensors_info_get_tensor_type (in_res, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_FLOAT32);

  ml_tensors_info_get_tensor_dimension (in_res, 0, res_dim);
  EXPECT_TRUE (in_dim[0] == res_dim[0]);
  EXPECT_TRUE (in_dim[1] == res_dim[1]);
  EXPECT_TRUE (in_dim[2] == res_dim[2]);
  EXPECT_TRUE (in_dim[3] == res_dim[3]);

  /* output tensor in filter */
  status = ml_single_get_output_info (single_h, &out_res);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count (out_res, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (count, 1U);

  status = ml_tensors_info_get_tensor_type (out_res, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_FLOAT32);

  ml_tensors_info_get_tensor_dimension (out_res, 0, res_dim);
  EXPECT_TRUE (out_dim[0] == res_dim[0]);
  EXPECT_TRUE (out_dim[1] == res_dim[1]);
  EXPECT_TRUE (out_dim[2] == res_dim[2]);
  EXPECT_TRUE (out_dim[3] == res_dim[3]);

  /* generate data */
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status = ml_tensors_data_get_tensor_data (input, 0, (void **)&data, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (data_size, sizeof (float));
  *data = 10.0;

  status = ml_single_invoke (single_h, input, &output);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (output != NULL);

  status = ml_tensors_data_get_tensor_data (output, 0, (void **)&data, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (data_size, sizeof (float));
  EXPECT_FLOAT_EQ (*data, 12.0);
}


/**
 * @brief Test nnfw subplugin with unsuccessful invoke (single ML-API)
 * @detail Model is not found
 */
TEST_F (MLAPIInferenceNNFW, invoke_single_01_n)
{
  int status;
  g_autofree gchar *invalid_model = nullptr;

  invalid_model = g_build_filename (
      root_path, "tests", "test_models", "models", "invalid_model.tflite", NULL);
  EXPECT_FALSE (g_file_test (invalid_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  status = ml_single_open (
      &single_h, invalid_model, in_info, out_info, ML_NNFW_TYPE_NNFW, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* generate data */
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status = ml_single_invoke (single_h, input, &output);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test nnfw subplugin with unsuccessful invoke (single ML-API)
 * @detail Dimension of model is not matched.
 */
TEST_F (MLAPIInferenceNNFW, invoke_single_02_n)
{
  int status;
  unsigned int count = 0;
  float *data;
  size_t data_size;
  ml_tensor_type_e type = ML_TENSOR_TYPE_UNKNOWN;

  ASSERT_TRUE (valid_model != nullptr);

  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  /* Open model with proper dimension */
  status = ml_single_open (
      &single_h, valid_model, in_info, out_info, ML_NNFW_TYPE_NNFW, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* let's ignore timeout (30 sec) */
  status = ml_single_set_timeout (single_h, 30000);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* input tensor in filter */
  status = ml_single_get_input_info (single_h, &in_res);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count (in_res, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (count, 1U);

  status = ml_tensors_info_get_tensor_type (in_res, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_FLOAT32);

  ml_tensors_info_get_tensor_dimension (in_res, 0, res_dim);
  EXPECT_TRUE (in_dim[0] == res_dim[0]);
  EXPECT_TRUE (in_dim[1] == res_dim[1]);
  EXPECT_TRUE (in_dim[2] == res_dim[2]);
  EXPECT_TRUE (in_dim[3] == res_dim[3]);

  /* Change and update dimension for mismatch */
  in_dim[0] = in_dim[1] = in_dim[2] = in_dim[3] = 2;
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status = ml_tensors_data_get_tensor_data (input, 0, (void **)&data, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (data_size, sizeof (float) * 16);
  data[0] = 10.0;

  status = ml_single_invoke (single_h, input, &output);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test nnfw subplugin with successful invoke (pipeline, ML-API)
 */
TEST_F (MLAPIInferenceNNFW, invoke_pipeline_00)
{
  int status;
  float *data;
  size_t data_size;
  ml_pipeline_src_h src_handle;
  ml_pipeline_sink_h sink_handle;
  g_autofree gchar *pipeline = nullptr;
  ml_pipeline_state_e state;
  guint call_cnt = 0;

  ASSERT_TRUE (valid_model != nullptr);

  pipeline = g_strdup_printf ("appsrc name=appsrc ! "
                              "other/tensor,dimension=(string)1:1:1:1,type=(string)float32,framerate=(fraction)0/1 ! "
                              "tensor_filter framework=nnfw model=%s ! "
                              "tensor_sink name=tensor_sink", valid_model);

  status = ml_pipeline_construct (pipeline, NULL, NULL, &pipeline_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* get tensor element using name */
  status = ml_pipeline_src_get_handle (pipeline_h, "appsrc", &src_handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_register (
      pipeline_h, "tensor_sink", MLAPIInferenceNNFW::cb_new_data, &call_cnt, &sink_handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  status = ml_pipeline_start (pipeline_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_get_state (pipeline_h, &state);
  EXPECT_EQ (status, ML_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  /* generate data */
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status = ml_tensors_data_get_tensor_data (input, 0, (void **)&data, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (data_size, sizeof (float));
  *data = 10.0;
  status = ml_tensors_data_set_tensor_data (input, 0, data, sizeof (float));
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Push data to the source pad */
  for (int i = 0; i < 5; i++) {
    status = ml_pipeline_src_input_data (src_handle, input, ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
    EXPECT_EQ (status, ML_ERROR_NONE);
    g_usleep (100000);
  }

  MLAPIInferenceNNFW::wait_for_sink (&call_cnt, 5);

  status = ml_pipeline_stop (pipeline_h);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test nnfw subplugin with invalid model file (pipeline, ML-API)
 * @detail Failure case with invalid model file
 */
TEST_F (MLAPIInferenceNNFW, invoke_pipeline_01_n)
{
  int status;
  g_autofree gchar *pipeline = nullptr;
  g_autofree gcha