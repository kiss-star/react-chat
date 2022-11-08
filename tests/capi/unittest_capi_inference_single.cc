
/**
 * @file        unittest_capi_inference.cc
 * @date        13 Mar 2019
 * @brief       Unit test for ML inference C-API.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <nnstreamer.h>
#include <nnstreamer-single.h>
#include <nnstreamer_internal.h>
#include <nnstreamer-tizen-internal.h>
#include <ml-api-inference-internal.h>
#include <ml-api-inference-single-internal.h>

#if defined (__APPLE__)
#define SO_FILE_EXTENSION ".dylib"
#else
#define SO_FILE_EXTENSION ".so"
#endif

static const unsigned int SINGLE_DEF_TIMEOUT_MSEC = 10000U;

#if defined (ENABLE_TENSORFLOW_LITE) || defined (ENABLE_TENSORFLOW2_LITE)
constexpr bool is_enabled_tensorflow_lite = true;
#else
constexpr bool is_enabled_tensorflow_lite = false;
#endif

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 */
TEST (nnstreamer_capi_singleshot, invoke_invalid_param_01_n)
{
  ml_single_h single;
  int status;
  ml_tensors_info_h in_info;
  ml_tensors_data_h input, output;

  const gchar *root_path = g_getenv ("MLAPI_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    EXPECT_EQ (status, ML_ERROR_NONE);
  } else {
    EXPECT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

  status = ml_single_get_input_info (single, &in_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_invoke (NULL, input, &output);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_single_invoke (single, NULL, &output);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_single_invoke (single, input, NULL);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_data_destroy (input);
  ml_tensors_info_destroy (in_info);

skip_test:
  g_free (test_model);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 */
TEST (nnstreamer_capi_singleshot, invoke_invalid_param_02_n)
{
  ml_single_h single;
  int status;
  ml_tensors_info_h in_info;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim;

  const gchar *root_path = g_getenv ("MLAPI_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    EXPECT_EQ (status, ML_ERROR_NONE);
  } else {
    EXPECT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

  ml_tensors_info_create (&in_info);

  in_dim[0] = 3;
  in_dim[1] = 224;
  in_dim[2] = 224;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  /* handle null data */
  status = _ml_tensors_data_create_no_alloc (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_invoke (single, input, &output);
  EXPECT_NE (status, ML_ERROR_NONE);

  ml_tensors_data_destroy (input);

  /* set invalid type to test wrong data size */
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_UINT32);

  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_invoke (single, input, &output);
  EXPECT_NE (status, ML_ERROR_NONE);

  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_data_destroy (input);

  /* set invalid input tensor number */
  ml_tensors_info_set_count (in_info, 2);
  ml_tensors_info_set_tensor_type (in_info, 1, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (in_info, 1, in_dim);

  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_invoke (single, input, &output);
  EXPECT_NE (status, ML_ERROR_NONE);

  ml_tensors_data_destroy (input);
  ml_tensors_info_destroy (in_info);

skip_test:
  g_free (test_model);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 */
TEST (nnstreamer_capi_singleshot, invoke_01)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_info_h in_res, out_res;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim, out_dim, res_dim;
  ml_tensor_type_e type = ML_TENSOR_TYPE_UNKNOWN;
  unsigned int count = 0;
  char *name = NULL;
  int status;

  const gchar *root_path = g_getenv ("MLAPI_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  in_dim[0] = 3;
  in_dim[1] = 224;
  in_dim[2] = 224;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  out_dim[0] = 1001;
  out_dim[1] = 1;
  out_dim[2] = 1;
  out_dim[3] = 1;
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    EXPECT_EQ (status, ML_ERROR_NONE);
  } else {
    EXPECT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

  /* input tensor in filter */
  status = ml_single_get_input_info (single, &in_res);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count (in_res, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (count, 1U);

  status = ml_tensors_info_get_tensor_name (in_res, 0, &name);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_free (name);

  status = ml_tensors_info_get_tensor_type (in_res, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_UINT8);

  ml_tensors_info_get_tensor_dimension (in_res, 0, res_dim);
  EXPECT_TRUE (in_dim[0] == res_dim[0]);
  EXPECT_TRUE (in_dim[1] == res_dim[1]);
  EXPECT_TRUE (in_dim[2] == res_dim[2]);
  EXPECT_TRUE (in_dim[3] == res_dim[3]);

  /* output tensor in filter */
  status = ml_single_get_output_info (single, &out_res);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count (out_res, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (count, 1U);

  status = ml_tensors_info_get_tensor_name (out_res, 0, &name);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_free (name);

  status = ml_tensors_info_get_tensor_type (out_res, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_UINT8);

  ml_tensors_info_get_tensor_dimension (out_res, 0, res_dim);
  EXPECT_TRUE (out_dim[0] == res_dim[0]);
  EXPECT_TRUE (out_dim[1] == res_dim[1]);
  EXPECT_TRUE (out_dim[2] == res_dim[2]);
  EXPECT_TRUE (out_dim[3] == res_dim[3]);

  input = output = NULL;

  /* generate dummy data */
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status = ml_single_set_timeout (single, SINGLE_DEF_TIMEOUT_MSEC);
  EXPECT_TRUE (status == ML_ERROR_NOT_SUPPORTED || status == ML_ERROR_NONE);

  status = ml_single_invoke (single, input, &output);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (output != NULL);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_data_destroy (output);
  ml_tensors_data_destroy (input);
  ml_tensors_info_destroy (in_res);
  ml_tensors_info_destroy (out_res);

skip_test:
  g_free (test_model);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Start pipeline without tensor info
 */
TEST (nnstreamer_capi_singleshot, invoke_02)
{
  ml_single_h single;
  ml_tensors_info_h in_info;
  ml_tensors_data_h input, output;
  int status;

  const gchar *root_path = g_getenv ("MLAPI_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    EXPECT_EQ (status, ML_ERROR_NONE);
  } else {
    EXPECT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

  status = ml_single_get_input_info (single, &in_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  input = output = NULL;

  /* generate dummy data */
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status = ml_single_set_timeout (single, SINGLE_DEF_TIMEOUT_MSEC);
  EXPECT_TRUE (status == ML_ERROR_NOT_SUPPORTED || status == ML_ERROR_NONE);

  status = ml_single_invoke (single, input, &output);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (output != NULL);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_data_destroy (output);
  ml_tensors_data_destroy (input);
  ml_tensors_info_destroy (in_info);
skip_test:
  g_free (test_model);
}

/**
 * @brief Measure the loading time and total time for the run
 */
static void
benchmark_single (const gboolean no_alloc, const gboolean no_timeout, const int count)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim, out_dim;
  int status;
  unsigned long open_duration = 0, invoke_duration = 0, close_duration = 0;
  gint64 start, end;

  const gchar *root_path = g_getenv ("MLAPI_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  in_dim[0] = 3;
  in_dim[1] = 224;
  in_dim[2] = 224;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  out_dim[0] = 1001;
  out_dim[1] = 1;
  out_dim[2] = 1;
  out_dim[3] = 1;
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  /** Initial run to warm up the cache */
  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    EXPECT_EQ (status, ML_ERROR_NONE);
  } else {
    EXPECT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }
  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  for (int i = 0; i < count; i++) {
    start = g_get_monotonic_time ();
    status = ml_single_open (&single, test_model, in_info, out_info,
        ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
    end = g_get_monotonic_time ();
    open_duration += end - start;
    ASSERT_EQ (status, ML_ERROR_NONE);

    if (!no_timeout) {
      status = ml_single_set_timeout (single, SINGLE_DEF_TIMEOUT_MSEC);
      EXPECT_TRUE (status == ML_ERROR_NOT_SUPPORTED || status == ML_ERROR_NONE);
    }

    /* generate dummy data */
    input = output = NULL;

    status = ml_tensors_data_create (in_info, &input);
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_TRUE (input != NULL);

    if (no_alloc) {
      status = ml_tensors_data_create (out_info, &output);
      EXPECT_EQ (status, ML_ERROR_NONE);
      EXPECT_TRUE (output != NULL);
    }

    start = g_get_monotonic_time ();
    if (no_alloc)
      status = ml_single_invoke_fast (single, input, output);
    else
      status = ml_single_invoke (single, input, &output);
    end = g_get_monotonic_time ();
    invoke_duration += end - start;
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_TRUE (output != NULL);

    start = g_get_monotonic_time ();
    status = ml_single_close (single);
    end = g_get_monotonic_time ();
    close_duration = end - start;
    EXPECT_EQ (status, ML_ERROR_NONE);

    ml_tensors_data_destroy (input);
    ml_tensors_data_destroy (output);
  }

  g_warning ("Time to open single = %f us", (open_duration * 1.0) / count);
  g_warning ("Time to invoke single = %f us", (invoke_duration * 1.0) / count);
  g_warning ("Time to close single = %f us", (close_duration * 1.0) / count);

skip_test:
  g_free (test_model);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @note Measure the loading time and total time for the run
 */
TEST (nnstreamer_capi_singleshot, benchmark_time)
{
  g_warning ("Benchmark (no timeout)");
  benchmark_single (FALSE, TRUE, 1);

  g_warning ("Benchmark (no alloc, no timeout)");
  benchmark_single (TRUE, TRUE, 1);
}

/**
 * @brief Test NNStreamer single shot (custom filter)
 * @detail Run pipeline with custom filter, handle multi tensors.
 */
TEST (nnstreamer_capi_singleshot, invoke_03)
{
  const gchar cf_name[] = "libnnstreamer_customfilter_passthrough_variable" SO_FILE_EXTENSION;
  gchar *lib_path = NULL;
  gchar *test_model = NULL;
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim;
  int status;
  unsigned int i;
  void *data_ptr;
  size_t data_size;

  lib_path = nnsconf_get_custom_value_string ("filter", "customfilters");
  if (lib_path == NULL) {
    /* cannot get custom-filter directory */
    goto skip_test;
  }

  test_model = g_build_filename (lib_path, cf_name, NULL);
  if (!g_file_test (test_model, G_FILE_TEST_EXISTS)) {
    goto skip_test;
  }

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  ml_tensors_info_set_count (in_info, 2);

  in_dim[0] = 10;
  in_dim[1] = 1;
  in_dim[2] = 1;
  in_dim[3] = 1;

  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_INT16);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  ml_tensors_info_set_tensor_type (in_info, 1, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (in_info, 1, in_dim);

  ml_tensors_info_clone (out_info, in_info);

  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_CUSTOM_FILTER, ML_NNFW_HW_ANY);
  ASSERT_EQ (status, ML_ERROR_NONE);

  input = output = NULL;

  /* generate input data */
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  ASSERT_TRUE (input != NULL);

  for (i = 0; i < 10; i++) {
    int16_t i16 = (int16_t) (i + 1);
    float f32 = (float)(i + .1);

    status = ml_tensors_data_get_tensor_data (input, 0, &data_ptr, &data_size);
    EXPECT_EQ (status, ML_ERROR_NONE);
    ((int16_t *)data_ptr)[i] = i16;