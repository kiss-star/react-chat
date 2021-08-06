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
#include <glib/gstdio.h> /* GStatBuf */
#include <nnstreamer.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_internal.h>
#include <nnstreamer-tizen-internal.h>
#include <ml-api-internal.h>
#include <ml-api-inference-internal.h>
#include <ml-api-inference-pipeline-internal.h>

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
 * @brief Struct to check the pipeline state changes.
 */
typedef struct {
  gboolean paused;
  gboolean playing;
} TestPipeState;

/**
 * @brief Macro to wait for pipeline state.
 */
#define wait_for_start(handle, state, status) do { \
    int counter = 0; \
    while ((state == ML_PIPELINE_STATE_PAUSED || state == ML_PIPELINE_STATE_READY) \
           && counter < 20) { \
      g_usleep (50000); \
      counter++; \
      status = ml_pipeline_get_state (handle, &state); \
      EXPECT_EQ (status, ML_ERROR_NONE); \
    } \
  } while (0)

/**
 * @brief Macro to wait for expected buffers to arrive.
 */
#define wait_pipeline_process_buffers(received, expected) do { \
    guint timer = 0; \
    while (received < expected) { \
      g_usleep (10000); \
      timer += 10; \
      if (timer > SINGLE_DEF_TIMEOUT_MSEC) \
        break; \
    } \
  } while (0)

#if defined (__TIZEN__)
#if TIZENPPM
/**
 * @brief Test NNStreamer pipeline construct with Tizen cam
 * @details Failure case to check permission (camera privilege)
 */
TEST (nnstreamer_capi_construct_destruct, tizen_cam_fail_01_n)
{
  ml_pipeline_h handle;
  gchar *pipeline;
  int status;

  pipeline = g_strdup_printf ("%s ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=320,height=240 ! tensor_converter ! tensor_sink",
      ML_TIZEN_CAM_VIDEO_SRC);

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_PERMISSION_DENIED);

  g_free (pipeline);
}

/**
 * @brief Test NNStreamer pipeline construct with Tizen cam
 * @details Failure case to check permission (camera privilege)
 */
TEST (nnstreamer_capi_construct_destruct, tizen_cam_fail_02_n)
{
  ml_pipeline_h handle;
  gchar *pipeline;
  int status;

  pipeline = g_strdup_printf ("%s ! audioconvert ! audio/x-raw,format=S16LE,rate=16000 ! tensor_converter ! tensor_sink",
      ML_TIZEN_CAM_AUDIO_SRC);

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_PERMISSION_DENIED);

  g_free (pipeline);
}
#endif /* TIZENPPM */

/**
 * @brief Test NNStreamer pipeline construct with Tizen internal API.
 */
TEST (nnstreamer_capi_construct_destruct, tizen_internal_01_p)
{
  ml_pipeline_h handle;
  gchar *pipeline;
  int status;

  pipeline = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=320,height=240 ! tensor_converter ! tensor_sink");

  status = ml_pipeline_construct_internal (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}

/**
 * @brief Test NNStreamer pipeline construct with Tizen internal API.
 */
TEST (nnstreamer_capi_construct_destruct, tizen_internal_02_p)
{
  ml_pipeline_h handle;
  gchar *pipeline;
  int status;

  pipeline = g_strdup_printf (
      "audiotestsrc ! audioconvert ! audio/x-raw,format=S16LE,rate=16000 ! tensor_converter ! tensor_sink");

  status = ml_pipeline_construct_internal (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}
#endif /* __TIZEN__ */

/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_construct_destruct, dummy_01)
{
  const char *pipeline = "videotestsrc num_buffers=2 ! fakesink";
  ml_pipeline_h handle;
  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_construct_destruct, dummy_02)
{
  const char *pipeline = "videotestsrc num_buffers=2 ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=224,height=224 ! tensor_converter ! fakesink";
  ml_pipeline_h handle;
  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_construct_destruct, dummy_03)
{
  const char *pipeline = "videotestsrc num_buffers=2 ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=224,height=224 ! tensor_converter ! valve name=valvex ! tensor_sink name=sinkx";
  ml_pipeline_h handle;
  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline construct with non-existent filter
 */
TEST (nnstreamer_capi_construct_destruct, failure_01_n)
{
  const char *pipeline = "nonexistsrc ! fakesink";
  ml_pipeline_h handle;
  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_STREAMS_PIPE);
}

/**
 * @brief Test NNStreamer pipeline construct with erroneous pipeline
 */
TEST (nnstreamer_capi_construct_destruct, failure_02_n)
{
  const char *pipeline = "videotestsrc num_buffers=2 ! audioconvert ! fakesink";
  ml_pipeline_h handle;
  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_STREAMS_PIPE);
}

/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_playstop, dummy_01)
{
  const char *pipeline = "videotestsrc is-live=true ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! valve name=valvex ! valve name=valvey ! input-selector name=is01 ! tensor_sink name=sinkx";
  ml_pipeline_h handle;
  ml_pipeline_state_e state;
  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status,
      ML_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  g_usleep (50000); /** 50ms is good for general systems, but not enough for
                       emulators to start gst pipeline. Let a few frames flow.
                       */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  wait_for_start (handle, state, status);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PLAYING);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (50000); /* 50ms. Let a few frames flow. */

  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PAUSED);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_playstop, dummy_02)
{
  const char *pipeline = "videotestsrc is-live=true ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! valve name=valvex ! valve name=valvey ! input-selector name=is01 ! tensor_sink name=sinkx";
  ml_pipeline_h handle;
  ml_pipeline_state_e state;
  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status,
      ML_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  g_usleep (50000); /** 50ms is good for general systems, but not enough for
                       emulators to start gst pipeline. Let a few frames flow.
                       */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  wait_for_start (handle, state, status);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PLAYING);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (50000); /* 50ms. Let a few frames flow. */

  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PAUSED);

  /* Resume playing */
  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  g_usleep (50000); /* 50ms. Enough to empty the queue */
  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PAUSED);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_valve, test01)
{
  const gchar *_tmpdir = g_get_tmp_dir ();
  const gchar *_dirname = "nns-tizen-XXXXXX";
  gchar *fullpath = g_build_path ("/", _tmpdir, _dirname, NULL);
  gchar *dir = g_mkdtemp ((gchar *)fullpath);
  gchar *file1 = g_build_path ("/", dir, "valve1", NULL);
  gchar *pipeline = g_strdup_printf (
      "videotestsrc is-live=true ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=16,height=16,framerate=10/1 ! tensor_converter ! queue ! valve name=valve1 ! filesink location=\"%s\"",
      file1);
  GStatBuf buf;

  ml_pipeline_h handle;
  ml_pipeline_state_e state;
  ml_pipeline_valve_h valve1;

  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_TRUE (dir != NULL);

  status = ml_pipeline_valve_get_handle (handle, "valve1", &valve1);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_valve_set_open (valve1, false); /* close */
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (50000); /* 50ms. Wait for the pipeline stgart. */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status,
      ML_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  wait_for_start (handle, state, status);
  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = g_lstat (file1, &buf);
  EXPECT_EQ (status, 0);
  EXPECT_EQ (buf.st_size, 0);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_valve_set_open (valve1, true); /* open */
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_valve_release_handle (valve1); /* release valve handle */
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (500000); /* 500ms. Let a few frames flow. (10Hz x 0.5s --> 5)*/

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = g_lstat (file1, &buf);
  EXPECT_EQ (status, 0);
  EXPECT_GE (buf.st_size, 2048); /* At least two frames during 500ms */
  EXPECT_LE (buf.st_size, 6144); /* At most six frames during 500ms */
  EXPECT_EQ (buf.st_size % 1024, 0); /* It should be divided by 1024 */

  g_free (fullpath);
  g_free (file1);
  g_free (pipeline);
}

/**
 * @brief Test NNStreamer pipeline valve
 * @detail Failure case to handle valve element with invalid param.
 */
TEST (nnstreamer_capi_valve, failure_01_n)
{
  ml_pipeline_valve_h valve_h;
  int status;

  /* invalid param : pipe */
  status = ml_pipeline_valve_get_handle (NULL, "valvex", &valve_h);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test NNStreamer pipeline valve
 * @detail Failure case to handle valve element with invalid param.
 */
TEST (nnstreamer_capi_valve, failure_02_n)
{
  ml_pipeline_h handle;
  ml_pipeline_valve_h valve_h;
  gchar *pipeline;
  int status;

  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! valve name=valvex ! tensor_converter ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : name */
  status = ml_pipeline_valve_get_handle (handle, NULL, &valve_h);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}
/**
 * @brief Test NNStreamer pipeline valve
 * @detail Failure case to handle valve element with invalid param.
 */
TEST (nnstreamer_capi_valve, failure_03_n)
{
  ml_pipeline_h handle;
  ml_pipeline_valve_h valve_h;
  gchar *pipeline;
  int status;

  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! valve name=valvex ! tensor_converter ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : wrong name */
  status = ml_pipeline_valve_get_handle (handle, "wrongname", &valve_h);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}

/**
 * @brief Test NNStreamer pipeline valve
 * @detail Failure case to handle valve element with invalid param.
 */
TEST (nnstreamer_capi_valve, failure_04_n)
{
  ml_pipeline_h handle;
  ml_pipeline_valve_h valve_h;
  gchar *pipeline;
  int status;

  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! valve name=valvex ! tensor_converter ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : invalid type */
  status = ml_pipeline_valve_get_handle (handle, "sinkx", &valve_h);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}

/**
 * @brief Test NNStreamer pipeline valve
 * @detail Failure case to handle valve element with invalid param.
 */
TEST (nnstreamer_capi_valve, failure_05_n)
{
  ml_pipeline_h handle;
  gchar *pipeline;
  int status;

  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! valve name=valvex ! tensor_converter ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : handle */
  status = ml_pipeline_valve_get_handle (handle, "valvex", NULL);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}

G_LOCK_DEFINE_STATIC (callback_lock);
/**
 * @brief A tensor-sink callback for sink handle in a pipeline
 */
static void
test_sink_callback_dm01 (
    const ml_tensors_data_h data, const ml_tensors_info_h info, void *user_data)
{
  gchar *filepath = (gchar *)user_data;
  unsigned int i, num = 0;
  void *data_ptr;
  size_t data_size;
  int status;
  FILE *fp = g_fopen (filepath, "a");

  if (fp == NULL)
    return;

  G_LOCK (callback_lock);
  ml_tensors_info_get_count (info, &num);

  for (i = 0; i < num; i++) {
    status = ml_tensors_data_get_tensor_data (data, i, &data_ptr, &data_size);
    if (status == ML_ERROR_NONE)
      fwrite (data_ptr, data_size, 1, fp);
  }
  G_UNLOCK (callback_lock);

  fclose (fp);
}

/**
 * @brief A tensor-sink callback for sink handle in a pipeline
 */
static void
test_sink_callback_count (
    const ml_tensors_data_h data, const ml_tensors_info_h info, void *user_data)
{
  guint *count = (guint *)user_data;

  G_LOCK (callback_lock);
  *count = *count + 1;
  G_UNLOCK (callback_lock);
}

/**
 * @brief Pipeline state changed callback
 */
static void
test_pipe_state_callback (ml_pipeline_state_e state, void *user_data)
{
  TestPipeState *pipe_state;

  G_LOCK (callback_lock);
  pipe_state = (TestPipeState *)user_data;

  switch (state) {
  case ML_PIPELINE_STATE_PAUSED:
    pipe_state->paused = TRUE;
    break;
  case ML_PIPELINE_STATE_PLAYING:
    pipe_state->playing = TRUE;
    break;
  default:
    break;
  }
  G_UNLOCK (callback_lock);
}

/**
 * @brief compare the two files.
 */
static int
file_cmp (const gchar *f1, const gchar *f2)
{
  gboolean r;
  gchar *content1 = NULL;
  gchar *content2 = NULL;
  gsize len1, len2;
  int cmp = 0;

  r = g_file_get_contents (f1, &content1, &len1, NULL);
  if (r != TRUE)
    return -1;

  r = g_file_get_contents (f2, &content2, &len2, NULL);
  if (r != TRUE) {
    g_free (content1);
    return -2;
  }

  if (len1 == len2) {
    cmp = memcmp (content1, content2, len1);
  } else {
    cmp = 1;
  }

  g_free (content1);
  g_free (content2);

  return cmp;
}

/**
 * @brief Wait until the change in pipeline status is done
 * @return ML_ERROR_NONE success, ML_ERROR_UNKNOWN if failed, ML_ERROR_TIMED_OUT if timeout happens.
 */
static int
waitPipelineStateChange (ml_pipeline_h handle, ml_pipeline_state_e state, guint timeout_ms)
{
  int status = ML_ERROR_UNKNOWN;
  guint counter = 0;
  ml_pipeline_state_e cur_state = ML_PIPELINE_STATE_NULL;

  do {
    status = ml_pipeline_get_state (handle, &cur_state);
    EXPECT_EQ (status, ML_ERROR_NONE);
    if (cur_state == ML_PIPELINE_STATE_UNKNOWN)
      return ML_ERROR_UNKNOWN;
    if (cur_state == state)
      return ML_ERROR_NONE;
    g_usleep (10000);
  } while ((timeout_ms / 10) >= counter++);

  return ML_ERROR_TIMED_OUT;
}

/**
 * @brief Test NNStreamer pipeline sink
 */
TEST (nnstreamer_capi_sink, dummy_01)
{
  const gchar *_tmpdir = g_get_tmp_dir ();
  const gchar *_dirname = "nns-tizen-XXXXXX";
  gchar *fullpath = g_build_path ("/", _tmpdir, _dirname, NULL);
  gchar *dir = g_mkdtemp ((gchar *)fullpath);

  ASSERT_NE (dir, (gchar *)NULL);

  gchar *file1 = g_build_path ("/", dir, "original", NULL);
  gchar *file2 = g_build_path ("/", dir, "sink", NULL);
  gchar *pipeline = g_strdup_printf (
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=BGRx,width=64,height=48,famerate=30/1 ! tee name=t t. ! queue ! filesink location=\"%s\" buffer-mode=unbuffered t. ! queue ! tensor_converter ! tensor_sink name=sinkx",
      file1);
  ml_pipeline_h handle;
  ml_pipeline_sink_h sinkhandle;
  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_register (
      handle, "sinkx", test_sink_callback_dm01, file2, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (sinkhandle != NULL);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = waitPipelineStateChange (handle, ML_PIPELINE_STATE_PLAYING, 200);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* 200ms. Give enough time for three frames to flow. */
  g_usleep (200000);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (10000); /* 10ms. Wait a bit. */

  status = ml_pipeline_sink_unregister (sinkhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);

  /* File Comparison to check the integrity */
  EXPECT_EQ (file_cmp (file1, file2), 0);

  g_free (fullpath);
  g_free (file1);
  g_free (file2);
}

/**
 * @brief Test NNStreamer pipeline sink
 */
TEST (nnstreamer_capi_sink, dummy_02)
{
  ml_pipeline_h handle;
  ml_pipeline_state_e state;
  ml_pipeline_sink_h sinkhandle;
  gchar *pipeline;
  int status;
  guint *count_sink;
  TestPipeState *pipe_state;

  /* pipeline with appsink */
  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! tensor_converter ! appsink name=sinkx sync=false");

  count_sink = (guint *)g_malloc (sizeof (guint));
  ASSERT_TRUE (count_sink != NULL);
  *count_sink = 0;

  pipe_state = (TestPipeState *)g_new0 (TestPipeState, 1);
  ASSERT_TRUE (pipe_state != NULL);

  status = ml_pipeline_construct (pipeline, test_pipe_state_callback, pipe_state, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_register (
      handle, "sinkx", test_sink_callback_count, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (sinkhandle != NULL);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (100000); /* 100ms. Let a few frames flow. */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PLAYING);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (10000); /* 10ms. Wait a bit. */

  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PAUSED);

  status = ml_pipeline_sink_unregister (sinkhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_TRUE (*count_sink > 0U);
  EXPECT_TRUE (pipe_state->paused);
  EXPECT_TRUE (pipe_state->playing);

  g_free (pipeline);
  g_free (count_sink);
  g_free (pipe_state);
}

/**
 * @brief Test NNStreamer pipeline sink
 */
TEST (nnstreamer_capi_sink, register_duplicated)
{
  ml_pipeline_h handle;
  ml_pipeline_sink_h sinkhandle0, sinkhandle1;
  gchar *pipeline;
  int status;
  guint *count_sink0, *count_sink1;
  TestPipeState *pipe_state;

  /* pipeline with appsink */
  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! tensor_converter ! appsink name=sinkx sync=false");
  count_sink0 = (guint *)g_malloc (sizeof (guint));
  ASSERT_TRUE (count_sink0 != NULL);
  *count_sink0 = 0;

  count_sink1 = (guint *)g_malloc (sizeof (guint));
  ASSERT_TRUE (count_sink1 != NULL);
  *count_sink1 = 0;

  pipe_state = (TestPipeState *)g_new0 (TestPipeState, 1);
  ASSERT_TRUE (pipe_state != NULL);

  status = ml_pipeline_construct (pipeline, test_pipe_state_callback, pipe_state, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_register (
      handle, "sinkx", test_sink_callback_count, count_sink0, &sinkhandle0);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (sinkhandle0 != NULL);

  status = ml_pipeline_sink_register (
      handle, "sinkx", test_sink_callback_count, count_sink1, &sinkhandle1);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (sinkhandle1 != NULL);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (100000); /* 100ms. Let a few frames flow. */

  status = ml_pipeline_sink_unregister (sinkhandle0);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_unregister (sinkhandle1);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_TRUE (*count_sink0 > 0U);
  EXPECT_TRUE (*count_sink1 > 0U);
  EXPECT_TRUE (pipe_state->paused);
  EXPECT_TRUE (pipe_state->playing);

  g_free (pipeline);
  g_free (count_sink0);
  g_free (count_sink1);
  g_free (pipe_state);
}

/**
 * @brief Test NNStreamer pipeline sink
 * @detail Failure case to register callback with invalid param.
 */
TEST (nnstreamer_capi_sink, failure_01_n)
{
  ml_pipeline_sink_h sinkhandle;
  int status;
  guint *count_sink;

  count_sink = (guint *)g_malloc (sizeof (guint));
  ASSERT_TRUE (count_sink != NULL);
  *count_sink = 0;

  /* invalid param : pipe */
  status = ml_pipeline_sink_register (
      NULL, "sinkx", test_sink_callback_count, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  g_free (count_sink);
}

/**
 * @brief Test NNStreamer pipeline sink
 * @detail Failure case to register callback with invalid param.
 */
TEST (nnstreamer_capi_sink, failure_02_n)
{
  ml_pipeline_h handle;
  ml_pipeline_sink_h sinkhandle;
  gchar *pipeline;
  int status;
  guint *count_sink;

  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! valve name=valvex ! tensor_converter ! tensor_sink name=sinkx");

  count_sink = (guint *)g_malloc (sizeof (guint));
  ASSERT_TRUE (count_sink != NULL);
  *count_sink = 0;

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : name */
  status = ml_pipeline_sink_register (
      handle, NULL, test_sink_callback_count, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  g_free (pipeline);
  g_free (count_sink);
}

/**
 * @brief Test NNStreamer pipeline sink
 * @detail Failure case to register callback with invalid param.
 */
TEST (nnstreamer_capi_sink, failure_03_n)
{
  ml_pipeline_h handle;
  ml_pipeline_sink_h sinkhandle;
  gchar *pipeline;
  int status;
  guint *count_sink;

  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! valve name=valvex ! tensor_converter ! tensor_sink name=sinkx");

  count_sink = (guint *)g_malloc (sizeof (guint));
  ASSERT_TRUE (count_sink != NULL);
  *count_sink = 0;

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : wrong name */
  status = ml_pipeline_sink_register (
      handle, "wrongname", test_sink_callback_count, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  g_free (pipeline);
  g_free (count_sink);
}

/**
 * @brief Test NNStreamer pipeline sink
 * @detail Failure case to register callback with invalid param.
 */
TEST (nnstreamer_capi_sink, failure_04_n)
{
  ml_pipeline_h handle;
  ml_pipeline_sink_h sinkhandle;
  gchar *pipeline;
  int status;
  guint *count_sink;

  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! valve name=valvex ! tensor_converter ! tensor_sink name=sinkx");

  count_sink = (guint *)g_malloc (sizeof (guint));
  ASSERT_TRUE (count_sink != NULL);
  *count_sink = 0;

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : invalid type */
  status = ml_pipeline_sink_register (
      handle, "valvex", test_sink_callback_count, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  g_free (pipeline);
  g_free (count_sink);
}

/**
 * @brief Test NNStreamer pipeline sink
 * @detail Failure case to register callback with invalid param.
 */
TEST (nnstreamer_capi_sink, failure_05_n)
{
  ml_pipeline_h handle;
  ml_pipeline_sink_h sinkhandle;
  gchar *pipeline;
  int status;
  guint *count_sink;

  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! valve name=valvex ! tensor_converter ! tensor_sink name=sinkx");

  count_sink = (guint *)g_malloc (sizeof (guint));
  ASSERT_TRUE (count_sink != NULL);
  *count_sink = 0;

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : callback */
  status = ml_pipeline_sink_register (handle, "sinkx", NULL, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  g_free (pipeline);
  g_free (count_sink);
}

/**
 * @brief Test NNStreamer pipeline sink
 * @detail Failure case to register callback with invalid param.
 */
TEST (nnstreamer_capi_sink, failure_06_n)
{
  ml_pipeline_h handle;
  gchar *pipeline;
  int status;
  guint *count_sink;

  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! valve name=valvex ! tensor_converter ! tensor_sink name=sinkx");

  count_sink = (guint *)g_malloc (sizeof (guint));
  ASSERT_TRUE (count_sink != NULL);
  *count_sink = 0;

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : handle */
  status = ml_pipeline_sink_register (
      handle, "sinkx", test_sink_callback_count, count_sink, NULL);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  g_free (pipeline);
  g_free (count_sink);
}

/**
 * @brief Test NNStreamer pipeline src
 */
TEST (nnstreamer_capi_src, dummy_01)
{
  const gchar *_tmpdir = g_get_tmp_dir ();
  const gchar *_dirname = "nns-tizen-XXXXXX";
  gchar *fullpath = g_build_path ("/", _tmpdir, _dirname, NULL);
  gchar *dir = g_mkdtemp ((gchar *)fullpath);
  gchar *file1 = g_build_path ("/", dir, "output", NULL);
  gchar *pipeline = g_strdup_printf (
      "appsrc name=srcx ! other/tensor,dimension=(string)4:1:1:1,type=(string)uint8,framerate=(fraction)0/1 ! filesink location=\"%s\" buffer-mode=unbuffered",
      file1);
  ml_pipeline_h handle;
  ml_pipeline_state_e state;
  ml_pipeline_src_h srchandle;
  int status;
  ml_tensors_info_h info;
  ml_tensors_data_h data1, data2;
  unsigned int count = 0;
  ml_tensor_type_e type = ML_TENSOR_TYPE_UNKNOWN;
  ml_tensor_dimension dim = {
    0,
  };

  int i;
  uint8_t *uintarray1[10];
  uint8_t *uintarray2[10];
  uint8_t *content = NULL;
  gsize len;

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (dir != NULL);
  for (i = 0; i < 10; i++) {
    uintarray1[i] = (uint8_t *)g_malloc (4);
    ASSERT_TRUE (uintarray1[i] != NULL);
    uintarray1[i][0] = i + 4;
    uintarray1[i][1] = i + 1;
    uintarray1[i][2] = i + 3;
    uintarray1[i][3] = i + 2;

    uintarray2[i] = (uint8_t *)g_malloc (4);
    ASSERT_TRUE (uintarray2[i] != NULL);
    uintarray2[i][0] = i + 3;
    uintarray2[i][1] = i + 2;
    uintarray2[i][2] = i + 1;
    uintarray2[i][3] = i + 4;
    /* These will be free'ed by gstreamer (ML_PIPELINE_BUF_POLICY_AUTO_FREE) */
    /** @todo Check whether gstreamer really deallocates this */
  }

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (10000); /* 10ms. Wait a bit. */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status,
      ML_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  status = ml_pipeline_src_get_handle (handle, "srcx", &srchandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_src_get_tensors_info (srchandle, &info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_info_get_count (info, &count);
  EXPECT_EQ (count, 1U);

  ml_tensors_info_get_tensor_type (info, 0, &type);
  EXPECT_EQ (type, ML_TENSOR_TYPE_UINT8);

  ml_tensors_info_get_tensor_dimension (info, 0, dim);
  EXPECT_EQ (dim[0], 4U);
  EXPECT_EQ (dim[1], 1U);
  EXPECT_EQ (dim[2], 1U);
  EXPECT_EQ (dim[3], 1U);

  status = ml_tensors_data_create (info, &data1);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_info_destroy (info);

  status = ml_tensors_data_set_tensor_data (data1, 0, uintarray1[0], 4);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_src_input_data (srchandle, data1, ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (50000); /* 50ms. Wait a bit. */

  status = ml_pipeline_src_input_data (srchandle, data1, ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (50000); /* 50ms. Wait a bit. */

  status = ml_pipeline_src_release_handle (srchandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_src_get_handle (handle, "srcx", &srchandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_src_get_tensors_info (srchandle, &info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_info_get_count (info, &count);
  EXPECT_EQ (count, 1U);

  ml_tensors_info_get_tensor_type (info, 0, &type);
  EXPECT_EQ (type, ML_TENSOR_TYPE_UINT8);

  ml_tensors_info_get_tensor_dimension (info, 0, dim);
  EXPECT_EQ (dim[0], 4U);
  EXPECT_EQ (dim[1], 1U);
  EXPECT_EQ (dim[2], 1U);
  EXPECT_EQ (dim[3], 1U);

  for (i = 0; i < 10; i++) {
    status = ml_tensors_data_set_tensor_data (data1, 0, uintarray1[i], 4);
    EXPECT_EQ (status, ML_ERROR_NONE);

    status = ml_pipeline_src_input_data (srchandle, data1, ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
    EXPECT_EQ (status, ML_ERROR_NONE);

    status = ml_tensors_data_create (info, &data2);
    EXPECT_EQ (status, ML_ERROR_NONE);

    status = ml_tensors_data_set_tensor_data (data2, 0, uintarray2[i], 4);
    EXPECT_EQ (status, ML_ERROR_NONE);

    status = ml_pipeline_src_input_data (srchandle, data2, ML_PIPELINE_BUF_POLICY_AUTO_FREE);
    EXPECT_EQ (status, ML_ERROR_NONE);

    g_usleep (50000); /* 50ms. Wait a bit. */
  }

  status = ml_pipeline_src_release_handle (srchandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (50000); /* Wait for the pipeline to flush all */

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);

  EXPECT_TRUE (g_file_get_contents (file1, (gchar **)&content, &len, NULL));
  EXPECT_EQ (len, 8U * 11);
  EXPECT_TRUE (content != nullptr);

  if (content && len == 88U) {
    for (i = 0; i < 10; i++) {
      EXPECT_EQ (content[i * 8 + 0 + 8], i + 4);
      EXPECT_EQ (content[i * 8 + 1 + 8], i + 1);
      EXPECT_EQ (content[i * 8 + 2 + 8], i + 3);
      EXPECT_EQ (content[i * 8 + 3 + 8], i + 2);
      EXPECT_EQ (content[i * 8 + 4 + 8], i + 3);
      EXPECT_EQ (content[i * 8 + 5 + 8], i + 2);
      EXPECT_EQ (content[i * 8 + 6 + 8], i + 1);
      EXPECT_EQ (content[i * 8 + 7 + 8], i + 4);
    }
  }

  g_free (content);
  ml_tensors_info_destroy (info);
  ml_tensors_data_destroy (data1);

  for (i = 0; i < 10; i++) {
    g_free (uintarray1[i]);
    g_free (uintarray2[i]);
  }

  g_free (fullpath);
  g_free (file1);
}

/**
 * @brief Test NNStreamer pipeline src
 * @detail Failure case when pipeline is NULL.
 */
TEST (nnstreamer_capi_src, failure_01_n)
{
  int status;
  ml_pipeline_src_h srchandle;

  status = ml_pipeline_src_get_handle (NULL, "dummy", &srchandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test NNStreamer pipeline src
 * @detail Failure case when the name of source node is wr