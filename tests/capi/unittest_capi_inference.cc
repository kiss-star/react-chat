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
    in