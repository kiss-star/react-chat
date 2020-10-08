/* SPDX-License-Identifier: Apache-2.0 */
/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file ml-api-inference-single.c
 * @date 29 Aug 2019
 * @brief NNStreamer/Single C-API Wrapper.
 *        This allows to invoke individual input frame with NNStreamer.
 * @see	https://github.com/nnstreamer/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <string.h>
#include <nnstreamer-single.h>
#include <nnstreamer-tizen-internal.h>  /* Tizen platform header */
#include <nnstreamer_internal.h>
#include <nnstreamer_plugin_api_util.h>
#include <tensor_filter_single.h>

#include "ml-api-inference-internal.h"
#include "ml-api-internal.h"
#include "ml-api-inference-single-internal.h"

#define ML_SINGLE_MAGIC 0xfeedfeed

/**
 * @brief Default time to wait for an output in milliseconds (0 will wait for the output).
 */
#define SINGLE_DEFAULT_TIMEOUT 0

/**
 * @brief Global lock for single shot API
 * @detail This lock ensures that ml_single_close is thread safe. All other API
 *         functions use the mutex from the single handle. However for close,
 *         single handle mutex cannot be used as single handle is destroyed at
 *         close
 * @note This mutex is automatically initialized as it is statically declared
 */
G_LOCK_DEFINE_STATIC (magic);

/**
 * @brief Get valid handle after magic verification
 * @note handle's mutex (single_h->mutex) is acquired after this
 * @param[out] single_h The handle properly casted: (ml_single *).
 * @param[in] single The handle to be validated: (void *).
 * @param[in] reset Set TRUE if the handle is to be reset (magic = 0).
 */
#define ML_SINGLE_GET_VALID_HANDLE_LOCKED(single_h, single, reset) do { \
  G_LOCK (magic); \
  single_h = (ml_single *) single; \
  if (G_UNLIKELY(single_h->magic != ML_SINGLE_MAGIC)) { \
    _ml_error_report \
        ("The given param, %s (ml_single_h), is invalid. It is not a single_h instance or the user thread has modified it.", \
        #single); \
    G_UNLOCK (magic); \
    return ML_ERROR_INVALID_PARAMETER; \
  } \
  if (G_UNLIKELY(reset)) \
    single_h->magic = 0; \
  g_mutex_lock (&single_h->mutex); \
  G_UNLOCK (magic); \
} while (0)

/**
 * @brief This is for the symmetricity with ML_SINGLE_GET_VALID_HANDLE_LOCKED
 * @param[in] single_h The casted handle (ml_single *).
 */
#define ML_SINGLE_HANDLE_UNLOCK(single_h) g_mutex_unlock (&single_h->mutex);

/** define string names for input/output */
#define INPUT_STR "input"
#define OUTPUT_STR "output"
#define TYPE_STR "type"
#define NAME_STR "name"

/** concat string from #define */
#define CONCAT_MACRO_STR(STR1,STR2) STR1 STR2

/** States for invoke thread */
typedef enum
{
  IDLE = 0,           /**< ready to accept next input */
  RUNNING,            /**< running an input, cannot accept more input */
  JOIN_REQUESTED      /**< should join the thread, will exit soon */
} thread_state;

/**
 * @brief The name of sub-plugin for defined neural net frameworks.
 * @note The sub-plugin for Android is not declared (e.g., snap)
 */
static const char *ml_nnfw_subplugin_name[] = {
  [ML_NNFW_TYPE_ANY] = "any",   /* DO NOT use this name ('any') to get the sub-plugin */
  [ML_NNFW_TYPE_CUSTOM_FILTER] = "custom",
  [ML_NNFW_TYPE_TENSORFLOW_LITE] = "tensorflow-lite",
  [ML_NNFW_TYPE_TENSORFLOW] = "tensorflow",
  [ML_NNFW_TYPE_NNFW] = "nnfw",
  [ML_NNFW_TYPE_MVNC] = "movidius-ncsdk2",
  [ML_NNFW_TYPE_OPENVINO] = "openvino",
  [ML_NNFW_TYPE_VIVANTE] = "vivante",
  [ML_NNFW_TYPE_EDGE_TPU] = "edgetpu",
  [ML_NNFW_TYPE_ARMNN] = "armnn",
  [ML_NNFW_TYPE_SNPE] = "snpe",
  [ML_NNFW_TYPE_PYTORCH] = "pytorch",
  [ML_NNFW_TYPE_NNTR_INF] = "nntrainer",
  [ML_NNFW_TYPE_VD_AIFW] = "vd_aifw",
  [ML_NNFW_TYPE_TRIX_ENGINE] = "trix-engine",
  [ML_NNFW_TYPE_MXNET] = "mxnet",
  [ML_NNFW_TYPE_TVM] = "tvm",
  NULL
};

/** ML single api data structure for handle */
typedef struct
{
  GTensorFilterSingleClass *klass;    /**< tensor filter class structure*/
  GTensorFilterSingle *filter;        /**< tensor filter element */
  ml_tensors_info_s in_info;          /**< info about input */
  ml_tensors_info_s out_info;         /**< info about output */
  ml_nnfw_type_e nnfw;                /**< nnfw type for this filter */
  guint magic;                        /**< code to verify valid handle */

  GThread *thread;                    /**< thread for invoking */
  GMutex mutex;                       /**< mutex for synchronization */
  GCond cond;                         /**< condition for synchronization */
  ml_tensors_data_h input;            /**< input received from user */
  ml_tensors_data_h output;           /**< output to be sent back to user */
  guint timeout;                      /**< timeout for invoking */
  thread_state state;                 /**< current state of the thread */
  gboolean free_output;               /**< true if output tensors are allocated in single-shot */
  int status;                         /**< status of processing */
  gboolean invoking;                  /**< invoke running flag */
  ml_tensors_data_s in_tensors;    /**< input tensor wrapper for processing */
  ml_tensors_data_s out_tensors;   /**< output tensor wrapper for processing */

  /** @todo Use only ml_tensor_info_s dimension instead of saving ranks value */
  guint input_ranks[ML_TENSOR_SIZE_LIMIT];   /**< the rank list of input tensors, it is calculated based on the dimension string. */
  guint output_ranks[ML_TENSOR_SIZE_LIMIT];  /**< the rank list of output tensors, it is calculated based on the dimension string. */

  GList *destroy_data_list;         /**< data to be freed by filter */
} ml_single;

/**
 * @brief Internal function to get the nnfw type.
 */
ml_nnfw_type_e
_ml_get_nnfw_type_by_subplugin_name (const char *name)
{
  ml_nnfw_type_e nnfw_type = ML_NNFW_TYPE_ANY;
  int idx = -1;

  if (name == NULL)
    return ML_NNFW_TYPE_ANY;

  idx = find_key_strv (ml_nnfw_subplugin_name, name);
  if (idx < 0) {
    /* check sub-plugin for android */
    if (g_ascii_strcasecmp (name, "snap") == 0)
      nnfw_type = ML_NNFW_TYPE_SNAP;
    else
      _ml_error_report ("Cannot find nnfw, %s is an invalid name.",
          _STR_NULL (name));
  } else {
    nnfw_type = (ml_nnfw_type_e) idx;
  }

  return nnfw_type;
}

/**
 * @brief Internal function to get the sub-plugin name.
 */
const char *
_ml_get_nnfw_subplugin_name (ml_nnfw_type_e nnfw)
{
  /* check sub-plugin for android */
  if (nnfw == ML_NNFW_TYPE_SNAP)
    return "snap";

  return ml_nnfw_subplugin_name[nnfw];
}

/**
 * @brief Convert c-api based hw to internal representation
 */
accl_hw
_ml_nnfw_to_accl_hw (const ml_nnfw_hw_e hw)
{
  switch (hw) {
    case ML_NNFW_HW_ANY:
      return ACCL_DEFAULT;
    case ML_NNFW_HW_AUTO:
      return ACCL_AUTO;
    case ML_NNFW_HW_CPU:
      return ACCL_CPU;
#if defined (__aarch64__) || defined (__arm__)
    case ML_NNFW_HW_CPU_NEON:
      return ACCL_CPU_NEON;
#else
    case ML_NNFW_HW_CPU_SIMD:
      return ACCL_CPU_SIMD;
#endif
    case ML_NNFW_HW_GPU:
      return ACCL_GPU;
    case ML_NNFW_HW_NPU:
      return ACCL_NPU;
    case ML_NNFW_HW_NPU_MOVIDIUS:
      return ACCL_NPU_MOVIDIUS;
    case ML_NNFW_HW_NPU_EDGE_TPU:
      return ACCL_NPU_EDGE_TPU;
    case ML_NNFW_HW_NPU_VIVANTE:
      return ACCL_NPU_VIVANTE;
    case ML_NNFW_HW_NPU_SLSI:
      return ACCL_NPU_SLSI;
    case ML_NNFW_HW_NPU_SR:
      /** @todo how to get srcn npu */
      return ACCL_NPU_SR;
    default:
      return ACCL_AUTO;
  }
}

/**
 * @brief Checks the availability of the given execution environments with custom option.
 */
int
ml_check_nnfw_availability_full (ml_nnfw_type_e nnfw, ml_nnfw_hw_e hw,
    const char *custom, bool *available)
{
  const char *fw_name = NULL;

  check_feature_state (ML_FEATURE_INFERENCE);

  if (!available)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, available (bool *), is NULL. It should be a valid pointer of bool. E.g., bool a; ml_check_nnfw_availability_full (..., &a);");

  /* init false */
  *available = false;

  if (nnfw == ML_NNFW_TYPE_ANY)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, nnfw (ml_nnfw_type_e), is ML_NNFW_TYPE_ANY. It should specify the framework to be probed for the hardware availability.");

  fw_name = _ml_get_nnfw_subplugin_name (nnfw);

  if (fw_name) {
    if (nnstreamer_filter_find (fw_name) != NULL) {
      accl_hw accl = _ml_nnfw_to_accl_hw (hw);

      if (gst_tensor_filter_check_hw_availability (fw_name, accl, custom)) {
        *available = true;
      } else {
        _ml_logi ("%s is supported but not with the specified hardware.",
            fw_name);
      }
    } else {
      _ml_logi ("%s is not supported.", fw_name);
    }
  } else {
    _ml_logw ("Cannot get the name of sub-plugin for given nnfw.");
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Checks the availability of the given execution environments.
 */
int
ml_check_nnfw_availability (ml_nnfw_type_e nnfw, ml_nnfw_hw_e hw,
    bool *available)
{
  return ml_check_nnfw_availability_full (nnfw, hw, NULL, available);
}

/**
 * @brief setup input and output tensor memory to pass to the tensor_filter.
 * @note this tensor memory wrapper will be reused for each invoke.
 */
static void
__setup_in_out_tensors (ml_single * single_h)
{
  int i;
  ml_tensors_data_s *in_tensors = &single_h->in_tensors;
  ml_tensors_data_s *out_tensors = &single_h->out_tensors;

  /** Setup input buffer */
  _ml_tensors_info_free (in_tensors->info);
  ml_tensors_info_clone (in_tensors->info, &single_h->in_info);

  in_tensors->num_tensors = single_h->in_info.num_tensors;
  for (i = 0; i < single_h->in_info.num_tensors; i++) {
    /** memory will be allocated by tensor_filter_single */
    in_tensors->tensors[i].tensor = NULL;
    in_tensors->tensors[i].size =
        _ml_tensor_info_get_size (&single_h->in_info.info[i],
        single_h->in_info.is_extended);
  }

  /** Setup output buffer */
  _ml_tensors_info_free (out_tensors->info);
  ml_tensors_info_clone (out_tensors->info, &single_h->out_info);

  out_tensors->num_tensors = single_h->out_info.num_tensors;
  for (i = 0; i < single_h->out_info.num_tensors; i++) {
    /** memory will be allocated by tensor_filter_single */
    out_tensors->tensors[i].tensor = NULL;
    out_tensors->tensors[i].size =
        _ml_tensor_info_get_size (&single_h->out_info.info[i],
        single_h->out_info.is_extended);
  }
}

/**
 * @brief To call the framework to destroy the allocated output data
 */
static inline void
__destroy_notify (gpointer data_h, gpointer single_data)
{
  ml_single *single_h;
  ml_tensors_data_s *data;

  data = (ml_tensors_data_s *) data_h;
  single_h = (ml_single *) single_data;

  if (G_LIKELY (single_h->filter)) {
    if (single_h->klass->allocate_in_invoke (single_h->filter)) {
      single_h->klass->destroy_notify (single_h->filter,
          (GstTensorMemory *) data->tensors);
    }
  }

  /* reset callback function */
  data->destroy = NULL;
}

/**
 * @brief Wrapper function for __destroy_notify
 */
static int
ml_single_destroy_notify_cb (void *handle, void *user_data)
{
  ml_tensors_data_h data = (ml_tensors_data_h) handle;
  ml_single_h single = (ml_single_h) user_data;
  ml_single *single_h;
  int status = ML_ERROR_NONE;

  if (G_UNLIKELY (!single))
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "Failed to destroy data buffer. Callback function argument from _ml_tensors_data_destroy_internal is invalid. The given 'user_data' is NULL. It appears to be an internal error of ML-API or the user thread has touched private data structure.");
  if (G_UNLIKELY (!data))
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "Failed to destroy data buffer. Callback function argument from _ml_tensors_data_destroy_internal is invalid. The given 'handle' is NULL. It appears to be an internal error of ML-API or the user thread has touched private data structure.");

  ML_SINGLE_GET_VA