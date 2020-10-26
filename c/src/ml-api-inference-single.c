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

  ML_SINGLE_GET_VALID_HANDLE_LOCKED (single_h, single, 0);

  if (G_UNLIKELY (!single_h->filter)) {
    status = ML_ERROR_INVALID_PARAMETER;
    _ml_error_report
        ("Failed to destroy the data buffer. The handle instance (single_h) is invalid. It appears to be an internal error of ML-API of the user thread has touched private data structure.");
    goto exit;
  }

  single_h->destroy_data_list =
      g_list_remove (single_h->destroy_data_list, data);
  __destroy_notify (data, single_h);

exit:
  ML_SINGLE_HANDLE_UNLOCK (single_h);

  return status;
}

/**
 * @brief setup the destroy notify for the allocated output data.
 * @note this stores the data entry in the single list.
 * @note this has not overhead if the allocation of output is not performed by
 * the framework but by tensor filter element.
 */
static void
set_destroy_notify (ml_single * single_h, ml_tensors_data_s * data,
    gboolean add)
{
  if (single_h->klass->allocate_in_invoke (single_h->filter)) {
    data->destroy = ml_single_destroy_notify_cb;
    data->user_data = single_h;
    add = TRUE;
  }

  if (add) {
    single_h->destroy_data_list = g_list_append (single_h->destroy_data_list,
        (gpointer) data);
  }
}

/**
 * @brief Internal function to call subplugin's invoke
 */
static inline int
__invoke (ml_single * single_h, ml_tensors_data_h in, ml_tensors_data_h out)
{
  ml_tensors_data_s *in_data, *out_data;
  int status = ML_ERROR_NONE;
  GstTensorMemory *in_tensors, *out_tensors;

  in_data = (ml_tensors_data_s *) in;
  out_data = (ml_tensors_data_s *) out;

  /* Prevent error case when input or output is null in invoke thread. */
  if (!in_data || !out_data) {
    _ml_error_report ("Failed to invoke a model, invalid data handle.");
    return ML_ERROR_STREAMS_PIPE;
  }

  in_tensors = (GstTensorMemory *) in_data->tensors;
  out_tensors = (GstTensorMemory *) out_data->tensors;

  /** invoke the thread */
  if (!single_h->klass->invoke (single_h->filter, in_tensors, out_tensors,
          single_h->free_output)) {
    const char *fw_name = _ml_get_nnfw_subplugin_name (single_h->nnfw);
    _ml_error_report
        ("Failed to invoke the tensors. The invoke callback of the tensor-filter subplugin '%s' has failed. Please contact the author of tensor-filter-%s (nnstreamer-%s) or review its source code. Note that this usually happens when the designated framework does not support the given model (e.g., trying to run tf-lite 2.6 model with tf-lite 1.13).",
        fw_name, fw_name, fw_name);
    status = ML_ERROR_STREAMS_PIPE;
  }

  return status;
}

/**
 * @brief Internal function to post-process given output.
 */
static inline void
__process_output (ml_single * single_h, ml_tensors_data_h output)
{
  ml_tensors_data_s *out_data;

  if (!single_h->free_output) {
    /* Do nothing. The output handle is not allocated in single-shot process. */
    return;
  }

  if (g_list_find (single_h->destroy_data_list, output)) {
    /**
     * Caller of the invoke thread has returned back with timeout.
     * So, free the memory allocated by the invoke as their is no receiver.
     */
    single_h->destroy_data_list =
        g_list_remove (single_h->destroy_data_list, output);
    ml_tensors_data_destroy (output);
  } else {
    out_data = (ml_tensors_data_s *) output;
    set_destroy_notify (single_h, out_data, FALSE);
  }
}

/**
 * @brief Initializes the rank information with default value.
 */
static int
_ml_tensors_rank_initialize (guint * rank)
{
  guint i;

  if (!rank)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, rank, is NULL. Provide a valid pointer.");

  for (i = 0; i < ML_TENSOR_SIZE_LIMIT; i++) {
    rank[i] = 0;
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Sets the rank information with given value.
 */
static int
_ml_tensors_set_rank (guint * rank, guint val)
{
  guint i;

  if (!rank)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, rank, is NULL. Provide a valid pointer.");

  for (i = 0; i < ML_TENSOR_SIZE_LIMIT; i++) {
    rank[i] = val;
  }

  return ML_ERROR_NONE;
}

/**
 * @brief thread to execute calls to invoke
 *
 * @details The thread behavior is detailed as below:
 *          - Starting with IDLE state, the thread waits for an input or change
 *          in state externally.
 *          - If state is not RUNNING, exit this thread, else process the
 *          request.
 *          - Process input, call invoke, process output. Any error in this
 *          state sets the status to be used by ml_single_invoke().
 *          - State is set back to IDLE and thread moves back to start.
 *
 *          State changes performed by this function when:
 *          RUNNING -> IDLE - processing is finished.
 *          JOIN_REQUESTED -> IDLE - close is requested.
 *
 * @note Error while processing an input is provided back to requesting
 *       function, and further processing of invoke_thread is not affected.
 */
static void *
invoke_thread (void *arg)
{
  ml_single *single_h;
  ml_tensors_data_h input, output;

  single_h = (ml_single *) arg;

  g_mutex_lock (&single_h->mutex);

  while (single_h->state <= RUNNING) {
    int status = ML_ERROR_NONE;

    /** wait for data */
    while (single_h->state != RUNNING) {
      g_cond_wait (&single_h->cond, &single_h->mutex);
      if (single_h->state >= JOIN_REQUESTED)
        goto exit;
    }

    input = single_h->input;
    output = single_h->output;
    /* Set null to prevent double-free. */
    single_h->input = NULL;

    single_h->invoking = TRUE;
    g_mutex_unlock (&single_h->mutex);
    status = __invoke (single_h, input, output);
    g_mutex_lock (&single_h->mutex);
    /* Clear input data after invoke is done. */
    ml_tensors_data_destroy (input);
    single_h->invoking = FALSE;

    if (status != ML_ERROR_NONE) {
      if (single_h->free_output) {
        single_h->destroy_data_list =
            g_list_remove (single_h->destroy_data_list, output);
        ml_tensors_data_destroy (output);
      }

      goto wait_for_next;
    }

    __process_output (single_h, output);

    /** loop over to wait for the next element */
  wait_for_next:
    single_h->status = status;
    if (single_h->state == RUNNING)
      single_h->state = IDLE;
    g_cond_broadcast (&single_h->cond);
  }

exit:
  /* Do not set IDLE if JOIN_REQUESTED */
  if (single_h->state == RUNNING)
    single_h->state = IDLE;
  g_mutex_unlock (&single_h->mutex);
  return NULL;
}

/**
 * @brief Sets the information (tensor dimension, type, name and so on) of required input data for the given model, and get updated output data information.
 * @details Note that a model/framework may not support setting such information.
 * @since_tizen 6.0
 * @param[in] single The model handle.
 * @param[in] in_info The handle of input tensors information.
 * @param[out] out_info The handle of output tensors information. The caller is responsible for freeing the information with ml_tensors_info_destroy().
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED This implies that the given framework does not support dynamic dimensions.
 *         Use ml_single_get_input_info() and ml_single_get_output_info() instead for this framework.
 * @retval #ML_ERROR_INVALID_PARAMETER Fail. The parameter is invalid.
 */
static int
ml_single_update_info (ml_single_h single,
    const ml_tensors_info_h in_info, ml_tensors_info_h * out_info)
{
  if (!single)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, single (ml_single_h), is NULL. It should be a valid ml_single_h instance, usually created by ml_single_open().");
  if (!in_info)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, in_info (const ml_tensors_info_h), is NULL. It should be a valid instance of ml_tensors_info_h, usually created by ml_tensors_info_create() and configured by the application.");
  if (!out_info)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, out_info (ml_tensors_info_h *), is NULL. It should be a valid pointer to an instance ml_tensors_info_h, usually created by ml_tensors_info_h(). Note that out_info is supposed to be overwritten by this API call.");

  /* init null */
  *out_info = NULL;

  _ml_error_report_return_continue_iferr (ml_single_set_input_info (single,
          in_info),
      "Configuring the neural network model with the given input information has failed with %d error code. The given input information ('in_info' parameter) might be invalid or the given neural network cannot accept it as its input data.",
      _ERRNO);

  __setup_in_out_tensors (single);
  _ml_error_report_return_continue_iferr (ml_single_get_output_info (single,
          out_info),
      "Fetching output info after configuring input information has failed with %d error code.",
      _ERRNO);

  return ML_ERROR_NONE;
}

/**
 * @brief Internal function to get the gst info from tensor-filter.
 */
static void
ml_single_get_gst_info (ml_single * single_h, gboolean is_input,
    GstTensorsInfo * gst_info)
{
  const gchar *prop_prefix, *prop_name, *prop_type;
  gchar *val;
  guint num;

  if (is_input) {
    prop_prefix = INPUT_STR;
    prop_type = CONCAT_MACRO_STR (INPUT_STR, TYPE_STR);
    prop_name = CONCAT_MACRO_STR (INPUT_STR, NAME_STR);
  } else {
    prop_prefix = OUTPUT_STR;
    prop_type = CONCAT_MACRO_STR (OUTPUT_STR, TYPE_STR);
    prop_name = CONCAT_MACRO_STR (OUTPUT_STR, NAME_STR);
  }

  gst_tensors_info_init (gst_info);

  /* get dimensions */
  g_object_get (single_h->filter, prop_prefix, &val, NULL);
  num = gst_tensors_info_parse_dimensions_string (gst_info, val);
  g_free (val);

  /* set the number of tensors */
  gst_info->num_tensors = num;

  /* get types */
  g_object_get (single_h->filter, prop_type, &val, NULL);
  num = gst_tensors_info_parse_types_string (gst_info, val);
  g_free (val);

  if (gst_info->num_tensors != num) {
    _ml_logw ("The number of tensor type is mismatched in filter.");
  }

  /* get names */
  g_object_get (single_h->filter, prop_name, &val, NULL);
  num = gst_tensors_info_parse_names_string (gst_info, val);
  g_free (val);

  if (gst_info->num_tensors != num) {
    _ml_logw ("The number of tensor name is mismatched in filter.");
  }
}

/**
 * @brief Internal function to set the gst info in tensor-filter.
 */
static int
ml_single_set_gst_info (ml_single * single_h, const ml_tensors_info_h info)
{
  GstTensorsInfo gst_in_info, gst_out_info;
  int status = ML_ERROR_NONE;
  int ret = -EINVAL;

  _ml_error_report_return_continue_iferr
      (_ml_tensors_info_copy_from_ml (&gst_in_info, info),
      "Cannot fetch tensor-info from the given info parameter. Error code: %d",
      _ERRNO);

  ret = single_h->klass->set_input_info (single_h->filter, &gst_in_info,
      &gst_out_info);
  if (ret == 0) {
    _ml_error_report_return_continue_iferr
        (_ml_tensors_info_copy_from_gst (&single_h->in_info, &gst_in_info),
        "Fetching input information from the given single_h instance has failed with %d",
        _ERRNO);
    _ml_error_report_return_continue_iferr (_ml_tensors_info_copy_from_gst
        (&single_h->out_info, &gst_out_info),
        "Fetching output information from the given single_h instance has failed with %d",
        _ERRNO);
    __setup_in_out_tensors (single_h);
  } else if (ret == -ENOENT) {
    status = ML_ERROR_NOT_SUPPORTED;
  } else {
    status = ML_ERROR_INVALID_PARAMETER;
  }

  return status;
}

/**
 * @brief Set the info for input/output tensors
 */
static int
ml_single_set_inout_tensors_info (GObject * object,
    const gboolean is_input, ml_tensors_info_s * tensors_info)
{
  int status = ML_ERROR_NONE;
  GstTensorsInfo info;
  gchar *str_dim, *str_type, *str_name;
  const gchar *str_type_name, *str_name_name;
  const gchar *prefix;

  if (is_input) {
    prefix = INPUT_STR;
    str_type_name = CONCAT_MACRO_STR (INPUT_STR, TYPE_STR);
    str_name_name = CONCAT_MACRO_STR (INPUT_STR, NAME_STR);
  } else {
    prefix = OUTPUT_STR;
    str_type_name = CONCAT_MACRO_STR (OUTPUT_STR, TYPE_STR);
    str_name_name = CONCAT_MACRO_STR (OUTPUT_STR, NAME_STR);
  }

  _ml_error_report_return_continue_iferr
      (_ml_tensors_info_copy_from_ml (&info, tensors_info),
      "Cannot fetch tensor-info from the given information. Error code: %d",
      _ERRNO);

  /* Set input option */
  str_dim = gst_tensors_info_get_dimensions_string (&info);
  str_type = gst_tensors_info_get_types_string (&info);
  str_name = gst_tensors_info_get_names_string (&info);

  if (!str_dim || !str_type || !str_name) {
    if (!str_dim)
      _ml_error_report
          ("Cannot fetch specific tensor-info from the given information: cannot fetch tensor dimension information.");
    if (!str_type)
      _ml_error_report
          ("Cannot fetch specific tensor-info from the given information: cannot fetch tensor type information.");
    if (!str_name)
      _ml_error_report
          ("Cannot fetch specific tensor-info from the given information: cannot fetch tensor name information. Even if tensor names are not defined, this should be able to fetch a list of empty strings.");

    status = ML_ERROR_INVALID_PARAMETER;
  } else {
    g_object_set (object, prefix, str_dim, str_type_name, str_type,
        str_name_name, str_name, NULL);
  }

  g_free (str_dim);
  g_free (str_type);
  g_free (str_name);

  gst_tensors_info_free (&info);

  return status;
}

/**
 * @brief Internal static function to set tensors info in the handle.
 */
static gboolean
ml_single_set_info_in_handle (ml_single_h single, gboolean is_input,
    ml_tensors_info_s * tensors_info)
{
  int status;
  ml_single *single_h;
  ml_tensors_info_s *dest;
  gboolean configured = FALSE;
  gboolean is_valid = FALSE;
  GObject *filter_obj;

  single_h = (ml_single *) single;
  filter_obj = G_OBJECT (single_h->filter);

  if (is_input) {
    dest = &single_h->in_info;
    configured = single_h->klass->input_configured (single_h->filter);
  } else {
    dest = &single_h->out_info;
    configured = single_h->klass->output_configured (single_h->filter);
  }

  if (configured) {
    /* get configured info and compare with input info */
    GstTensorsInfo gst_info;
    ml_tensors_info_h info = NULL;

    ml_single_get_gst_info (single_h, is_input, &gst_info);
    _ml_tensors_info_create_from_gst (&info, &gst_info);

    gst_tensors_info_free (&gst_info);

    if (tensors_info && !ml_tensors_info_is_equal (tensors_info, info)) {
      /* given input info is not matched with configured */
      ml_tensors_info_destroy (info);
      if (is_input) {
        /* try to update tensors info */
        status = ml_single_update_info (single, tensors_info, &info);
        if (status != ML_ERROR_NONE)
          goto done;
      } else {
        goto done;
      }
    }

    ml_tensors_info_clone (dest, info);
    ml_tensors_info_destroy (info);
  } else if (tensors_info) {
    status =
        ml_single_set_inout_tensors_info (filter_obj, is_input, tensors_info);
    if (status != ML_ERROR_NONE)
      goto done;
    ml_tensors_info_clone (dest, tensors_info);
  }

  is_valid = ml_tensors_info_is_valid (dest);

done:
  return is_valid;
}

/**
 * @brief Internal function to create and initialize the single handle.
 */
static ml_single *
ml_single_create_handle (ml_nnfw_type_e nnfw)
{
  ml_single *single_h;
  GError *error;

  single_h = g_new0 (ml_single, 1);
  if (single_h == NULL)
    _ml_error_report_return (NULL,
        "Failed to allocate memory for the single_h handle. Out of memory?");

  single_h->filter = g_object_new (G_TYPE_TENSOR_FILTER_SINGLE, NULL);
  if (single_h->filter == NULL) {
    _ml_error_report
        ("Failed to create a new instance for filter. Out of memory?");
    g_free (single_h);
    return NULL;
  }

  single_h->magic = ML_SINGLE_MAGIC;
  single_h->timeout = SINGLE_DEFAULT_TIMEOUT;
  single_h->nnfw = nnfw;
  single_h->state = IDLE;
  single_h->thread = NULL;
  single_h->input = NULL;
  single_h->output = NULL;
  single_h->destroy_data_list = NULL;
  single_h->invoking = FALSE;

  _ml_tensors_info_initialize (&single_h->in_info);
  _ml_tensors_info_initialize (&single_h->out_info);
  _ml_tensors_rank_initialize (single_h->input_ranks);
  _ml_tensors_rank_initialize (single_h->output_ranks);
  g_mutex_init (&single_h->mutex);
  g_cond_init (&single_h->cond);

  single_h->klass = g_type_class_ref (G_TYPE_TENSOR_FILTER_SINGLE);
  if (single_h->klass == NULL) {
    _ml_error_report
        ("Failed to get class of the tensor-filter of single API. This binary is not compiled properly or required libraries are not loaded.");
    ml_single_close (single_h);
    return NULL;
  }

  single_h->thread =
      g_thread_try_new (NULL, invoke_thread, (gpointer) single_h, &error);
  if (single_h->thread == NULL) {
    _ml_error_report
        ("Failed to create the invoke thread of single API, g_thread_try_new has reported an error: %s.",
        error->message);
    g_clear_error (&error);
    ml_single_close (single_h);
    return NULL;
  }

  return single_h;
}

/**
 * @brief Validate arguments for open
 */
static int
_ml_single_open_custom_validate_arguments (ml_single_h * single,
    ml_single_preset * info)
{
  if (!single)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, 'single' (ml_single_h *), is NULL. It should be a valid pointer to an instance of ml_single_h.");
  if (!info)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, 'info' (ml_single_preset *), is NULL. It should be a valid pointer to a valid instance of ml_single_preset.");

  /* Validate input tensor info. */
  if (info->input_info && !ml_tensors_info_is_valid (info->input_info))
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, 'info' (ml_single_preset *), is not valid. It has 'input_info' entry that cannot be validated. ml_tensors_info_is_valid(info->input_info) has failed while info->input_info exists.");

  /* Validate output tensor info. */
  if (info->output_info && !ml_tensors_info_is_valid (info->output_info))
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, 'info' (ml_single_preset *), is not valid. It has 'output_info' entry that cannot be validated. ml_tensors_info_is_valid(info->output_info) has failed while info->output_info exists.");

  if (!info->models)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, 'info' (ml_single_preset *), is not valid. Its models entry if NULL (info->models is NULL).");

  return ML_ERROR_NONE;
}

/**
 * @brief Internal function to convert accelerator as tensor_filter property format.
 * @note returned value must be freed by the caller
 * @note More details on format can be found in gst_tensor_filter_install_properties() in tensor_filter_common.c.
 */
char *
_ml_