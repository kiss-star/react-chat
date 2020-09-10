
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file ml-api-inference-pipeline.c
 * @date 11 March 2019
 * @brief NNStreamer/Pipeline(main) C-API Wrapper.
 *        This allows to construct and control NNStreamer pipelines.
 * @see	https://github.com/nnstreamer/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug Thread safety for ml_tensors_data should be addressed.
 */

#include <string.h>
#include <glib.h>
#include <gst/gstbuffer.h>
#include <gst/app/app.h>        /* To push data to pipeline */
#include <nnstreamer_plugin_api.h>
#include <tensor_if.h>
#include <tensor_typedef.h>
#include <tensor_filter_custom_easy.h>

#include <nnstreamer.h>
#include <nnstreamer-tizen-internal.h>

#include "ml-api-internal.h"
#include "ml-api-inference-internal.h"
#include "ml-api-inference-pipeline-internal.h"


#define handle_init(name, h) \
  ml_pipeline_common_elem *name= (h); \
  ml_pipeline *p; \
  ml_pipeline_element *elem; \
  int ret = ML_ERROR_NONE; \
  check_feature_state (ML_FEATURE_INFERENCE); \
  if ((h) == NULL) { \
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER, \
        "The parameter, %s, (handle) is invalid (NULL). Please provide a valid handle.", \
        #h); \
  } \
  p = name->pipe; \
  elem = name->element; \
  if (p == NULL) \
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER, \
        "Internal error. The contents of parameter, %s, (handle), is invalid. The pipeline entry (%s->pipe) is NULL. The handle (%s) is either not properly created or application threads may have touched its contents.", \
        #h, #h, #h); \
  if (elem == NULL) \
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER, \
        "Internal error. The contents of parameter, %s, (handle), is invalid. The element entry (%s->element) is NULL. The handle (%s) is either not properly created or application threads may have touched its contents.", \
        #h, #h, #h); \
  if (elem->pipe == NULL) \
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER, \
        "Internal error. The contents of parameter, %s, (handle), is invalid. The pipeline entry of the element entry (%s->element->pipe) is NULL. The handle (%s) is either not properly created or application threads may have touched its contents.", \
        #h, #h, #h); \
  g_mutex_lock (&p->lock); \
  g_mutex_lock (&elem->lock); \
  if (NULL == g_list_find (elem->handles, name)) { \
    _ml_error_report \
        ("Internal error. The handle name, %s, does not exists in the list of %s->element->handles.", \
        #h, #h); \
    ret = ML_ERROR_INVALID_PARAMETER; \
    goto unlock_return; \
  }

#define handle_exit(h) \
unlock_return: \
  g_mutex_unlock (&elem->lock); \
  g_mutex_unlock (&p->lock); \
  return ret;

/**
 * @brief The enumeration for custom data type.
 */
typedef enum
{
  PIPE_CUSTOM_TYPE_NONE,
  PIPE_CUSTOM_TYPE_IF,
  PIPE_CUSTOM_TYPE_FILTER,

  PIPE_CUSTOM_TYPE_MAX
} pipe_custom_type_e;

/**
 * @brief The struct for custom data.
 */
typedef struct
{
  pipe_custom_type_e type;
  gchar *name;
  gpointer handle;
} pipe_custom_data_s;

static void ml_pipeline_custom_filter_ref (ml_custom_easy_filter_h custom);
static void ml_pipeline_custom_filter_unref (ml_custom_easy_filter_h custom);
static void ml_pipeline_if_custom_ref (ml_pipeline_if_h custom);
static void ml_pipeline_if_custom_unref (ml_pipeline_if_h custom);

/**
 * @brief Global lock for pipeline functions.
 */