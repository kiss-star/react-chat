/* SPDX-License-Identifier: Apache-2.0 */
/**
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file ml-api-service-query-client.c
 * @date 30 Aug 2022
 * @brief Query client implementation of NNStreamer/Service C-API
 * @see https://github.com/nnstreamer/nnstreamer
 * @author Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <glib.h>
#include <gst/gst.h>
#include <gst/gstbuffer.h>
#include <gst/app/app.h>
#include <string.h>

#include "ml-api-internal.h"
#include "ml-api-service.h"
#include "ml-api-service-private.h"

/**
 * @brief Sink callback for query_client
 */
static void
_sink_callback_for_query_client (const ml_tensors_data_h data,
    const ml_tensors_info_h info, void *user_data)
{
  _ml_service_query_s *mls = (_ml_service_query_s *) user_data;
  ml_tensors_data_s *data_s = (ml_tensors_data_s *) data;
  ml_tensors_data_h copied_data = NULL;
  ml_tensors_data_s *_copied_data_s;

  guint i, count = 0U;
  int status;

  status = ml_tensors_data_create (info, &copied_data);
  if (ML_ERROR_NONE != status) {
    _ml_error_report_continue
        ("Failed to create a new tensors data for query_client.");
    return;
  }
  _copied_data_s = (ml_tensors_data_s *) copied_data;

  status = ml_tensors_info_get_count (info, &count);
  if (ML_ERROR_NONE != status) {
    _ml_error_report_continue
        ("Failed to get count of tensors info from tensor_sink.");
    return;
  }

  for (i = 0; i < count; ++i) {
    memcpy (_copied_data_s->tensors[i].tensor, data_s->tensors[i].tensor,
        data_s->tensors[i].size);
  }

  g_async_queue_push (mls->out_data_queue, copied_data);
}

/**
 * @brief Creates query client service handle with given ml-option handle.
 */
int
ml_service_query_create (ml_option_h option, ml_service_h * h)
{
  int status = ML_ERROR_NONE;

  gchar *description = NULL;

  ml_option_s *_option;
  GHashTableIter iter;
  gchar *key;
  ml_option_value_s *_option_value;

  GString *tensor_query_client_prop;
  gchar *prop = NULL;

  ml_service_s *mls;

  _ml_service_query_s *query_s;
  ml_pipeline_h pipe_h;
  ml_pipeline_src_h src_h;
  ml_pipeline_sink_h sink_h;
  gchar *caps = NULL;
  guint timeout = 1000U;        /* default 1s timeout */

  check_feature_state (ML_FEATURE_SERVICE);
  check_feature_state (ML_FEATURE_INFERENCE);

  if (!option) {
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, 'option' is NULL. It should be a valid ml_option_h, which s