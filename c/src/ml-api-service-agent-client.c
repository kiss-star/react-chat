
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file ml-api-service-agent-client.c
 * @date 20 Jul 2022
 * @brief agent (dbus) implementation of NNStreamer/Service C-API
 * @see https://github.com/nnstreamer/nnstreamer
 * @author Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <glib/gstdio.h>
#include <json-glib/json-glib.h>

#include "ml-api-internal.h"
#include "ml-api-service.h"
#include "ml-api-service-private.h"

/**
 * @brief Set the pipeline description with a given name.
 */
int
ml_service_set_pipeline (const char *name, const char *pipeline_desc)
{
  int ret = ML_ERROR_NONE;
  MachinelearningServicePipeline *mlsp;
  GError *err = NULL;
  gboolean result;

  check_feature_state (ML_FEATURE_SERVICE);