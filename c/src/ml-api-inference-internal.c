/* SPDX-License-Identifier: Apache-2.0 */
/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file ml-api-inference-internal.c
 * @date 19 October 2021
 * @brief ML-API Internal Utility Functions for inference implementations
 * @see	https://github.com/nnstreamer/api
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <string.h>

#include <nnstreamer_plugin_api_util.h>
#include <tensor_typedef.h>
#include "ml-api-inference-internal.h"
#include "ml-api-internal.h"

/**
 * @brief Convert the type from ml_tensor_type_e to tensor_type.
 * @note This code is based on the same order between NNS type and ML type.
 * The index should be the same in case of adding a new type.
 */
static tensor_type
convert_tensor_type_from (ml_tensor_type_e type)
{
  if (type < ML_TENSOR_TYPE_INT32 || type >= ML_TENSOR_TYPE_UNKNOWN) {
    _ml_error_report
        ("Failed to convert the type. Input ml_tensor_type_e %d is invalid.",
        type);
    return _NNS_END;
  }

  return (tensor_type) type;
}

/**
 * @brief Convert the type from tensor_type to ml_tensor_type_e.
 * @note This code is based on the same order between NNS type and ML type.
 * The index should be the same in case of adding a new type.
 */
static ml_tensor_type_e
convert_ml_tensor_type_from (tensor_type type)
{
  if (type < _NNS_INT32 || type >= _NNS_END) {
    _ml_error_report
        ("Failed to convert the type. Input tensor_type %d is invalid.", type);
    return ML_TENSOR_TYPE_UNKNOWN;
  }

  return (ml_tensor_type_e) type;
}

static gboolean
gst_info_is_extended (const GstTensorsInfo * gst_info)
{
  int i, j;
  for (i = 0; i < 