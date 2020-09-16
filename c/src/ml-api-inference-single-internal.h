/* SPDX-License-Identifier: Apache-2.0 */
/**
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file ml-api-inference-single-internal.h
 * @date 24 Jan 2022
 * @brief ML C-API single internal header with NNStreamer deps.
 *        This file should NOT be exported to SDK or devel package.
 * @see	https://github.com/nnstreamer/api
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __ML_API_INF_SINGLE_INTERNAL_H__
#define __ML_API_INF_SINGLE_INTERNAL_H__

#include <glib.h>

#include "ml-api-internal.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Internal function to get the sub-plugin name.
 */
const char * _ml_get_nnfw_subplugin_name (ml_nnfw_type_e nnfw);

/**
 * @brief Convert c-api based hw to internal representation
 */
accl_hw _ml_nnfw_to_accl_hw (const ml_nnfw_hw_e hw);

/**
 * @brief Internal function to get the nnfw type.
 * @return Returns ML_NNFW_TYPE_ANY if there is an error.
 */
ml_nnfw_type_e _ml_get_nnfw_type_by_subplugin_name (co