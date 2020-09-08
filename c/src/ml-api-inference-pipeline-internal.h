/* SPDX-License-Identifier: Apache-2.0 */
/**
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file ml-api-inference-pipeline-internal.h
 * @date 24 Jan 2022
 * @brief ML C-API pipeline internal header with NNStreamer deps.
 *        This file should NOT be exported to SDK or devel package.
 * @see	https://github.com/nnstreamer/api
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __ML_API_INF_PIPELINE_INTERNAL_H__
#define __ML_API_INF_PIPELINE_INTERNAL_H__

#include <glib.h>
#include <gst/gst.h>
#include <nnstreamer_internal.h>

#include "ml-api-internal.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/***** Wrappers of tizen-api-internal.h for pipelines *****/
#if defined (__TIZEN__)
#if defined (__PRIVILEGE_CHECK_SUPPORT__)

#define convert_tizen_elemen