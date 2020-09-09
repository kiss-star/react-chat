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

#define convert_tizen_element(...) _ml_tizen_convert_element(__VA_ARGS__)

#if (TIZENVERSION >= 5) && (TIZENVERSION < 9999)
#define get_tizen_resource(...) _ml_tizen_get_resource(__VA_ARGS__)
#define release_tizen_resource(...) _ml_tizen_release_resource(__VA_ARGS__)

#elif (TIZENVERSION < 5)
#define get_tizen_resource(...) (0)
#define release_tizen_resource(...) do { } while (0)
typedef void * mm_resource_manager_h;
typedef enum { MM_RESOURCE_MANAGER_RES_TYPE_MAX } mm_resource_manager_res_type_e;

#else /* TIZENVERSION */
#error Tizen version is not defined.
#endif /* TIZENVERSION */

#else /* __PRIVILEGE_CHECK_SUPPORT__ */

#define convert_tizen_element(...) ML_ERROR_NONE
#define get_tizen_resource(...) ML_ERROR_NONE
#define release_tizen_resource(...)

#endif  /* __PRIVILEGE_CHECK_SUPPORT__ */

#else /* __TIZEN */

#define convert_tizen_element(...) ML_ERROR_NONE
#define get_tizen_resource(...) ML_ERROR_NONE
#define release_tizen_resource(...)

#endif  /* __TIZEN__ */

/**
 * @brief Internal private representation of custom filter handle.
 */
typedef struct {
  char *name;
  unsigned int ref_count;
  GMutex lock;
  ml_tensors_info_h in_info;
  ml_tensors_info_h out_info;
  ml_custom_easy_invoke_cb cb;
  void *pdata;
} ml_custom_filter_s;

/**
 * @brief Internal private representation of tensor_if custom condition.
 * @since_tizen 6.5
 */
typedef struct {
  char *name;
  unsigned int ref_count;
  GMutex lock;
  ml_pipeline_if_custom_cb cb;
  void *pdata;
} ml_if_custom_s;

/**
 * @brief Possible controls on elements of a pipeline.
 */
typedef enum {
  ML_PIPELINE_ELEMENT_UNKNOWN = 0x0,
  ML_PIPELINE_ELEMENT_SINK = 0x1,
  ML_PIPELINE_ELEMENT_APP_SRC = 0x2,
  ML_PIPELINE_ELEMENT_APP_SINK = 0x3,
  ML_PI