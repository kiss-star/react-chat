/* SPDX-License-Identifier: Apache-2.0 */
/**
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file ml-api-inference-tizen-privilege-check.c
 * @date 22 July 2020
 * @brief NNStreamer/C-API Tizen dependent functions for inference APIs.
 * @see	https://github.com/nnstreamer/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#if !defined (__TIZEN__) || !defined (__PRIVILEGE_CHECK_SUPPORT__)
#error "This file can be included only in Tizen."
#endif

#include <glib.h>
#include <nnstreamer.h>
#include "ml-api-internal.h"
#include "ml-api-inference-internal.h"
#include "ml-api-inference-pipeline-internal.h"

#include <system_info.h>
#include <restriction.h>        /* device policy manager */
#if TIZENPPM
#include <privacy_privilege_manager.h>
#endif
#if TIZEN5PLUS
#include <mm_resource_manager.h>
#endif
#include <mm_camcorder.h>


#if TIZENMMCONF
/* We can use "MMCAM_VIDEOSRC_ELEMENT_NAME and MMCAM_AUDIOSRC_ELEMENT_NAME */
#else /* TIZENMMCONF */
/* Tizen multimedia framework */
/* Defined in mm_camcorder_configure.h */

/**
 * @brief Structure to parse ini file for mmfw elements.
 */
typedef struct _type_int
{
  const char *name;
  int value;
} type_int;

/**
 * @brief Structure to parse ini file for mmfw elements.
 */
typedef struct _type_string
{
  const char *name;
  const char *value;
} type_string;

/**
 * @brief Structure to parse ini file for mmfw elements.
 */
typedef struct _type_element
{
  const char *name;
  const char *element_name;
  type_int **value_int;
  int count_int;
  type_string **value_string;
  int count_string;
} type_ele