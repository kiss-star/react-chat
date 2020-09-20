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
 * @param[in] single Th