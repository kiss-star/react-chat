/* SPDX-License-Identifier: Apache-2.0 */
/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file nnstreamer.h
 * @date 07 March 2019
 * @brief NNStreamer/Pipeline(main) C-API Header.
 *        This allows to construct and control NNStreamer pipelines.
 * @see	https://github.com/nnstreamer/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __TIZEN_MACHINELEARNING_NNSTREAMER_H__
#define __TIZEN_MACHINELEARNING_NNSTREAMER_H__

#include <ml-api-common.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
/**
 * @addtogroup CAPI_ML_NNSTREAMER_PIPELINE_MODULE
 * @{
 */

/**
 * @brief The virtual name to set the video source of camcorder in Tizen.
 * @details If an application needs to access the camera device to construct the pipeline, set the virtual name as a video source element.
 *          Note that you have to add '%http://tizen.org/privilege/camera' into the manifest of your application.
 * @since_tizen 5.5
 */
#define ML_TIZEN_CAM_VIDEO_SRC "tizencamvideosrc"

/**
 * @brief The virtual name to set the audio source of camcorder in Tizen.
 * @details If an application needs to access the recorder device to construct the pipeline, set the virtual name as an audio source element.
 *          Note that you have to add '%http://tizen.org/privilege/recorder' into the manifest of your application.
 * @since_tizen 5.5
 */
#define ML_TIZEN_CAM_AUDIO_SRC "tizencamaudiosrc"

/**
 * @brief A handle of an NNStreamer pipeline.
 * @since_tizen 5.5
 */
typedef void *ml_pipeline_h;

/**
 * @brief A handle of a "sink node" of an NNStreamer pipeline.
 * @since_tizen 5.5
 */
typedef void *ml_pipeline_sink_h;

/**
 * @brief A handle of a "src node" of an NNStreamer pipeline.
 * @since_tizen 5.5
 */
typedef void *ml_pipeline_src_h;

/**
 * @brief A handle of a "switch" of an NNStreamer pipeline.
 * @since_tizen 5.5
 */
typedef void *ml_pipeline_switch_h;

/**
 * @brief A handle of a "valve node" of an NNStreamer pipeline.
 * @since_tizen 5.5
 */
typedef void *ml_pipeline_valve_h;

/**
 * @brief A handle of a common element (i.e. All GstElement except AppSrc, AppSink, TensorSink, Selector and Valve) of an NNStreamer pipeline.
 * @since_tizen 6.0
 */
typedef void *ml_pipeline_element_h;

/**
 * @brief A handle of a "custom-easy filter" of an NNStreamer pipeline.
 * @since_tizen 6.0
 */
typedef void *ml_custom_easy_filter_h;

/**
 * @brief A handle of a "if node" of an NNStreamer pipeline.
 * @since_tizen 6.5
 */
typedef void *ml_pipeline_if_h;

/**
 * @brief Enumeration for buffer deallocation policies.
 * @since_tizen 5.5
 */
typedef enum {
  ML_PIPELINE_BUF_POLICY_AUTO_FREE      = 0, /**< Default. Application should not deallocate this buffer. NNStreamer will deallocate when the buffer is no more needed. */
  ML_PIPELINE_BUF_POLICY_DO_NOT_FREE    = 1, /**< This buffer is not to be freed by NNStreamer (i.e., it's a static object). However, be careful: NNStreamer might be accessing this object after the return of the API call. */
  ML_PIPELINE_BUF_POLICY_MAX,   /**< Max size of #ml_pipeline_buf_policy_e structure. */
  ML_PIPELINE_BUF_SRC_EVENT_EOS         = 0x10000, /**< Trigger End-Of-Stream event for the corresponding appsrc and ignore the given input value. The corresponding appsrc will no longer accept new data after this. */
} ml_pipeline_buf_policy_e;

/**
 * @brief Enumeration for pipeline state.
 * @details The pipeline state is described on @ref CAPI_ML_NNSTREAMER_PIPELINE_STATE_DIAGRAM.
 * Refer to https://gstreamer.freedesktop.org/documentation/plugin-development/basics/states.html.
 * @since_tizen 5.5
 */
typedef enum {
  ML_PIPELINE_STATE_UNKNOWN				= 0, /**< Unknown state. Maybe not constructed? */
  ML_PIPELINE_STATE_NULL				= 1, /**< GST-State "Null" */
  ML_PIPELINE_STATE_READY				= 2, /**< GST-State "Ready" */
  ML_PIPELINE_STATE_PAUSED				= 3, /**< GST-State "Paused" */
  ML_PIPELINE_STATE_PLAYING				= 4, /**< GST-State "Playing" */
} ml_pipeline_state_e;

/**
 * @brief Enumeration for switch types.
 * @details This designates different GStreamer filters, "GstInputSelector"/"GstOutputSelector".
 * @since_tizen 5.5
 */
typedef enum {
  ML_PIPELINE_SWITCH_OUTPUT_SELECTOR			= 0, /**< GstOutputSelector */
  ML_PIPELINE_SWITCH_INPUT_SELECTOR			= 1, /**< GstInputSelector */
} ml_pipeline_switch_e;

/**
 * @brief Callback for sink element of NNStreamer pipelines (pipeline's output).
 * @details If an application wants to accept data outputs of an NNStreamer stream, use this callb