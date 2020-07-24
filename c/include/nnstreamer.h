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
 * @details If an application wants to accept data outputs of an NNStreamer stream, use this callback to get data from the stream. Note that the buffer may be deallocated after the return and this is synchronously called. Thus, if you need the data afterwards, copy the data to another buffer and return fast. Do not spend too much time in the callback. It is recommended to use very small tensors at sinks.
 * @since_tizen 5.5
 * @remarks The @a data can be used only in the callback. To use outside, make a copy.
 * @remarks The @a info can be used only in the callback. To use outside, make a copy.
 * @param[in] data The handle of the tensor output of the pipeline (a single frame. tensor/tensors). Number of tensors is determined by ml_tensors_info_get_count() with the handle 'info'. Note that the maximum number of tensors is 16 (#ML_TENSOR_SIZE_LIMIT).
 * @param[in] info The handle of tensors information (cardinality, dimension, and type of given tensor/tensors).
 * @param[in,out] user_data User application's private data.
 */
typedef void (*ml_pipeline_sink_cb) (const ml_tensors_data_h data, const ml_tensors_info_h info, void *user_data);

/**
 * @brief Callback for the change of pipeline state.
 * @details If an application wants to get the change of pipeline state, use this callback. This callback can be registered when constructing the pipeline using ml_pipeline_construct(). Do not spend too much time in the callback.
 * @since_tizen 5.5
 * @param[in] state The new state of the pipeline.
 * @param[out] user_data User application's private data.
 */
typedef void (*ml_pipeline_state_cb) (ml_pipeline_state_e state, void *user_data);

/**
 * @brief Callback for custom condition of tensor_if.
 * @since_tizen 6.5
 * @remarks The @a data can be used only in the callback. To use outside, make a copy.
 * @remarks The @a info can be used only in the callback. To use outside, make a copy.
 * @remarks The @a result can be used only in the callback and should not be released.
 * @param[in] data The handle of the tensor output of the pipeline (a single frame. tensor/tensors). Number of tensors is determined by ml_tensors_info_get_count() with the handle 'info'. Note that the maximum number of tensors is 16 (#ML_TENSOR_SIZE_LIMIT).
 * @param[in] info The handle of tensors information (cardinality, dimension, and type of given tensor/tensors).
 * @param[out] result Result of the user-defined condition. 0 refers to FALSE and a non-zero value refers to TRUE. The application should set the result value for given data.
 * @param[in,out] user_data User application's private data.
 * @return @c 0 on success. Otherwise a negative error value.
 */
typedef int (*ml_pipeline_if_custom_cb) (const ml_tensors_data_h data, const ml_tensors_info_h info, int *result, void *user_data);

/****************************************************
 ** NNStreamer Pipeline Construction (gst-parse)   **
 ****************************************************/
/**
 * @brief Constructs the pipeline (GStreamer + NNStreamer).
 * @details Use this function to create gst_parse_launch compatible NNStreamer pipelines.
 * @since_tizen 5.5
 * @remarks If the function succeeds, @a pipe handle must be released using ml_pipeline_destroy().
 * @remarks %http://tizen.org/privilege/mediastorage is needed if @a pipeline_description is relevant to media storage.
 * @remarks %http://tizen.org/privilege/externalstorage is needed if @a pipeline_description is relevant to external storage.
 * @remarks %http://tizen.org/privilege/camera is needed if @a pipeline_description accesses the camera device.
 * @remarks %http://tizen.org/privilege/recorder is needed if @a pipeline_description accesses the recorder device.
 * @param[in] pipeline_description The pipeline description compatible with GStreamer gst_parse_launch(). Refer to GStreamer manual or NNStreamer (https://github.com/nnstreamer/nnstreamer) documentation for examples and the grammar.
 * @param[in] cb The function to be called when the pipeline state is changed. You may set NULL if it's not required.
 * @param[in] user_data Private data for the callback. This value is passed to the callback when it's invoked.
 * @param[out] pipe The NNStreamer pipeline handler from the given description.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE 