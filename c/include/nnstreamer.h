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
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_PERMISSION_DENIED The application does not have the required privilege to access to the media storage, external storage, microphone, or camera.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid. (Pipeline is not negotiated yet.)
 * @retval #ML_ERROR_STREAMS_PIPE Pipeline construction is failed because of wrong parameter or initialization failure.
 * @retval #ML_ERROR_OUT_OF_MEMORY Failed to allocate required memory to construct the pipeline.
 *
 * @pre The pipeline state should be #ML_PIPELINE_STATE_UNKNOWN or #ML_PIPELINE_STATE_NULL.
 * @post The pipeline state will be #ML_PIPELINE_STATE_PAUSED in the same thread.
 */
int ml_pipeline_construct (const char *pipeline_description, ml_pipeline_state_cb cb, void *user_data, ml_pipeline_h *pipe);

/**
 * @brief Destroys the pipeline.
 * @details Use this function to destroy the pipeline constructed with ml_pipeline_construct().
 * @since_tizen 5.5
 * @param[in] pipe The pipeline to be destroyed.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER The parameter is invalid. (Pipeline is not negotiated yet.)
 * @retval #ML_ERROR_STREAMS_PIPE Failed to access the pipeline state.
 *
 * @pre The pipeline state should be #ML_PIPELINE_STATE_PLAYING or #ML_PIPELINE_STATE_PAUSED.
 * @post The pipeline state will be #ML_PIPELINE_STATE_NULL.
 */
int ml_pipeline_destroy (ml_pipeline_h pipe);

/**
 * @brief Gets the state of pipeline.
 * @details Gets the state of the pipeline handle returned by ml_pipeline_construct().
 * @since_tizen 5.5
 * @param[in] pipe The pipeline handle.
 * @param[out] state The pipeline state.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid. (Pipeline is not negotiated yet.)
 * @retval #ML_ERROR_STREAMS_PIPE Failed to get state from the pipeline.
 */
int ml_pipeline_get_state (ml_pipeline_h pipe, ml_pipeline_state_e *state);

/****************************************************
 ** NNStreamer Pipeline Start/Stop Control         **
 ****************************************************/
/**
 * @brief Starts the pipeline, asynchronously.
 * @details The pipeline handle returned by ml_pipeline_construct() is started.
 *          Note that this is asynchronous function. State might be "pending".
 *          If you need to get the changed state, add a callback while constructing a pipeline with ml_pipeline_construct().
 * @since_tizen 5.5
 * @param[in] pipe The pipeline handle.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid. (Pipeline is not negotiated yet.)
 * @retval #ML_ERROR_STREAMS_PIPE Failed to start the pipeline.
 *
 * @pre The pipeline state should be #ML_PIPELINE_STATE_PAUSED.
 * @post The pipeline state will be #ML_PIPELINE_STATE_PLAYING.
 */
int ml_pipeline_start (ml_pipeline_h pipe);

/**
 * @brief Stops the pipeline, asynchronously.
 * @details The pipeline handle returned by ml_pipeline_construct() is stopped.
 *          Note that this is asynchronous function. State might be "pending".
 *          If you need to get the changed state, add a callback while constructing a pipeline with ml_pipeline_construct().
 * @since_tizen 5.5
 * @param[in] pipe The pipeline to be stopped.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid. (Pipeline is not negotiated yet.)
 * @retval #ML_ERROR_STREAMS_PIPE Failed to stop the pipeline.
 *
 * @pre The pipeline state should be #ML_PIPELINE_STATE_PLAYING.
 * @post The pipeline state will be #ML_PIPELINE_STATE_PAUSED.
 */
int ml_pipeline_stop (ml_pipeline_h pipe);

/**
 * @brief Clears all data and resets the pipeline.
 * @details During the flush operation, the pipeline is stopped and after the operation is done, the pipeline is resumed and ready to start the data flow.
 * @since_tizen 6.5
 * @param[in] pipe The pipeline to be flushed.
 * @param[in] start @c true to start the pipeline after the flush operation is done.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #ML_ERROR_STREAMS_PIPE Failed to flush the pipeline.
 */
int ml_pipeline_flush (ml_pipeline_h pipe, bool start);

/****************************************************
 ** NNStreamer Pipeline Sink/Src Control           **
 ****************************************************/
/**
 * @brief Registers a callback for sink node of NNStreamer pipelines.
 * @since_tizen 5.5
 * @remarks If the function succeeds, @a sink_handle handle must be unregistered using ml_pipeline_sink_unregister().
 * @param[in] pipe The pipeline to be attached with a sink node.
 * @param[in] sink_name The name of sink node, described with ml_pipeline_construct().
 * @param[in] cb The function to be called by the sink node.
 * @param[in] user_data Private data for the callback. This value is passed to the callback when it's invoked.
 * @param[out] sink_handle The sink handle.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid. (Not negotiated, @a sink_name is not found, or @a sink_name has an invalid type.)
 * @retval #ML_ERROR_STREAMS_PIPE Failed to connect a signal to sink element.
 * @retval #ML_ERROR_OUT_OF_MEMORY Failed to allocate required memory.
 *
 * @pre The pipeline state should be #ML_PIPELINE_STATE_PAUSED.
 */
int ml_pipeline_sink_register (ml_pipeline_h pipe, const char *sink_name, ml_pipeline_sink_cb cb, void *user_data, ml_pipeline_sink_h *sink_handle);

/**
 * @brief Unregisters a callback for sink node of NNStreamer pipelines.
 * @since_tizen 5.5
 * @param[in] sink_handle The sink handle to be unregistered.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 *
 * @pre The pipeline state should be #ML_PIPELINE_STATE_PAUSED.
 */
int ml_pipeline_sink_unregister (ml_pipeline_sink_h sink_handle);

/**
 * @brief Gets a handle to operate as a src node of NNStreamer pipelines.
 * @since_tizen 5.5
 * @remarks If the function succeeds, @a src_handle handle must be released using ml_pipeline_src_release_handle().
 * @param[in] pipe The pipeline to be attached with a src node.
 * @param[in] src_name The name of src node, described with ml_pipeline_construct().
 * @param[out] src_handle The src handle.
 * @return 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #ML_ERROR_STREAMS_PIPE Failed to get src element.
 * @retval #ML_ERROR_TRY_AGAIN The pipeline is not ready yet.
 * @retval #ML_ERROR_OUT_OF_MEMORY Failed to allocate required memory.
 */
int ml_pipeline_src_get_handle (ml_pipeline_h pipe, const char *src_name, ml_pipeline_src_h *src_handle);

/**
 * @brief Releases the given src handle.
 * @since_tizen 5.5
 * @param[in] src_handle The src handle to be released.
 * @return 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_pipeline_src_release_handle (ml_pipeline_src_h src_handle);

/**
 * @brief Adds an input data frame.
 * @since_tizen 5.5
 * @param[in] src_handle The source handle returned by ml_pipeline_src_get_handle().
 * @param[in] data The handle of input tensors, in the format of tensors info given by ml_pipeline_src_get_tensors_info().
 *                 This function takes ownership of the data if @a policy is #ML_PIPELINE_BUF_POLICY_AUTO_FREE.
 * @param[in] policy The policy of buffer deallocation. The policy value may include buffer deallocation mechanisms or event triggers for appsrc elements. If event triggers are provided, these functions will not give input data to the appsrc element, but will trigger the given event only.
 * @return 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #ML_ERROR_STREAMS_PIPE The pipeline has inconsistent pad caps. (Pipeline is not negotiated yet.)
 * @retval #ML_ERROR_TRY_AGAIN The pipeline is not ready yet.
 */
int ml_pipeline_src_input_data (ml_pipeline_src_h src_handle, ml_tensors_data_h data, ml_pipeline_buf_policy_e policy);

/**
 * @brief Callbacks for src input events.
 * @details A set of callbacks that can be installed on the appsrc with ml_pipeline_src_set_event_cb().
 * @since_tizen 6.5
 */
typedef struct {
  void (*need_data) (ml_pipeline_src_h src_handle, unsigned int length, void *user_data);   /**< Called when the appsrc needs more data. User may submit a buffer via ml_pipeline_src_input_data() from this thread or another thread. length is just a hint and when it is set to -1, any number of bytes can be pushed into appsrc. */
  void (*enough_data) (ml_pipeline_src_h src_handle, void *user_data);                      /**< Called when appsrc has enough data. It is recommended that the application stops calling push-buffer until the need_data callback is emitted again to avoid excessive buffer queueing. */
  void (*seek_data) (ml_pipeline_src_h src_handle, uint64_t offset, void *user_data);       /**< Called when a seek should be performed to the offset. The next push-buffer should produce buffers from the new offset . This callback is only called for seekable stream types. */
} ml_pipeline_src_callbacks_s;

/**
 * @brief Sets the callbacks which will be invoked when a new input frame may be accepted.
 * @details Note that, the last installed callbacks on appsrc are available in the pipeline. If developer sets new callbacks, old callbacks will be replaced with new one.
 * @since_tizen 6.5
 * @param[in] src_handle The source handle returned by ml_pipeline_src_get_handle().
 * @param[in] cb The app-src callbacks for event handling.
 * @param[in] user_data The user's custom data given to callbacks.
 * @return 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #ML_ERROR_OUT_OF_MEMORY Failed to allocate required memory.
 */
int ml_pipeline_src_set_event_cb (ml_pipeline_src_h src_handle, ml_pipeline_src_callbacks_s *cb, void *user_data);

/**
 * @brief Gets a handle for the tensors information of given src node.
 * @details If the media type is not other/tensor or other/tensors, @a info handle may not be correct. If want to use other media types, you MUST set the correct properties.
 * @since_tizen 5.5
 * @remarks If the function succeeds, @a info handle must be released using ml_tensors_info_destroy().
 * @param[in] src_handle The source handle returned by ml_pipeline_src_get_handle().
 * @param[out] info The handle of tensors information.
 * @return 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #ML_ERROR_STREAMS_PIPE The pipeline has inconsistent pad caps. (Pipeline is not negotiated yet.)
 * @retval #ML_ERROR_TRY_AGAIN The pipeline is not ready yet.
 */
int ml_pipeline_src_get_tensors_info (ml_pipeline_src_h src_handle, ml_tensors_info_h *info);

/****************************************************
 ** NNStreamer Pipeline Switch/Valve Control       **
 ****************************************************/

/**
 * @brief Gets a handle to operate a "GstInputSelector"/"GstOutputSelector" node of NNStreamer pipelines.
 * @details Refer to https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-plugins/html/gstreamer-plugins-input-selector.html for input selectors.
 *          Refer to https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-plugins/html/gstreamer-plugins-output-selector.html for output selectors.
 * @since_tizen 5.5
 * @remarks If the function succeeds, @a switch_handle handle must be released using ml_pipeline_switch_release_handle().
 * @param[in] pipe The pipeline to be managed.
 * @param[in] switch_name The name of switch (InputSelector/OutputSelector).
 * @param[out] switch_type The type of the switch. If NULL, it is ignored.
 * @param[out] switch_handle The switch handle.
 * @return 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #ML_ERROR_OUT_OF_MEMORY Failed to allocate required memory.
 */
int ml_pipeline_switch_get_handle (ml_pipeline_h pipe, const char *switch_name, ml_pipeline_switch_e *switch_type, ml_pipeline_switch_h *switch_handle);

/**
 * @brief Releases the given switch handle.
 * @since_tizen 5.5
 * @param[in] switch_handle The handle to be released.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_pipeline_switch_release_handle (ml_pipeline_switch_h switch_handle);

/**
 * @brief Controls the switch with the given handle to select input/output nodes(pads).
 * @since_tizen 5.5
 * @param[in] switch_handle The switch handle returned by ml_pipeline_switch_get_handle().
 * @param[in] pad_name The name of the chosen pad to be activated. Use ml_pipeline_switch_get_pad_list() to list the available pad names.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_pipeline_switch_select (ml_pipeline_switch_h switch_handle, const char *pad_name);

/**
 * @brief Gets the pad names of a switch.
 * @since_tizen 5.5
 * @remarks If the function succeeds, @a list and its contents should be released using g_free(). Refer the below sample code.
 * @param[in] switch_handle The switch handle returned by ml_pipeline_switch_get_handle().
 * @param[out] list NULL terminated array of char*. The caller must free each string (char*) in the list and free the list itself.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #ML_ERROR_STREAMS_PIPE The element is not both input and output switch (Internal data inconsistency).
 * @retval #ML_ERROR_OUT_OF_MEMORY Failed to allocate required memory.
 *
 * Here is an example of the usage:
 * @code
 * int status;
 * gchar *pipeline;
 * ml_pipeline_h handle;
 * ml_pipeline_switch_e switch_type;
 * ml_pipeline_switch_h switch_handle;
 * gchar **node_list = NULL;
 *
 * // pipeline description
 * pipeline = g_strdup ("videotestsrc is-live=true ! videoconvert ! tensor_converter ! output-selector name=outs "
 *     "outs.src_0 ! tensor_sink name=sink0 async=false "
 * 