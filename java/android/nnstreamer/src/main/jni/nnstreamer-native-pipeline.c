
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * NNStreamer Android API
 * Copyright (C) 2019 Samsung Electronics Co., Ltd.
 *
 * @file	nnstreamer-native-pipeline.c
 * @date	10 July 2019
 * @brief	Native code for NNStreamer API
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include "nnstreamer-native.h"

#if defined(__ANDROID__)
#include <android/native_window.h>
#include <android/native_window_jni.h>

/**
 * @brief Macro to release native window.
 */
#define release_native_window(w) do { \
  if (w) { \
    ANativeWindow_release (w); \
    w = NULL; \
  } \
} while (0)
#endif /* __ANDROID__ */

/**
 * @brief Private data for Pipeline class.
 */
typedef struct
{
  jmethodID mid_state_cb;
  jmethodID mid_sink_cb;
} pipeline_priv_data_s;

/**
 * @brief Private data for sink node.
 */
typedef struct
{
  ml_tensors_info_h out_info;
  jobject out_info_obj;
} pipeline_sink_priv_data_s;

#if defined(__ANDROID__)
/**
 * @brief Private data for video sink.
 */
typedef struct
{
  ANativeWindow *window;
  ANativeWindow *old_window;
} pipeline_video_sink_priv_data_s;

/**
 * @brief Release private data in video sink.
 */
static void
nns_pipeline_video_sink_priv_free (gpointer data, JNIEnv * env)
{
  pipeline_video_sink_priv_data_s *priv;

  priv = (pipeline_video_sink_priv_data_s *) data;
  if (priv) {
    release_native_window (priv->old_window);
    release_native_window (priv->window);

    g_free (priv);
  }
}
#endif /* __ANDROID__ */

/**
 * @brief Release private data in pipeline info.
 */
static void
nns_pipeline_priv_free (gpointer data, JNIEnv * env)
{
  pipeline_priv_data_s *priv = (pipeline_priv_data_s *) data;

  /* nothing to free */
  g_free (priv);
}

/**
 * @brief Release private data in sink node.
 */
static void
nns_pipeline_sink_priv_free (gpointer data, JNIEnv * env)
{
  pipeline_sink_priv_data_s *priv = (pipeline_sink_priv_data_s *) data;

  ml_tensors_info_destroy (priv->out_info);
  if (priv->out_info_obj)
    (*env)->DeleteGlobalRef (env, priv->out_info_obj);

  g_free (priv);
}

/**
 * @brief Update output info in sink node data.
 */
static gboolean
nns_pipeline_sink_priv_set_out_info (element_data_s * item, JNIEnv * env,
    const ml_tensors_info_h out_info)
{
  pipeline_sink_priv_data_s *priv;
  jobject obj_info = NULL;

  if ((priv = item->priv_data) == NULL) {
    priv = g_new0 (pipeline_sink_priv_data_s, 1);
    ml_tensors_info_create (&priv->out_info);

    item->priv_data = priv;
    item->priv_destroy_func = nns_pipeline_sink_priv_free;
  }

  if (ml_tensors_info_is_equal (out_info, priv->out_info)) {
    /* do nothing, tensors info is equal. */
    return TRUE;
  }

  if (!nns_convert_tensors_info (item->pipe_info, env, out_info, &obj_info)) {
    _ml_loge ("Failed to convert output info.");
    return FALSE;
  }

  _ml_tensors_info_free (priv->out_info);
  ml_tensors_info_clone (priv->out_info, out_info);

  if (priv->out_info_obj)
    (*env)->DeleteGlobalRef (env, priv->out_info_obj);
  priv->out_info_obj = (*env)->NewGlobalRef (env, obj_info);
  (*env)->DeleteLocalRef (env, obj_info);
  return TRUE;
}

/**
 * @brief Pipeline state change callback.
 */
static void
nns_pipeline_state_cb (ml_pipeline_state_e state, void *user_data)
{
  pipeline_info_s *pipe_info;
  pipeline_priv_data_s *priv;
  jint new_state = (jint) state;
  JNIEnv *env;

  pipe_info = (pipeline_info_s *) user_data;
  priv = (pipeline_priv_data_s *) pipe_info->priv_data;

  if ((env = nns_get_jni_env (pipe_info)) == NULL) {
    _ml_logw ("Cannot get jni env in the state callback.");
    return;
  }

  (*env)->CallVoidMethod (env, pipe_info->instance, priv->mid_state_cb,
      new_state);

  if ((*env)->ExceptionCheck (env)) {
    _ml_loge ("Failed to call the state-change callback method.");
    (*env)->ExceptionClear (env);
  }
}

/**
 * @brief New data callback for sink node.
 */
static void
nns_sink_data_cb (const ml_tensors_data_h data, const ml_tensors_info_h info,
    void *user_data)
{
  element_data_s *item;
  pipeline_info_s *pipe_info;
  pipeline_priv_data_s *priv;
  pipeline_sink_priv_data_s *priv_sink;
  jobject obj_data = NULL;
  JNIEnv *env;

  item = (element_data_s *) user_data;
  pipe_info = item->pipe_info;

  if ((env = nns_get_jni_env (pipe_info)) == NULL) {
    _ml_logw ("Cannot get jni env in the sink callback.");
    return;
  }

  /* cache output tensors info */
  if (!nns_pipeline_sink_priv_set_out_info (item, env, info)) {
    return;
  }

  priv = (pipeline_priv_data_s *) pipe_info->priv_data;
  priv_sink = (pipeline_sink_priv_data_s *) item->priv_data;

  if (nns_convert_tensors_data (pipe_info, env, data, priv_sink->out_info_obj,
          &obj_data)) {
    jstring sink_name = (*env)->NewStringUTF (env, item->name);

    (*env)->CallVoidMethod (env, pipe_info->instance, priv->mid_sink_cb,
        sink_name, obj_data);

    if ((*env)->ExceptionCheck (env)) {
      _ml_loge ("Failed to call the new-data callback method.");
      (*env)->ExceptionClear (env);
    }

    (*env)->DeleteLocalRef (env, sink_name);
    (*env)->DeleteLocalRef (env, obj_data);
  } else {
    _ml_loge ("Failed to convert the result to data object.");
  }
}

/**
 * @brief Get sink handle.
 */
static void *
nns_get_sink_handle (pipeline_info_s * pipe_info, const gchar * element_name)
{
  const nns_element_type_e etype = NNS_ELEMENT_TYPE_SINK;
  ml_pipeline_sink_h handle;
  ml_pipeline_h pipe;
  element_data_s *item;
  int status;

  g_assert (pipe_info);
  pipe = pipe_info->pipeline_handle;

  handle = (ml_pipeline_sink_h) nns_get_element_handle (pipe_info,
      element_name, etype);
  if (handle == NULL) {
    /* get sink handle and register to table */
    item = g_new0 (element_data_s, 1);
    if (item == NULL) {
      _ml_loge ("Failed to allocate memory for sink handle data.");
      return NULL;
    }

    status = ml_pipeline_sink_register (pipe, element_name, nns_sink_data_cb,
        item, &handle);
    if (status != ML_ERROR_NONE) {
      _ml_loge ("Failed to get sink node %s.", element_name);
      g_free (item);
      return NULL;
    }

    item->name = g_strdup (element_name);
    item->type = etype;
    item->handle = handle;
    item->pipe_info = pipe_info;

    if (!nns_add_element_data (pipe_info, element_name, item)) {
      _ml_loge ("Failed to add sink node %s.", element_name);
      nns_free_element_data (item);
      return NULL;
    }
  }

  return handle;
}

/**
 * @brief Get src handle.
 */
static void *
nns_get_src_handle (pipeline_info_s * pipe_info, const gchar * element_name)
{
  const nns_element_type_e etype = NNS_ELEMENT_TYPE_SRC;
  ml_pipeline_src_h handle;
  ml_pipeline_h pipe;
  element_data_s *item;
  int status;

  g_assert (pipe_info);
  pipe = pipe_info->pipeline_handle;

  handle = (ml_pipeline_src_h) nns_get_element_handle (pipe_info,
      element_name, etype);
  if (handle == NULL) {
    /* get src handle and register to table */
    status = ml_pipeline_src_get_handle (pipe, element_name, &handle);
    if (status != ML_ERROR_NONE) {
      _ml_loge ("Failed to get src node %s.", element_name);
      return NULL;
    }

    item = g_new0 (element_data_s, 1);
    if (item == NULL) {
      _ml_loge ("Failed to allocate memory for src handle data.");
      ml_pipeline_src_release_handle (handle);
      return NULL;
    }

    item->name = g_strdup (element_name);
    item->type = etype;
    item->handle = handle;
    item->pipe_info = pipe_info;

    if (!nns_add_element_data (pipe_info, element_name, item)) {
      _ml_loge ("Failed to add src node %s.", element_name);
      nns_free_element_data (item);
      return NULL;
    }
  }

  return handle;
}

/**
 * @brief Get switch handle.
 */
static void *
nns_get_switch_handle (pipeline_info_s * pipe_info, const gchar * element_name)
{
  const nns_element_type_e etype = NNS_ELEMENT_TYPE_SWITCH;
  ml_pipeline_switch_h handle;
  ml_pipeline_switch_e switch_type;
  ml_pipeline_h pipe;
  element_data_s *item;
  int status;

  g_assert (pipe_info);
  pipe = pipe_info->pipeline_handle;

  handle = (ml_pipeline_switch_h) nns_get_element_handle (pipe_info,
      element_name, etype);
  if (handle == NULL) {
    /* get switch handle and register to table */
    status = ml_pipeline_switch_get_handle (pipe, element_name, &switch_type,
        &handle);
    if (status != ML_ERROR_NONE) {
      _ml_loge ("Failed to get switch %s.", element_name);
      return NULL;
    }

    item = g_new0 (element_data_s, 1);
    if (item == NULL) {
      _ml_loge ("Failed to allocate memory for switch handle data.");
      ml_pipeline_switch_release_handle (handle);
      return NULL;
    }

    item->name = g_strdup (element_name);
    item->type = etype;
    item->handle = handle;
    item->pipe_info = pipe_info;

    if (!nns_add_element_data (pipe_info, element_name, item)) {
      _ml_loge ("Failed to add switch %s.", element_name);
      nns_free_element_data (item);
      return NULL;
    }
  }

  return handle;
}

/**
 * @brief Get valve handle.
 */
static void *
nns_get_valve_handle (pipeline_info_s * pipe_info, const gchar * element_name)
{
  const nns_element_type_e etype = NNS_ELEMENT_TYPE_VALVE;
  ml_pipeline_valve_h handle;
  ml_pipeline_h pipe;
  element_data_s *item;
  int status;

  g_assert (pipe_info);
  pipe = pipe_info->pipeline_handle;

  handle = (ml_pipeline_valve_h) nns_get_element_handle (pipe_info,
      element_name, etype);
  if (handle == NULL) {
    /* get valve handle and register to table */
    status = ml_pipeline_valve_get_handle (pipe, element_name, &handle);
    if (status != ML_ERROR_NONE) {
      _ml_loge ("Failed to get valve %s.", element_name);
      return NULL;
    }

    item = g_new0 (element_data_s, 1);
    if (item == NULL) {
      _ml_loge ("Failed to allocate memory for valve handle data.");
      ml_pipeline_valve_release_handle (handle);
      return NULL;
    }

    item->name = g_strdup (element_name);
    item->type = etype;
    item->handle = handle;
    item->pipe_info = pipe_info;

    if (!nns_add_element_data (pipe_info, element_name, item)) {
      _ml_loge ("Failed to add valve %s.", element_name);
      nns_free_element_data (item);
      return NULL;
    }
  }

  return handle;
}

#if defined(__ANDROID__)
/**
 * @brief Get video sink element data in the pipeline.
 */
static element_data_s *
nns_get_video_sink_data (pipeline_info_s * pipe_info,
    const gchar * element_name)
{
  const nns_element_type_e etype = NNS_ELEMENT_TYPE_VIDEO_SINK;
  ml_pipeline_h pipe;
  element_data_s *item;
  int status;

  g_assert (pipe_info);