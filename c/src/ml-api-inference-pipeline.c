
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file ml-api-inference-pipeline.c
 * @date 11 March 2019
 * @brief NNStreamer/Pipeline(main) C-API Wrapper.
 *        This allows to construct and control NNStreamer pipelines.
 * @see	https://github.com/nnstreamer/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug Thread safety for ml_tensors_data should be addressed.
 */

#include <string.h>
#include <glib.h>
#include <gst/gstbuffer.h>
#include <gst/app/app.h>        /* To push data to pipeline */
#include <nnstreamer_plugin_api.h>
#include <tensor_if.h>
#include <tensor_typedef.h>
#include <tensor_filter_custom_easy.h>

#include <nnstreamer.h>
#include <nnstreamer-tizen-internal.h>

#include "ml-api-internal.h"
#include "ml-api-inference-internal.h"
#include "ml-api-inference-pipeline-internal.h"


#define handle_init(name, h) \
  ml_pipeline_common_elem *name= (h); \
  ml_pipeline *p; \
  ml_pipeline_element *elem; \
  int ret = ML_ERROR_NONE; \
  check_feature_state (ML_FEATURE_INFERENCE); \
  if ((h) == NULL) { \
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER, \
        "The parameter, %s, (handle) is invalid (NULL). Please provide a valid handle.", \
        #h); \
  } \
  p = name->pipe; \
  elem = name->element; \
  if (p == NULL) \
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER, \
        "Internal error. The contents of parameter, %s, (handle), is invalid. The pipeline entry (%s->pipe) is NULL. The handle (%s) is either not properly created or application threads may have touched its contents.", \
        #h, #h, #h); \
  if (elem == NULL) \
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER, \
        "Internal error. The contents of parameter, %s, (handle), is invalid. The element entry (%s->element) is NULL. The handle (%s) is either not properly created or application threads may have touched its contents.", \
        #h, #h, #h); \
  if (elem->pipe == NULL) \
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER, \
        "Internal error. The contents of parameter, %s, (handle), is invalid. The pipeline entry of the element entry (%s->element->pipe) is NULL. The handle (%s) is either not properly created or application threads may have touched its contents.", \
        #h, #h, #h); \
  g_mutex_lock (&p->lock); \
  g_mutex_lock (&elem->lock); \
  if (NULL == g_list_find (elem->handles, name)) { \
    _ml_error_report \
        ("Internal error. The handle name, %s, does not exists in the list of %s->element->handles.", \
        #h, #h); \
    ret = ML_ERROR_INVALID_PARAMETER; \
    goto unlock_return; \
  }

#define handle_exit(h) \
unlock_return: \
  g_mutex_unlock (&elem->lock); \
  g_mutex_unlock (&p->lock); \
  return ret;

/**
 * @brief The enumeration for custom data type.
 */
typedef enum
{
  PIPE_CUSTOM_TYPE_NONE,
  PIPE_CUSTOM_TYPE_IF,
  PIPE_CUSTOM_TYPE_FILTER,

  PIPE_CUSTOM_TYPE_MAX
} pipe_custom_type_e;

/**
 * @brief The struct for custom data.
 */
typedef struct
{
  pipe_custom_type_e type;
  gchar *name;
  gpointer handle;
} pipe_custom_data_s;

static void ml_pipeline_custom_filter_ref (ml_custom_easy_filter_h custom);
static void ml_pipeline_custom_filter_unref (ml_custom_easy_filter_h custom);
static void ml_pipeline_if_custom_ref (ml_pipeline_if_h custom);
static void ml_pipeline_if_custom_unref (ml_pipeline_if_h custom);

/**
 * @brief Global lock for pipeline functions.
 */
G_LOCK_DEFINE_STATIC (g_ml_pipe_lock);

/**
 * @brief The list of custom data. This should be managed with lock.
 */
static GList *g_ml_custom_data = NULL;

/**
 * @brief Finds a position of custom data in the list.
 * @note This function should be called with lock.
 */
static GList *
pipe_custom_find_link (const pipe_custom_type_e type, const gchar * name)
{
  pipe_custom_data_s *data;
  GList *link;

  g_return_val_if_fail (name != NULL, NULL);

  link = g_ml_custom_data;
  while (link) {
    data = (pipe_custom_data_s *) link->data;

    if (data->type == type && g_str_equal (data->name, name))
      break;

    link = link->next;
  }

  return link;
}

/**
 * @brief Finds custom data matched with data type and name.
 */
static pipe_custom_data_s *
pipe_custom_find_data (const pipe_custom_type_e type, const gchar * name)
{
  pipe_custom_data_s *data;
  GList *link;

  G_LOCK (g_ml_pipe_lock);

  link = pipe_custom_find_link (type, name);
  data = (link != NULL) ? (pipe_custom_data_s *) link->data : NULL;

  G_UNLOCK (g_ml_pipe_lock);
  return data;
}

/**
 * @brief Adds new custom data into the list.
 */
static void
pipe_custom_add_data (const pipe_custom_type_e type, const gchar * name,
    gpointer handle)
{
  pipe_custom_data_s *data;

  data = g_new0 (pipe_custom_data_s, 1);
  data->type = type;
  data->name = g_strdup (name);
  data->handle = handle;

  G_LOCK (g_ml_pipe_lock);
  g_ml_custom_data = g_list_prepend (g_ml_custom_data, data);
  G_UNLOCK (g_ml_pipe_lock);
}

/**
 * @brief Removes custom data from the list.
 */
static void
pipe_custom_remove_data (const pipe_custom_type_e type, const gchar * name)
{
  pipe_custom_data_s *data;
  GList *link;

  G_LOCK (g_ml_pipe_lock);

  link = pipe_custom_find_link (type, name);
  if (link) {
    data = (pipe_custom_data_s *) link->data;

    g_ml_custom_data = g_list_delete_link (g_ml_custom_data, link);

    g_free (data->name);
    g_free (data);
  }

  G_UNLOCK (g_ml_pipe_lock);
}

/**
 * @brief The callback function called when the element node with custom data is released.
 */
static int
pipe_custom_destroy_cb (void *handle, void *user_data)
{
  pipe_custom_data_s *custom_data;

  custom_data = (pipe_custom_data_s *) handle;
  if (custom_data == NULL)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, handle, is NULL. It should be a valid internal object. This is possibly a bug in ml-api-inference-pipeline.c along with tensor-if or tensor-filter/custom function. Please report to https://github.com/nnstreamer/nnstreamer/issues");

  switch (custom_data->type) {
    case PIPE_CUSTOM_TYPE_IF:
      ml_pipeline_if_custom_unref (custom_data->handle);
      break;
    case PIPE_CUSTOM_TYPE_FILTER:
      ml_pipeline_custom_filter_unref (custom_data->handle);
      break;
    default:
      break;
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Internal function to create a referable element in a pipeline
 */
static ml_pipeline_element *
construct_element (GstElement * e, ml_pipeline * p, const char *name,
    ml_pipeline_element_e t)
{
  ml_pipeline_element *ret = g_new0 (ml_pipeline_element, 1);

  if (ret == NULL)
    _ml_error_report_return (NULL,
        "Failed to allocate memory for the pipeline.");

  ret->element = e;
  ret->pipe = p;
  ret->name = g_strdup (name);
  ret->type = t;
  ret->handles = NULL;
  ret->src = NULL;
  ret->sink = NULL;
  _ml_tensors_info_initialize (&ret->tensors_info);
  ret->size = 0;
  ret->maxid = 0;
  ret->handle_id = 0;
  ret->is_media_stream = FALSE;
  ret->is_flexible_tensor = FALSE;
  g_mutex_init (&ret->lock);
  return ret;
}

/**
 * @brief Internal function to get the tensors info from the element caps.
 */
static gboolean
get_tensors_info_from_caps (GstCaps * caps, ml_tensors_info_s * info,
    gboolean * is_flexible)
{
  GstStructure *s;
  GstTensorsConfig config;
  guint i, n_caps;
  gboolean found = FALSE;

  _ml_tensors_info_initialize (info);
  n_caps = gst_caps_get_size (caps);

  for (i = 0; i < n_caps; i++) {
    s = gst_caps_get_structure (caps, i);
    found = gst_tensors_config_from_structure (&config, s);

    if (found) {
      _ml_tensors_info_copy_from_gst (info, &config.info);
      *is_flexible = gst_tensors_config_is_flexible (&config);
      break;
    }
  }

  return found;
}

/**
 * @brief Handle a sink element for registered ml_pipeline_sink_cb
 */
static void
cb_sink_event (GstElement * e, GstBuffer * b, gpointer user_data)
{
  ml_pipeline_element *elem = user_data;

  /** @todo CRITICAL if the pipeline is being killed, don't proceed! */
  GstMemory *mem[ML_TENSOR_SIZE_LIMIT];
  GstMapInfo map[ML_TENSOR_SIZE_LIMIT];
  guint i;
  guint num_mems;
  GList *l;
  ml_tensors_data_s *_data = NULL;
  ml_tensors_info_s *_info;
  ml_tensors_info_s info_flex_tensor;
  size_t total_size = 0;
  int status;

  _info = &elem->tensors_info;
  num_mems = gst_buffer_n_memory (b);

  if (num_mems > ML_TENSOR_SIZE_LIMIT) {
    _ml_loge (_ml_detail
        ("Number of memory chunks in a GstBuffer exceed the limit: %u > %u. Please check the version or variants of GStreamer you use. If you have modified the maximum number of memory chunks of a GST-Buffer, this might happen. Please update nnstreamer and ml-api code to make them consistent with your modification of GStreamer.",
            num_mems, ML_TENSOR_SIZE_LIMIT));
    return;
  }

  /* set tensor data */
  status =
      _ml_tensors_data_create_no_alloc (NULL, (ml_tensors_data_h *) & _data);
  if (status != ML_ERROR_NONE) {
    _ml_loge (_ml_detail
        ("Failed to allocate memory for tensors data in sink callback, which is registered by ml_pipeline_sink_register ()."));
    return;
  }

  g_mutex_lock (&elem->lock);

  _data->num_tensors = num_mems;
  for (i = 0; i < num_mems; i++) {
    mem[i] = gst_buffer_peek_memory (b, i);
    if (!gst_memory_map (mem[i], &map[i], GST_MAP_READ)) {
      _ml_loge (_ml_detail
          ("Failed to map the output in sink '%s' callback, which is registered by ml_pipeline_sink_register ()",
              elem->name));
      num_mems = i;
      goto error;
    }

    _data->tensors[i].tensor = map[i].data;
    _data->tensors[i].size = map[i].size;

    total_size += map[i].size;
  }

  /** @todo This assumes that padcap is static */
  if (elem->sink == NULL) {
    /* Get the sink-pad-cap */
    elem->sink = gst_element_get_static_pad (elem->element, "sink");

    if (elem->sink) {
      /* sinkpadcap available (negotiated) */
      GstCaps *caps = gst_pad_get_current_caps (elem->sink);

      if (caps) {
        gboolean flexible = FALSE;
        gboolean found = get_tensors_info_from_caps (caps, _info, &flexible);

        gst_caps_unref (caps);

        if (found) {
          elem->size = 0;

          /* cannot get exact info from caps */
          if (flexible) {
            elem->is_flexible_tensor = TRUE;
            goto send_cb;
          }

          if (_info->num_tensors != num_mems) {
            _ml_loge (_ml_detail
                ("The sink event of [%s] cannot be handled because the number of tensors mismatches.",
                    elem->name));

            gst_object_unref (elem->sink);
            elem->sink = NULL;
            goto error;
          }

          for (i = 0; i < _info->num_tensors; i++) {
            size_t sz =
                _ml_tensor_info_get_size (&_info->info[i], _info->is_extended);

            /* Not configured, yet. */
            if (sz == 0)
              _ml_loge (_ml_detail
                  ("The caps for sink(%s) is not configured.", elem->name));

            if (sz != _data->tensors[i].size) {
              _ml_loge (_ml_detail
                  ("The sink event of [%s] cannot be handled because the tensor dimension mismatches.",
                      elem->name));

              gst_object_unref (elem->sink);
              elem->sink = NULL;
              goto error;
            }

            elem->size += sz;
          }
        } else {
          gst_object_unref (elem->sink);
          elem->sink = NULL;    /* It is not valid */
          goto error;
          /** @todo What if it keeps being "NULL"? Exception handling at 2nd frame? */
        }
      }
    }
  }

  /* Get the data! */
  if (gst_buffer_get_size (b) != total_size ||
      (elem->size > 0 && total_size != elem->size)) {
    _ml_loge (_ml_detail
        ("The buffersize mismatches. All the three values must be the same: %zu, %zu, %zu",
            total_size, elem->size, gst_buffer_get_size (b)));
    goto error;
  }

send_cb:
  /* set info for flexible stream */
  if (elem->is_flexible_tensor) {
    GstTensorMetaInfo meta;
    GstTensorsInfo gst_info;
    gsize hsize;

    gst_tensors_info_init (&gst_info);
    gst_info.num_tensors = num_mems;
    _info = &info_flex_tensor;

    /* handle header for flex tensor */
    for (i = 0; i < num_mems; i++) {
      gst_tensor_meta_info_parse_header (&meta, map[i].data);
      hsize = gst_tensor_meta_info_get_header_size (&meta);

      gst_tensor_meta_info_convert (&meta, &gst_info.info[i]);

      _data->tensors[i].tensor = map[i].data + hsize;
      _data->tensors[i].size = map[i].size - hsize;
    }

    _ml_tensors_info_copy_from_gst (_info, &gst_info);
  }

  /* Iterate e->handles, pass the data to them */
  for (l = elem->handles; l != NULL; l = l->next) {
    ml_pipeline_sink_cb callback;
    ml_pipeline_common_elem *sink = l->data;
    if (sink->callback_info == NULL)
      continue;

    callback = sink->callback_info->sink_cb;
    if (callback)
      callback (_data, _info, sink->callback_info->pdata);

    /** @todo Measure time. Warn if it takes long. Kill if it takes too long. */
  }

error:
  g_mutex_unlock (&elem->lock);

  for (i = 0; i < num_mems; i++) {
    gst_memory_unmap (mem[i], &map[i]);
  }

  _ml_tensors_data_destroy_internal (_data, FALSE);
  _data = NULL;

  return;
}

/**
 * @brief Handle a appsink element for registered ml_pipeline_sink_cb
 */
static GstFlowReturn
cb_appsink_new_sample (GstElement * e, gpointer user_data)
{
  GstSample *sample;
  GstBuffer *buffer;

  /* get the sample from appsink */
  sample = gst_app_sink_pull_sample (GST_APP_SINK (e));
  buffer = gst_sample_get_buffer (sample);

  cb_sink_event (e, buffer, user_data);

  gst_sample_unref (sample);
  return GST_FLOW_OK;
}

/**
 * @brief Callback for bus message.
 */
static void
cb_bus_sync_message (GstBus * bus, GstMessage * message, gpointer user_data)
{
  ml_pipeline *pipe_h;

  pipe_h = (ml_pipeline *) user_data;

  if (pipe_h == NULL)
    return;

  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_EOS:
      pipe_h->isEOS = TRUE;
      break;
    case GST_MESSAGE_STATE_CHANGED:
      if (GST_MESSAGE_SRC (message) == GST_OBJECT_CAST (pipe_h->element)) {
        GstState old_state, new_state;

        gst_message_parse_state_changed (message, &old_state, &new_state, NULL);
        pipe_h->pipe_state = (ml_pipeline_state_e) new_state;

        _ml_logd (_ml_detail ("The pipeline state changed from %s to %s.",
                gst_element_state_get_name (old_state),
                gst_element_state_get_name (new_state)));

        if (pipe_h->state_cb.cb) {
          pipe_h->state_cb.cb (pipe_h->pipe_state, pipe_h->state_cb.user_data);
        }
      }
      break;
    default:
      break;
  }
}

/**
 * @brief Clean up each element of the pipeline.
 */
static void
free_element_handle (gpointer data)
{
  ml_pipeline_common_elem *item = (ml_pipeline_common_elem *) data;
  ml_pipeline_element *elem;

  if (!(item && item->callback_info)) {
    g_free (item);
    return;
  }

  /* clear callbacks */
  item->callback_info->sink_cb = NULL;
  elem = item->element;
  if (elem->type == ML_PIPELINE_ELEMENT_APP_SRC) {
    GstAppSrcCallbacks appsrc_cb = { 0, };
    gst_app_src_set_callbacks (GST_APP_SRC (elem->element), &appsrc_cb,
        NULL, NULL);
  }

  g_free (item->callback_info);
  item->callback_info = NULL;
  g_free (item);
}

/**
 * @brief Private function for ml_pipeline_destroy, cleaning up nodes in namednodes
 */
static void
cleanup_node (gpointer data)
{
  ml_pipeline_element *e = data;

  g_mutex_lock (&e->lock);
  /** @todo CRITICAL. Stop the handle callbacks if they are running/ready */
  if (e->handle_id > 0) {
    g_signal_handler_disconnect (e->element, e->handle_id);
    e->handle_id = 0;
  }

  /* clear all handles first */
  if (e->handles)
    g_list_free_full (e->handles, free_element_handle);
  e->handles = NULL;

  if (e->type == ML_PIPELINE_ELEMENT_APP_SRC && !e->pipe->isEOS) {
    int eos_check_cnt = 0;

    /** to push EOS event, the pipeline should be in PAUSED state */
    gst_element_set_state (e->pipe->element, GST_STATE_PAUSED);

    if (gst_app_src_end_of_stream (GST_APP_SRC (e->element)) != GST_FLOW_OK) {
      _ml_logw (_ml_detail
          ("Cleaning up a pipeline has failed to set End-Of-Stream for the pipeline element of %s",
              e->name));
    }
    g_mutex_unlock (&e->lock);
    while (!e->pipe->isEOS) {
      eos_check_cnt++;
      /** check EOS every 1ms */
      g_usleep (1000);
      if (eos_check_cnt >= EOS_MESSAGE_TIME_LIMIT) {
        _ml_loge (_ml_detail
            ("Cleaning up a pipeline has requested to set End-Of-Stream. However, the pipeline has not become EOS after the timeout. It has failed to become EOS with the element of %s.",
                e->name));
        break;
      }
    }
    g_mutex_lock (&e->lock);
  }

  if (e->custom_destroy) {
    e->custom_destroy (e->custom_data, e);
  }

  g_free (e->name);
  if (e->src)
    gst_object_unref (e->src);
  if (e->sink)
    gst_object_unref (e->sink);

  _ml_tensors_info_free (&e->tensors_info);

  g_mutex_unlock (&e->lock);
  g_mutex_clear (&e->lock);

  g_free (e);
}

/**
 * @brief Private function to release the pipeline resources
 */
static void
cleanup_resource (gpointer data)
{
  pipeline_resource_s *res = data;

  /* check resource type and free data */
  if (g_str_has_prefix (res->type, "tizen")) {
    release_tizen_resource (res->handle, res->type);
  }

  g_free (res->type);
  g_free (res);
}

/**
 * @brief Converts predefined element in pipeline description.
 */
static int
convert_element (ml_pipeline_h pipe, const gchar * description, gchar ** result,
    gboolean is_internal)
{
  gchar *converted;
  int status = ML_ERROR_NONE;

  g_return_val_if_fail (pipe, ML_ERROR_INVALID_PARAMETER);
  g_return_val_if_fail (description && result, ML_ERROR_INVALID_PARAMETER);

  /* init null */
  *result = NULL;

  converted = g_strdup (description);

  /* convert pre-defined element for Tizen */
  status = convert_tizen_element (pipe, &converted, is_internal);

  if (status == ML_ERROR_NONE) {
    _ml_logd (_ml_detail
        ("Pipeline element converted with aliases for gstreamer (Tizen element aliases): %s",
            converted));
    *result = converted;
  } else {
    g_free (converted);
    _ml_error_report_continue
        ("Failed to convert element: convert_tizen_element() returned %d",
        status);
  }

  return status;
}

/**
 * @brief Handle tensor-filter options.
 */
static void
process_tensor_filter_option (ml_pipeline_element * e)
{
  gchar *fw = NULL;
  gchar *model = NULL;
  pipe_custom_data_s *custom_data;

  g_object_get (G_OBJECT (e->element), "framework", &fw, "model", &model, NULL);

  if (fw && g_ascii_strcasecmp (fw, "custom-easy") == 0) {
    /* ref to tensor-filter custom-easy handle. */
    custom_data = pipe_custom_find_data (PIPE_CUSTOM_TYPE_FILTER, model);
    if (custom_data) {
      ml_pipeline_custom_filter_ref (custom_data->handle);

      e->custom_destroy = pipe_custom_destroy_cb;
      e->custom_data = custom_data;
    }
  }

  g_free (fw);
  g_free (model);
}

/**
 * @brief Handle tensor-if options.
 */
static void
process_tensor_if_option (ml_pipeline_element * e)
{
  gint cv = 0;
  gchar *cv_option = NULL;
  pipe_custom_data_s *custom_data;

  g_object_get (G_OBJECT (e->element), "compared-value", &cv,
      "compared-value-option", &cv_option, NULL);

  if (cv == 5) {
    /* cv is TIFCV_CUSTOM, ref to tensor-if custom handle. */
    custom_data = pipe_custom_find_data (PIPE_CUSTOM_TYPE_IF, cv_option);
    if (custom_data) {
      ml_pipeline_if_custom_ref (custom_data->handle);

      e->custom_destroy = pipe_custom_destroy_cb;
      e->custom_data = custom_data;
    }
  }

  g_free (cv_option);
}

/**
 * @brief Initializes the GStreamer library. This is internal function.
 */
int
_ml_initialize_gstreamer (void)
{
  GError *err = NULL;

  if (!gst_init_check (NULL, NULL, &err)) {
    if (err) {
      _ml_error_report
          ("Initrializing ML-API failed: GStreamer has the following error from gst_init_check(): %s",
          err->message);
      g_clear_error (&err);
    } else {
      _ml_error_report ("Cannot initialize GStreamer. Unknown reason.");
    }

    return ML_ERROR_STREAMS_PIPE;
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Checks the element is registered and available on the pipeline.
 */
int
ml_check_element_availability (const char *element_name, bool *available)
{
  GstElementFactory *factory;
  int status;

  check_feature_state (ML_FEATURE_INFERENCE);

  if (!element_name)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, element_name, is NULL. It should be a name (string) to be queried if it exists as a GStreamer/NNStreamer element.");

  if (!available)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, available, is NULL. It should be a valid pointer to a bool entry so that the API (ml_check_element_availability) may return the queried result via \"available\" parameter. E.g., bool available; ml_check_element_availability (\"tensor_converter\", &available);");

  _ml_error_report_return_continue_iferr (_ml_initialize_gstreamer (),
      "Internal error of _ml_initialize_gstreamer(). Check the availability of gstreamer libraries in your system.");

  /* init false */
  *available = false;

  factory = gst_element_factory_find (element_name);
  if (factory) {
    GstPluginFeature *feature = GST_PLUGIN_FEATURE (factory);
    const gchar *plugin_name = gst_plugin_feature_get_plugin_name (feature);

    /* check restricted element */
    status = _ml_check_plugin_availability (plugin_name, element_name);
    if (status == ML_ERROR_NONE)
      *available = true;

    gst_object_unref (factory);
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Checks the availability of the plugin.
 */
int
_ml_check_plugin_availability (const char *plugin_name,
    const char *element_name)
{
  static gboolean list_loaded = FALSE;
  static gchar **allowed_elements = NULL;

  if (!plugin_name)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, plugin_name, is NULL. It should be a valid string.");

  if (!element_name)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, element_name, is NULL. It should be a valid string.");

  if (!list_loaded) {
    gboolean restricted;

    restricted =
        nnsconf_get_custom_value_bool ("element-restriction",
        "enable_element_restriction", FALSE);
    if (restricted) {
      gchar *elements;

      /* check white-list of available plugins */
      elements =
          nnsconf_get_custom_value_string ("element-restriction",
          "allowed_elements");
      if (elements) {
        allowed_elements = g_strsplit_set (elements, " ,;", -1);
        g_free (elements);
      }
    }

    list_loaded = TRUE;
  }

  /* nnstreamer elements */
  if (g_str_has_prefix (plugin_name, "nnstreamer") &&
      g_str_has_prefix (element_name, "tensor_")) {
    return ML_ERROR_NONE;
  }

  if (allowed_elements &&
      find_key_strv ((const gchar **) allowed_elements, element_name) < 0) {
    _ml_error_report_return (ML_ERROR_NOT_SUPPORTED,
        "The element %s is restricted.", element_name);
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Get the ml_pipeline_element_e type from its element name
 */
static ml_pipeline_element_e
get_elem_type_from_name (GHashTable * table, const gchar * name)
{
  gpointer value = g_hash_table_lookup (table, name);
  if (!value)
    return ML_PIPELINE_ELEMENT_UNKNOWN;

  return GPOINTER_TO_INT (value);
}

/**
 * @brief Iterate elements and prepare element handle.
 */
static int
iterate_element (ml_pipeline * pipe_h, GstElement * pipeline,
    gboolean is_internal)
{
  GstIterator *it = NULL;
  int status = ML_ERROR_NONE;

  g_return_val_if_fail (pipe_h && pipeline, ML_ERROR_INVALID_PARAMETER);

  g_mutex_lock (&pipe_h->lock);

  it = gst_bin_iterate_elements (GST_BIN (pipeline));
  if (it != NULL) {
    gboolean done = FALSE;
    GValue item = G_VALUE_INIT;
    GObject *obj;
    gchar *name;

    /* Fill in the hashtable, "namednodes" with named Elements */
    while (!done) {
      switch (gst_iterator_next (it, &item)) {
        case GST_ITERATOR_OK:
          obj = g_value_get_object (&item);

          if (GST_IS_ELEMENT (obj)) {
            GstElement *elem = GST_ELEMENT (obj);
            GstPluginFeature *feature =
                GST_PLUGIN_FEATURE (gst_element_get_factory (elem));
            const gchar *plugin_name =
                gst_plugin_feature_get_plugin_name (feature);
            const gchar *element_name = gst_plugin_feature_get_name (feature);

            /* validate the availability of the plugin */
            if (!is_internal && _ml_check_plugin_availability (plugin_name,
                    element_name) != ML_ERROR_NONE) {
              _ml_error_report_continue
                  ("There is a pipeline element (filter) that is not allowed for applications via ML-API (privilege not granted) or now available: '%s'/'%s'.",
                  plugin_name, element_name);
              status = ML_ERROR_NOT_SUPPORTED;
              done = TRUE;
              break;
            }

            name = gst_element_get_name (elem);
            if (name != NULL) {
              ml_pipeline_element_e element_type =
                  get_elem_type_from_name (pipe_h->pipe_elm_type, element_name);

              /* check 'sync' property in sink element */
              if (element_type == ML_PIPELINE_ELEMENT_SINK ||
                  element_type == ML_PIPELINE_ELEMENT_APP_SINK) {
                gboolean sync = FALSE;

                g_object_get (G_OBJECT (elem), "sync", &sync, NULL);
                if (sync) {
                  _ml_logw (_ml_detail
                      ("It is recommended to apply 'sync=false' property to a sink element in most AI applications. Otherwise, inference results of large neural networks will be frequently dropped by the synchronization mechanism at the sink element."));
                }
              }

              if (element_type != ML_PIPELINE_ELEMENT_UNKNOWN) {
                ml_pipeline_element *e;

                e = construct_element (elem, pipe_h, name, element_type);
                if (e != NULL) {
                  if (g_str_equal (element_name, "tensor_if"))
                    process_tensor_if_option (e);
                  else if (g_str_equal (element_name, "tensor_filter"))
                    process_tensor_filter_option (e);

                  g_hash_table_insert (pipe_h->namednodes, g_strdup (name), e);
                } else {
                  /* allocation failure */
                  _ml_error_report_continue
                      ("Cannot allocate memory with construct_element().");
                  status = ML_ERROR_OUT_OF_MEMORY;
                  done = TRUE;
                }
              }

              g_free (name);
            }
          }

          g_value_reset (&item);