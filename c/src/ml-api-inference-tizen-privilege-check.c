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
} type_element;

/**
 * @brief Structure to parse ini file for mmfw elements.
 */
typedef struct _conf_detail
{
  int count;
  void **detail_info;
} conf_detail;

/**
 * @brief Structure to parse ini file for mmfw elements.
 */
typedef struct _camera_conf
{
  int type;
  conf_detail **info;
} camera_conf;

#define MMFW_CONFIG_MAIN_FILE "mmfw_camcorder.ini"

extern int
_mmcamcorder_conf_get_info (MMHandleType handle, int type, const char *ConfFile,
    camera_conf ** configure_info);

extern void
_mmcamcorder_conf_release_info (MMHandleType handle,
    camera_conf ** configure_info);

extern int
_mmcamcorder_conf_get_element (MMHandleType handle,
    camera_conf * configure_info, int category, const char *name,
    type_element ** element);

extern int
_mmcamcorder_conf_get_value_element_name (type_element * element,
    const char **value);
#endif /* TIZENMMCONF */

/**
 * @brief Internal structure for tizen mm framework.
 */
typedef struct
{
  gboolean invalid; /**< flag to indicate rm handle is valid */
  mm_resource_manager_h rm_h; /**< rm handle */
  device_policy_manager_h dpm_h; /**< dpm handle */
  int dpm_cb_id; /**< dpm callback id */
  gboolean has_video_src; /**< pipeline includes video src */
  gboolean has_audio_src; /**< pipeline includes audio src */
  GHashTable *res_handles; /**< hash table of resource handles */
} tizen_mm_handle_s;

/**
 * @brief Tizen resource type for multimedia.
 */
#define TIZEN_RES_MM "tizen_res_mm"

/**
 * @brief Tizen Privilege Camera (See https://www.tizen.org/privilege)
 */
#define TIZEN_PRIVILEGE_CAMERA "http://tizen.org/privilege/camera"

/**
 * @brief Tizen Privilege Recoder (See https://www.tizen.org/privilege)
 */
#define TIZEN_PRIVILEGE_RECODER "http://tizen.org/privilege/recorder"

/** The following functions are either not used or supported in Tizen 4 */
#if TIZEN5PLUS
#if TIZENPPM
/**
 * @brief Function to check tizen privilege.
 */
static int
ml_tizen_check_privilege (const gchar * privilege)
{
  int status = ML_ERROR_NONE;
  ppm_check_result_e priv_result;
  int err;

  /* check privilege */
  err = ppm_check_permission (privilege, &priv_result);
  if (err == PRIVACY_PRIVILEGE_MANAGER_ERROR_NONE &&
      priv_result == PRIVACY_PRIVILEGE_MANAGER_CHECK_RESULT_ALLOW) {
    /* privilege allowed */
  } else {
    _ml_loge ("Failed to check the privilege %s.", privilege);
    status = ML_ERROR_PERMISSION_DENIED;
  }

  return status;
}
#else
#define ml_tizen_check_privilege(...) (ML_ERROR_NONE)
#endif /* TIZENPPM */

/**
 * @brief Function to check device policy.
 */
static int
ml_tizen_check_dpm_restriction (device_policy_manager_h dpm_handle, int type)
{
  int err = DPM_ERROR_NOT_PERMITTED;
  int dpm_is_allowed = 0;

  switch (type) {
    case 1:                    /* camera */
      err = dpm_restriction_get_camera_state (dpm_handle, &dpm_is_allowed);
      break;
    case 2:                    /* mic */
      err = dpm_restriction_get_microphone_state (dpm_handle, &dpm_is_allowed);
      break;
    default:
      /* unknown type */
      break;
  }

  if (err != DPM_ERROR_NONE || dpm_is_allowed != 1) {
    _ml_loge ("Failed, device policy is not allowed.");
    return ML_ERROR_PERMISSION_DENIED;
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Callback to be called when device policy is changed.
 */
static void
ml_tizen_dpm_policy_changed_cb (const char *name, const char *state,
    void *user_data)
{
  ml_pipeline *p;

  g_return_if_fail (state);
  g_return_if_fail (user_data);

  p = (ml_pipeline *) user_data;

  if (g_ascii_strcasecmp (state, "disallowed") == 0) {
    g_mutex_lock (&p->lock);

    /* pause the pipeline */
    gst_element_set_state (p->element, GST_STATE_PAUSED);

    g_mutex_unlock (&p->lock);
  }

  return;
}

/**
 * @brief Function to get key string of resource type to handle hash table.
 */
static gchar *
ml_tizen_mm_res_get_key_string (mm_resource_manager_res_type_e type)
{
  gchar *res_key = NULL;

  switch (type) {
    case MM_RESOURCE_MANAGER_RES_TYPE_VIDEO_DECODER:
      res_key = g_strdup ("tizen_mm_res_video_decoder");
      break;
    case MM_RESOURCE_MANAGER_RES_TYPE_VIDEO_OVERLAY:
      res_key = g_strdup ("tizen_mm_res_video_overlay");
      break;
    case MM_RESOURCE_MANAGER_RES_TYPE_CAMERA:
      res_key = g_strdup ("tizen_mm_res_camera");
      break;
    case MM_RESOURCE_MANAGER_RES_TYPE_VIDEO_ENCODER:
      res_key = g_strdup ("tizen_mm_res_video_encoder");
      break;
    case MM_RESOURCE_MANAGER_RES_TYPE_RADIO:
      res_key = g_strdup ("tizen_mm_res_radio");
      break;
    default:
      _ml_logw ("The resource type %d is invalid.", type);
      break;
  }

  return res_key;
}

/**
 * @brief Function to get resource type from key string to handle hash table.
 */
static mm_resource_manager_res_type_e
ml_tizen_mm_res_get_type (const gchar * res_key)
{
  mm_resource_manager_res_type_e type = MM_RESOURCE_MANAGER_RES_TYPE_MAX;

  g_return_val_if_fail (res_key, type);

  if (g_str_equal (res_key, "tizen_mm_res_video_decoder")) {
    type = MM_RESOURCE_MANAGER_RES_TYPE_VIDEO_DECODER;
  } else if (g_str_equal (res_key, "tizen_mm_res_video_overlay")) {
    type = MM_RESOURCE_MANAGER_RES_TYPE_VIDEO_OVERLAY;
  } else if (g_str_equal (res_key, "tizen_mm_res_camera")) {
    type = MM_RESOURCE_MANAGER_RES_TYPE_CAMERA;
  } else if (g_str_equal (res_key, "tizen_mm_res_video_encoder")) {
    type = MM_RESOURCE_MANAGER_RES_TYPE_VIDEO_ENCODER;
  } else if (g_str_equal (res_key, "tizen_mm_res_radio")) {
    type = MM_RESOURCE_MANAGER_RES_TYPE_RADIO;
  }

  return type;
}

/**
 * @brief Callback to be called from mm resource manager.
 */
static int
ml_tizen_mm_res_release_cb (mm_resource_manager_h rm,
    mm_resource_manager_res_h resource_h, void *user_data)
{
  ml_pipeline *p;
  pipeline_resource_s *res;
  tizen_mm_handle_s *mm_handle;

  g_return_val_if_fail (user_data, FALSE);
  p = (ml_pipeline *) user_data;
  g_mutex_lock (&p->lock);

  res =
      (pipeline_resource_s *) g_hash_table_lookup (p->resources, TIZEN_RES_MM);
  if (!res) {
    /* rm handle is not registered or removed */
    goto done;
  }

  mm_handle = (tizen_mm_handle_s *) res->handle;
  if (!mm_handle) {
    /* supposed the rm handle is already released */
    goto done;
  }

  /* pause pipeline */
  gst_element_set_state (p->element, GST_STATE_PAUSED);
  mm_handle->invalid = TRUE;

done:
  g_mutex_unlock (&p->lock);
  return FALSE;
}

/**
 * @brief Callback to be called from mm resource manager.
 */
static void
ml_tizen_mm_res_status_cb (mm_resource_manager_h rm,
    mm_resource_manager_status_e status, void *user_data)
{
  ml_pipeline *p;
  pipeline_resource_s *res;
  tizen_mm_handle_s *mm_handle;

  g_return_if_fail (user_data);

  p = (ml_pipeline *) user_data;
  g_mutex_lock (&p->lock);

  res =
      (pipeline_resource_s *) g_hash_table_lookup (p->resources, TIZEN_RES_MM);
  if (!res) {
    /* rm handle is not registered or removed */
    goto done;
  }

  mm_handle = (tizen_mm_handle_s *) res->handle;
  if (!mm_handle) {
    /* supposed the rm handle is already released */
    goto done;
  }

  switch (status) {
    case MM_RESOURCE_MANAGER_STATUS_DISCONNECTED:
      /* pause pipeline, rm handle should be released */
      gst_element_set_state (p->element, GST_STATE_PAUSED);
      mm_handle->invalid = TRUE;
      break;
    default:
      break;
  }

done:
  g_mutex_unlock (&p->lock);
}

/**
 * @brief Function to get the handle of resource type.
 */
static int
ml_tizen_mm_res_get_handle (mm_resource_manager_h rm,
    mm_resource_manager_res_type_e res_type, gpointer * handle)
{
  mm_resource_manager_res_h rm_res_h;
  int err;

  /* add resource handle */
  err = mm_resource_manager_mark_for_acquire (rm, res_type,
      MM_RESOURCE_MANAGER_RES_VOLUME_FULL, &rm_res_h);
  if (err != MM_RESOURCE_MANAGER_ERROR_NONE)
    _ml_error_report_return (ML_ERROR_STREAMS_PIPE,
        "Internal error of Tizen multimedia resource manager: mm_resource_manager_mark_for_acquire () cannot acquire resources. It has returned %d.",
        err);

  err = mm_resource_manager_commit (rm);
  if (err != MM_RESOURCE_MANAGER_ERROR_NONE)
    _ml_error_report_return (ML_ERROR_STREAMS_PIPE,
        "Internal error of Tizen multimedia resource manager: mm_resource_manager_commit has failed with error code: %d",
        err);

  *handle = rm_res_h;
  return ML_ERROR_NONE;
}

/**
 * @brief Function to release the resource handle of tizen mm resource manager.
 */
static void
ml_tizen_mm_res_release (gpointer handle, gboolean destroy)
{
  tizen_mm_handle_s *mm_handle;

  g_return_if_fail (handle);

  mm_handle = (tizen_mm_handle_s *) handle;

  /* release res handles */
  if (g_hash_table_size (mm_handle->res_handles)) {
    GHashTableIter iter;
    gpointer key, value;
    gboolean marked = FALSE;

    g_hash_table_iter_init (&iter, mm_handle->res_handles);
    while (g_hash_table_iter_next (&iter, &key, &value)) {
      pipeline_resource_s *mm_res = value;

      if (mm_res->handle) {
        mm_resource_manager_mark_for_release (mm_handle->rm_h, mm_res->handle);
        mm_res->handle = NULL;
        marked = TRUE;
      }

      if (destroy)
        g_free (mm_res->type);
    }

    if (marked)
      mm_resource_manager_commit (mm_handle->rm_h);
  }

  mm_resource_manager_set_status_cb (mm_handle->rm_h, NULL, NULL);
  mm_resource_manager_destroy (mm_handle->rm_h);
  mm_handle->rm_h = NULL;

  mm_handle->invalid = FALSE;

  if (destroy) {
    if (mm_handle->dpm_h) {
      if (mm_handle->dpm_cb_id > 0) {
        dpm_remove_policy_changed_cb (mm_handle->dpm_h, mm_handle->dpm_cb_id);
        mm_handle->dpm_cb_id = 0;
      }

      dpm_manager_destroy (mm_handle->dpm_h);
      mm_handle->dpm_h = NULL;
    }

    g_hash_table_remove_all (mm_handle->res_handles);
    g_free (mm_handle);
  }
}

/**
 * @brief Function to initialize mm resource manager.
 */
static int
ml_tizen_mm_res_initialize (ml_pipeline_h pipe, gboolean has_video_src,
    gboolean has_audio_src)
{
  ml_pipeline *p;
  pipeline_resource_s *res;
  tizen_mm_handle_s *mm_handle = NULL;
  int status = ML_ERROR_STREAMS_PIPE;

  p = (ml_pipeline *) pipe;

  res =
      (pipeline_resource_s *) g_hash_table_lookup (p->resources, TIZEN_RES_MM);

  /* register new resource handle of tizen mmfw */
  if (!res) {
    res = g_new0 (pipeline_resource_s, 1);
    if (!res) {
      _ml_loge ("Failed to allocate pipeline resource handle.");
      status = ML_ERROR_OUT_OF_MEMORY;
      goto rm_error;
    }

    res->type = g_strdup (TIZEN_RES_MM);
    g_hash_table_insert (p->resources, g_strdup (TIZEN_RES_MM), res);
  }

  mm_handle = (tizen_mm_handle_s *) res->handle;
  if (!mm_handle) {
    mm_handle = g_new0 (tizen_mm_handle_s, 1);
    if (!mm_handle) {
      _ml_loge ("Failed to allocate media resource handle.");
      status = ML_ERROR_OUT_OF_MEMORY;
      goto rm_error;
    }

    mm_handle->res_handles =
        g_hash_table_new_full (g_str_hash, g_str_equal, g_free, NULL);

    /* device policy manager */
    mm_handle->dpm_h = dpm_manager_create ();
    if (dpm_add_policy_changed_cb (mm_handle->dpm_h, "camera",
            ml_tizen_dpm_policy_changed_cb, pipe,
            &mm_handle->dpm_cb_id) != DPM_ERROR_NONE) {
      _ml_loge ("Failed to add device policy callback.");
      status = ML_ERROR_PERMISSION_DENIED;
      goto rm_error;
    }

    /* set mm handle */
    res->handle = mm_handle;
  }

  mm_handle->has_video_src = has_video_src;
  mm_handle->has_audio_src = has_audio_src;
  status = ML_ERROR_NONE;

rm_error:
  if (status != ML_ERROR_NONE) {
    /* failed to initialize mm handle */
    if (mm_handle)
      ml_tizen_mm_res_release (mm_handle, TRUE);
  }

  return status;
}

/**
 * @brief Function to acquire the resource from mm resource manager.
 */
static int
ml_tizen_mm_res_acquire (ml_pipeline_h pipe,
    mm_resource_manager_res_type_e res_type)
{
  ml_pipeline *p;
  pipeline_resource_s *res;
  tizen_mm_handle_s *mm_handle;
  gchar *res_key;
  int status = ML_ERROR_STREAMS_PIPE;
  int err;

  p = (ml_pipeline *