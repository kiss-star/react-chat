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
      bre