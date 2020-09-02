
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file nnstreamer-capi-util.c
 * @date 10 June 2019
 * @brief NNStreamer/Utilities C-API Wrapper.
 * @see	https://github.com/nnstreamer/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <glib.h>

#include "nnstreamer.h"
#include "ml-api-internal.h"

/**
 * @brief Allocates a tensors information handle with default value.
 */
int
ml_tensors_info_create (ml_tensors_info_h * info)
{
  ml_tensors_info_s *tensors_info;

  check_feature_state (ML_FEATURE);

  if (!info)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, info, is NULL. Provide a valid pointer.");

  *info = tensors_info = g_new0 (ml_tensors_info_s, 1);
  if (tensors_info == NULL)
    _ml_error_report_return (ML_ERROR_OUT_OF_MEMORY,
        "Failed to allocate the tensors info handle. Out of memory?");

  g_mutex_init (&tensors_info->lock);
  tensors_info->is_extended = false;

  /* init tensors info struct */
  return _ml_tensors_info_initialize (tensors_info);
}

/**
 * @brief Allocates an extended tensors information handle with default value.
 */
int
ml_tensors_info_create_extended (ml_tensors_info_h * info)
{
  ml_tensors_info_s *tensors_info;

  check_feature_state (ML_FEATURE);

  if (!info)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, info, is NULL. Provide a valid pointer.");

  *info = tensors_info = g_new0 (ml_tensors_info_s, 1);
  if (tensors_info == NULL)
    _ml_error_report_return (ML_ERROR_OUT_OF_MEMORY,
        "Failed to allocate the tensors info handle. Out of memory?");

  g_mutex_init (&tensors_info->lock);
  tensors_info->is_extended = true;

  /* init tensors info struct */
  return _ml_tensors_info_initialize (tensors_info);
}

/**
 * @brief Frees the given handle of a tensors information.
 */
int
ml_tensors_info_destroy (ml_tensors_info_h info)
{
  ml_tensors_info_s *tensors_info;

  check_feature_state (ML_FEATURE);

  tensors_info = (ml_tensors_info_s *) info;

  if (!tensors_info)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, info, is NULL. Provide a valid pointer.");

  G_LOCK_UNLESS_NOLOCK (*tensors_info);

  _ml_tensors_info_free (tensors_info);
  G_UNLOCK_UNLESS_NOLOCK (*tensors_info);
  g_mutex_clear (&tensors_info->lock);
  g_free (tensors_info);

  return ML_ERROR_NONE;
}

/**
 * @brief Initializes the tensors information with default value.
 */
int
_ml_tensors_info_initialize (ml_tensors_info_s * info)
{
  guint i, j;

  if (!info)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, info, is NULL. Provide a valid pointer.");

  info->num_tensors = 0;

  for (i = 0; i < ML_TENSOR_SIZE_LIMIT; i++) {
    info->info[i].name = NULL;
    info->info[i].type = ML_TENSOR_TYPE_UNKNOWN;

    for (j = 0; j < ML_TENSOR_RANK_LIMIT; j++) {
      info->info[i].dimension[j] = 0;
    }
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Compares the given tensor info.
 */
static gboolean
ml_tensor_info_compare (const ml_tensor_info_s * i1,
    const ml_tensor_info_s * i2, bool is_extended)
{
  guint i, valid_rank = ML_TENSOR_RANK_LIMIT;

  if (i1 == NULL || i2 == NULL)
    return FALSE;

  if (i1->type != i2->type)
    return FALSE;

  if (!is_extended)
    valid_rank = ML_TENSOR_RANK_LIMIT_PREV;

  for (i = 0; i < valid_rank; i++) {
    if (i1->dimension[i] != i2->dimension[i])
      return FALSE;
  }

  return TRUE;
}

/**
 * @brief Validates the given tensor info is valid.
 * @note info should be locked by caller if nolock == 0.
 */
static gboolean
ml_tensor_info_validate (const ml_tensor_info_s * info, bool is_extended)
{
  guint i;

  if (!info)
    return FALSE;

  if (info->type < 0 || info->type >= ML_TENSOR_TYPE_UNKNOWN)
    return FALSE;

  for (i = 0; i < ML_TENSOR_RANK_LIMIT; i++) {
    if (info->dimension[i] == 0)
      return FALSE;
  }

  if (!is_extended) {
    for (i = ML_TENSOR_RANK_LIMIT_PREV; i < ML_TENSOR_RANK_LIMIT; i++) {
      if (info->dimension[i] != 1)
        return FALSE;
    }
  }

  return TRUE;
}

/**
 * @brief Validates the given tensors info is valid without acquiring lock
 * @note This function assumes that lock on ml_tensors_info_h has already been acquired
 */
static int
_ml_tensors_info_validate_nolock (const ml_tensors_info_s * info, bool *valid)
{
  guint i;

  G_VERIFYLOCK_UNLESS_NOLOCK (*info);
  /* init false */
  *valid = false;

  if (info->num_tensors < 1) {
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The given tensors_info to be validated has invalid num_tensors (%u). It should be 1 or more.",
        info->num_tensors);
  }

  for (i = 0; i < info->num_tensors; i++) {
    if (!ml_tensor_info_validate (&info->info[i], info->is_extended))
      goto done;
  }

  *valid = true;

done:
  return ML_ERROR_NONE;
}

/**
 * @brief Validates the given tensors info is valid.
 */
int
ml_tensors_info_validate (const ml_tensors_info_h info, bool *valid)
{
  ml_tensors_info_s *tensors_info;
  int ret = ML_ERROR_NONE;

  check_feature_state (ML_FEATURE);

  if (!valid)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The data-return parameter, valid, is NULL. It should be a pointer pre-allocated by the caller.");

  tensors_info = (ml_tensors_info_s *) info;

  if (!tensors_info)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The input parameter, tensors_info, is NULL. It should be a valid ml_tensors_info_h, which is usually created by ml_tensors_info_create().");

  G_LOCK_UNLESS_NOLOCK (*tensors_info);

  ret = _ml_tensors_info_validate_nolock (info, valid);

  G_UNLOCK_UNLESS_NOLOCK (*tensors_info);
  return ret;
}

/**
 * @brief Compares the given tensors information.
 */
int
_ml_tensors_info_compare (const ml_tensors_info_h info1,
    const ml_tensors_info_h info2, bool *equal)
{
  ml_tensors_info_s *i1, *i2;
  guint i;

  check_feature_state (ML_FEATURE);

  if (info1 == NULL)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The input parameter, info1, should be a valid ml_tensors_info_h handle, which is usually created by ml_tensors_info_create(). However, info1 is NULL.");
  if (info2 == NULL)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The input parameter, info2, should be a valid ml_tensors_info_h handle, which is usually created by ml_tensors_info_create(). However, info2 is NULL.");
  if (equal == NULL)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The output parameter, equal, should be a valid pointer allocated by the caller. However, equal is NULL.");

  i1 = (ml_tensors_info_s *) info1;
  G_LOCK_UNLESS_NOLOCK (*i1);
  i2 = (ml_tensors_info_s *) info2;
  G_LOCK_UNLESS_NOLOCK (*i2);

  /* init false */
  *equal = false;

  if (i1->num_tensors != i2->num_tensors)
    goto done;

  if (i1->is_extended != i2->is_extended)
    goto done;

  for (i = 0; i < i1->num_tensors; i++) {
    if (!ml_tensor_info_compare (&i1->info[i], &i2->info[i], i1->is_extended))
      goto done;
  }

  *equal = true;

done:
  G_UNLOCK_UNLESS_NOLOCK (*i2);
  G_UNLOCK_UNLESS_NOLOCK (*i1);
  return ML_ERROR_NONE;
}

/**
 * @brief Sets the number of tensors with given handle of tensors information.
 */
int
ml_tensors_info_set_count (ml_tensors_info_h info, unsigned int count)
{
  ml_tensors_info_s *tensors_info;

  check_feature_state (ML_FEATURE);

  if (!info)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, info, is NULL. It should be a valid ml_tensors_info_h handle, which is usually created by ml_tensors_info_create().");
  if (count > ML_TENSOR_SIZE_LIMIT || count == 0)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, count, is the number of tensors, which should be between 1 and 16. The given count is %u.",
        count);

  tensors_info = (ml_tensors_info_s *) info;

  /* This is atomic. No need for locks */
  tensors_info->num_tensors = count;

  return ML_ERROR_NONE;
}

/**
 * @brief Gets the number of tensors with given handle of tensors information.
 */
int
ml_tensors_info_get_count (ml_tensors_info_h info, unsigned int *count)
{
  ml_tensors_info_s *tensors_info;

  check_feature_state (ML_FEATURE);

  if (!info)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The paramter, info, is NULL. It should be a valid ml_tensors_info_h handle, which is usually created by ml_tensors_info_create().");
  if (!count)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, count, is NULL. It should be a valid unsigned int * pointer, allocated by the caller.");

  tensors_info = (ml_tensors_info_s *) info;
  /* This is atomic. No need for locks */
  *count = tensors_info->num_tensors;

  return ML_ERROR_NONE;
}

/**
 * @brief Sets the tensor name with given handle of tensors information.
 */
int
ml_tensors_info_set_tensor_name (ml_tensors_info_h info,
    unsigned int index, const char *name)
{
  ml_tensors_info_s *tensors_info;

  check_feature_state (ML_FEATURE);

  if (!info)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, info, is NULL. It should be a valid ml_tensors_info_h handle, which is usually created by ml_tensors_info_create().");

  tensors_info = (ml_tensors_info_s *) info;
  G_LOCK_UNLESS_NOLOCK (*tensors_info);

  if (tensors_info->num_tensors <= index) {
    G_UNLOCK_UNLESS_NOLOCK (*tensors_info);
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, index, is too large, it should be smaller than the number of tensors, given by info. info says num_tensors is %u and index is %u.",
        tensors_info->num_tensors, index);
  }

  if (tensors_info->info[index].name) {
    g_free (tensors_info->info[index].name);
    tensors_info->info[index].name = NULL;
  }

  if (name)
    tensors_info->info[index].name = g_strdup (name);

  G_UNLOCK_UNLESS_NOLOCK (*tensors_info);
  return ML_ERROR_NONE;
}

/**
 * @brief Gets the tensor name with given handle of tensors information.
 */
int
ml_tensors_info_get_tensor_name (ml_tensors_info_h info,
    unsigned int index, char **name)
{
  ml_tensors_info_s *tensors_info;

  check_feature_state (ML_FEATURE);

  if (!info)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, info, is NULL. It should be a valid ml_tensors_info_h handle, which is usually created by ml_tensors_info_create().");
  if (!name)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, name, is NULL. It should be a valid char ** pointer, allocated by the caller. E.g., char *name; ml_tensors_info_get_tensor_name (info, index, &name);");

  tensors_info = (ml_tensors_info_s *) info;
  G_LOCK_UNLESS_NOLOCK (*tensors_info);

  if (tensors_info->num_tensors <= index) {
    G_UNLOCK_UNLESS_NOLOCK (*tensors_info);
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, index, is too large. It should be smaller than the number of tensors, given by info. info says num_tensors is %u and index is %u.",
        tensors_info->num_tensors, index);
  }

  *name = g_strdup (tensors_info->info[index].name);

  G_UNLOCK_UNLESS_NOLOCK (*tensors_info);
  return ML_ERROR_NONE;
}

/**
 * @brief Sets the tensor type with given handle of tensors information.
 */
int
ml_tensors_info_set_tensor_type (ml_tensors_info_h info,
    unsigned int index, const ml_tensor_type_e type)
{
  ml_tensors_info_s *tensors_info;

  check_feature_state (ML_FEATURE);

  if (!info)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, info, is NULL. It should be a valid pointer of ml_tensors_info_h, which is usually created by ml_tensors_info_create().");

  if (type >= ML_TENSOR_TYPE_UNKNOWN || type < 0)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, type, ML_TENSOR_TYPE_UNKNOWN or out of bound. The value of type should be between 0 and ML_TENSOR_TYPE_UNKNOWN - 1. type = %d, ML_TENSOR_TYPE_UNKNOWN = %d.",
        type, ML_TENSOR_TYPE_UNKNOWN);

#ifndef FLOAT16_SUPPORT
  if (type == ML_TENSOR_TYPE_FLOAT16)
    _ml_error_report_return (ML_ERROR_NOT_SUPPORTED,
        "Float16 (IEEE 754) is not supported by the machine (or the compiler or your build configuration). You cannot configure ml_tensors_info instance with Float16 type.");
#endif
  /** @todo add BFLOAT16 when nnstreamer is ready for it. */

  tensors_info = (ml_tensors_info_s *) info;
  G_LOCK_UNLESS_NOLOCK (*tensors_info);

  if (tensors_info->num_tensors <= index) {
    G_UNLOCK_UNLESS_NOLOCK (*tensors_info);
    return ML_ERROR_INVALID_PARAMETER;
  }

  tensors_info->info[index].type = type;

  G_UNLOCK_UNLESS_NOLOCK (*tensors_info);
  return ML_ERROR_NONE;
}

/**
 * @brief Gets the tensor type with given handle of tensors information.
 */
int
ml_tensors_info_get_tensor_type (ml_tensors_info_h info,
    unsigned int index, ml_tensor_type_e * type)
{
  ml_tensors_info_s *tensors_info;

  check_feature_state (ML_FEATURE);

  if (!info)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, info, is NULL. It should be a valid ml_tensors_info_h handle, which is usually created by ml_tensors_info_create().");
  if (!type)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, type, is NULL. It should be a valid pointer of ml_tensor_type_e *, allocated by the caller. E.g., ml_tensor_type_e t; ml_tensors_info_get_tensor_type (info, index, &t);");

  tensors_info = (ml_tensors_info_s *) info;
  G_LOCK_UNLESS_NOLOCK (*tensors_info);

  if (tensors_info->num_tensors <= index) {
    G_UNLOCK_UNLESS_NOLOCK (*tensors_info);
    return ML_ERROR_INVALID_PARAMETER;
  }

  *type = tensors_info->info[index].type;

  G_UNLOCK_UNLESS_NOLOCK (*tensors_info);
  return ML_ERROR_NONE;
}

/**
 * @brief Sets the tensor dimension with given handle of tensors information.
 */
int
ml_tensors_info_set_tensor_dimension (ml_tensors_info_h info,
    unsigned int index, const ml_tensor_dimension dimension)
{
  ml_tensors_info_s *tensors_info;
  guint i;

  check_feature_state (ML_FEATURE);

  if (!info)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, info, is NULL. It should be a valid pointer of ml_tensors_info_h, which is usually created by ml_tensors_info_create().");

  tensors_info = (ml_tensors_info_s *) info;
  G_LOCK_UNLESS_NOLOCK (*tensors_info);

  if (tensors_info->num_tensors <= index) {
    G_UNLOCK_UNLESS_NOLOCK (*tensors_info);
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The number of tensors in 'info' parameter is %u, which is not larger than the given 'index' %u. Thus, we cannot get %u'th tensor from 'info'. Please set the number of tensors of 'info' correctly or check the value of the given 'index'.",
        tensors_info->num_tensors, index, index);
  }

  for (i = 0; i < ML_TENSOR_RANK_LIMIT; i++) {
    tensors_info->info[index].dimension[i] = dimension[i];
  }

  if (!tensors_info->is_extended) {
    for (i = ML_TENSOR_RANK_LIMIT_PREV; i < ML_TENSOR_RANK_LIMIT; i++) {
      tensors_info->info[index].dimension[i] = 1;
    }
  }

  G_UNLOCK_UNLESS_NOLOCK (*tensors_info);
  return ML_ERROR_NONE;
}

/**
 * @brief Gets the tensor dimension with given handle of tensors information.
 */
int
ml_tensors_info_get_tensor_dimension (ml_tensors_info_h info,
    unsigned int index, ml_tensor_dimension dimension)
{
  ml_tensors_info_s *tensors_info;
  guint i, valid_rank = ML_TENSOR_RANK_LIMIT;

  check_feature_state (ML_FEATURE);

  if (!info)
    _ml_error_report_return (ML_ERROR_INVALID_PARAMETER,
        "The parameter, info, is NULL. It should be a valid pointer of ml_tensors_info_h, which is usually created by ml_tensors_info_create().");

  tensors_info = (ml_tensors_info_s *) info;
  G_LOCK_UNLESS_NOLOCK (*tensors_info);

  if (tensors_info->num_tensors <= index) {
    G_UNLOCK_UNLESS_NOLOCK (*tensors_info);
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (!tensors_info->is_extended)
    valid_rank = ML_TENSOR_RANK_LIMIT_PREV;

  for (i = 0; i < valid_rank; i++) {
    dimension[i] = tensors_info->info[index].dimension[i];
  }

  G_UNLOCK_UNLESS_NOLOCK (*tensors_info);
  return ML_ERROR_NONE;
}