
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file service-db.cc
 * @date 29 Jul 2022
 * @brief Database implementation of NNStreamer/Service C-API
 * @see https://github.com/nnstreamer/api
 * @author Sangjung Woo <sangjung.woo@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <glib.h>

#include "service-db.hh"

#define sqlite3_clear_errmsg(m) do { if (m) { sqlite3_free (m); (m) = nullptr; } } while (0)

#define ML_DATABASE_PATH      DB_PATH"/.ml-service.db"
#define DB_KEY_PREFIX         MESON_KEY_PREFIX

/**
 * @brief The version of pipeline table schema. It should be a positive integer.
 */
#define TBL_VER_PIPELINE_DESCRIPTION (1)

/**
 * @brief The version of model table schema. It should be a positive integer.
 */
#define TBL_VER_MODEL_INFO (1)

typedef enum
{
  TBL_DB_INFO = 0,
  TBL_PIPELINE_DESCRIPTION = 1,
  TBL_MODEL_INFO = 2,

  TBL_MAX