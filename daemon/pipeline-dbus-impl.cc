/* SPDX-License-Identifier: Apache-2.0 */
/**
 * NNStreamer API / Machine Learning Agent Daemon
 * Copyright (C) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
 */

/**
 * @file    pipeline-dbus-impl.cc
 * @date    20 Jul 2022
 * @brief   Implementation of pipeline dbus interface.
 * @see     https://github.com/nnstreamer/api
 * @author  Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @bug     No known bugs except for NYI items
 * @details
 *    This implements the pipeline dbus interface.
 */

#include <glib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <gst/gst.h>

#include <common.h>
#include <modules.h>
#include <gdbus-util.h>
#include <log.h>

#include "pipeline-dbus.h"
#include "service-db.hh"
#include "dbus-interface.h"

static MachinelearningServicePipeline *g_gdbus_instance = NULL;
static GHashTable *pipeline_table = NULL;
G_LOCK_DEFINE_STATIC (pipeline_table_lock);

/**
 * @brief Structure for pipeline.
 */
typedef struct _pipeline
{
  GstElement *element;
  gint64 id;
  GMutex lock;
  gchar *service_name;
  gchar *description;
} pipeline_s;

/**
 * @brief Internal function to destroy pipeline instances.
 */
static void
_pipeline_free (gpointer data)
{
  pipeline_s *p;

  if (!data) {
    _E ("internal error, the data should not be NULL");
    return;
  }

  p = (pipeline_s *) data;

  if (p->element)
    gst_object_unref (p->element);

  g_free (p->service_name);
  g_free (p->description);
  g_mutex_clear (&p->lock);

  g_free (p);
}

/**
 * @brief Get the skeleton object of the DBus interface.
 */
static MachinelearningServicePipeline *
gdbus_get_pipeline_instance (void)
{
  return machinelearning_service_pipeline_skeleton_new ();
}

/**
 * @brief Put the obtained skeleton object and release the resource.
 */
static void
gdbus_put_pipeline_instance (MachinelearningServicePipeline **instance)
{
  g_clear_object (instance);
}

/**
 * @brief Set the service with given description. Return the call result.
 */
static gboolean
dbus_cb_core_set_pipeline (MachinelearningServicePipeline *obj,
    GDBusMethodInvocation *invoc, const gchar *service_name,
    const gchar *pipeline_desc, gpointer user_data)
{
  gint result = 0;
  MLServiceDB &db = MLServiceDB::getInstance ();

  try {
    db.connectDB ();
    db.set_pipeline (service_name, pipeline_desc);
  } catch (const std::invalid_argument &e) {
    _E ("An exception occurred during write to the DB. Error message: %s", e.what ());
    result = -EINVAL;
  } catch (const std::exception &e) {
    _E ("An exception occurred during write to the DB. Error messa