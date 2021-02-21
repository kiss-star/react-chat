/* SPDX-License-Identifier: Apache-2.0 */
/**
 * NNStreamer API / Machine Learning Agent Daemon
 * Copyright (C) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
 */

/**
 * @file    gdbus-util.h
 * @date    25 June 2022
 * @brief   Internal GDbus utility header of Machine Learning agent daemon
 * @see     https://github.com/nnstreamer/api
 * @author  Sangjung Woo <sangjung.woo@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * @details
 *    This provides the wrapper functions to use DBus easily.
 */
#ifndef __GDBUS_UTIL_H__
#define __GDBUS_UTIL_H__

#include <glib.h>
#include <gio/gio.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief DBus signal handler information to connect
 */
struct gdbus_signal_info
{
  const gchar *signal_name; /**< specific signal name to handle */
  GCallback cb;         /**< Callback function to connect */
  gpointer cb_data;     /**< Data to pass to callback function */
  gulong handler_id;    /**< Connected handler ID */
};

/**
 * @brief Export the DBus interface at the Object path on the bus connection.
 * @param 