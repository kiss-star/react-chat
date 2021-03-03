/* SPDX-License-Identifier: Apache-2.0 */
/**
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file modules.c
 * @date 25 June 2022
 * @brief NNStreamer/Utilities C-API Wrapper.
 * @see	https://github.com/nnstreamer/api
 * @author Sangjung Woo <sangjung.woo@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <glib.h>
#include <stdio.h>

#include "common.h"
#include "modules.h"
#include "log.h"

static GList *module_head = NULL;

/**
 * @brief Add the specific DBus interface into the Machine Learning agent daemon.
 */
void
add_module (const struct module_ops *module)
{
  module_head = g_list_append (module_head, (gpointer) module);
}

/**
 * @brief Remove the specific DBus interface from the Machine Learning agent daemon.
 */
void
remove_module (const struct module_ops *module)
{
  module_head = g_list_remove (module_head, (gconstpointer) module