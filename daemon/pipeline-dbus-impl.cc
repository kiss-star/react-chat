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

static MachinelearningServicePipeline 