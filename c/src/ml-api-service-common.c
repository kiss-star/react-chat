
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file ml-api-service-common.c
 * @date 31 Aug 2022
 * @brief Some implementation of NNStreamer/Service C-API
 * @see https://github.com/nnstreamer/nnstreamer
 * @author Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include "ml-api-internal.h"
#include "ml-api-service.h"
#include "ml-api-service-private.h"

/**
 * @brief Internal function to get proxy of the pipeline d-bus interface
 */
MachinelearningServicePipeline *
_get_mlsp_proxy_new_for_bus_sync (void)
{
  MachinelearningServicePipeline *mlsp;

  /** @todo deal with GError */
  mlsp = machinelearning_service_pipeline_proxy_new_for_bus_sync
      (G_BUS_TYPE_SYSTEM, G_DBUS_PROXY_FLAGS_NONE,
      "org.tizen.machinelearning.service",
      "/Org/Tizen/MachineLearning/Service/Pipeline", NULL, NULL);

  if (mlsp)
    return mlsp;

  /** Try with session type */
  mlsp = machinelearning_service_pipeline_proxy_new_for_bus_sync
      (G_BUS_TYPE_SESSION, G_DBUS_PROXY_FLAGS_NONE,
      "org.tizen.machinelearning.service",
      "/Org/Tizen/MachineLearning/Service/Pipeline", NULL, NULL);

  return mlsp;
}

/**
 * @brief Internal function to get proxy of the model d-bus interface
 */
MachinelearningServiceModel *
_get_mlsm_proxy_new_for_bus_sync (void)